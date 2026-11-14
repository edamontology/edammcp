"""Score EDAM concept retrieval strategies on gold sets.

Gold sets: A = benchmark/*/qa.tsv, B = bio.tools sample (cached JSON), C = synonym hold-out.
Usage: uv run python benchmark/evaluate.py --concepts .cache/concepts.pkl \
           --strategies baseline-memory,baseline-chroma,numpy:minilm --gold A,C
Strategy grammar: baseline-memory | baseline-chroma | <kind>:<model>[:<text>] with
kind in numpy, tfidf, hybrid, rerank; model in minilm,bge-small,bge-base,sapbert,biolord;
text in cur (label+def+syn), rich (cur + parent labels), multi (max over label/syns/def vectors).
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import statistics as st
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from edam_mcp.utils.text_processing import preprocess_text  # noqa: E402

MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "biolord": "FremyCompany/BioLORD-2023",
}
KS = (1, 5, 10)
URI = re.compile(r"https?://edamontology\.org/\w+")


# ---------- gold sets ----------
def gold_a(concepts):
    rows = []
    for p in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*", "qa.tsv"))):
        with open(p, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                gold = {u for u in URI.findall(r["Answer"]) if u in concepts}
                if gold:
                    rows.append((r["Question"], gold))
    return rows


def gold_b(concepts, n_pages=4):
    cache = os.path.join(os.path.dirname(__file__), "biotools_sample.json")
    if not os.path.exists(cache):
        import requests

        tools = []
        for page in range(1, n_pages + 1):
            r = requests.get(
                "https://bio.tools/api/t/", params={"format": "json", "page": page, "sort": "score"}, timeout=60
            ).json()
            tools += r["list"]
        with open(cache, "w") as f:
            json.dump(tools, f)
    rows = []
    for t in json.load(open(cache)):
        gold = {x["uri"] for x in t.get("topic", [])}
        for fn in t.get("function", []):
            gold |= {x["uri"] for x in fn.get("operation", [])}
        gold = {u for u in gold if u in concepts}
        if gold and t.get("description"):
            rows.append((t["description"], gold))
    return rows


def gold_c(concepts):
    """Hold out each concept's first synonym: queried for, dropped from the index."""
    held = {u: c["synonyms"][0] for u, c in concepts.items() if c["synonyms"]}
    return [(s, {u}) for u, s in held.items()], held


# ---------- index text ----------
def concept_text(c, concepts, held, kind):
    syns = [s for s in c["synonyms"] if s != held.get(c["uri"])]
    parts = [c["label"], c["definition"] or "", *syns]
    if kind == "rich":
        parts += [concepts[p]["label"] for p in c["parents"] if p in concepts]
    return preprocess_text(" ".join(parts))


def multi_units(c, held):
    syns = [s for s in c["synonyms"] if s != held.get(c["uri"])]
    return [preprocess_text(x) for x in [c["label"], *syns, c["definition"] or ""] if x]


# ---------- strategies: .build(), .search(q, k) -> [uri] ----------
class Dense:
    def __init__(self, concepts, held, model, text):
        self.concepts, self.held, self.model_name, self.text = concepts, held, MODELS[model], text

    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    def build(self):
        self.model = self._model()
        self.uris = list(self.concepts)
        # One text per concept, aligned with self.uris (rerank input); "multi" falls back to "cur"
        self.docs = [concept_text(self.concepts[u], self.concepts, self.held, self.text) for u in self.uris]
        if self.text == "multi":
            units, owner = [], []
            for u in self.uris:
                for t in multi_units(self.concepts[u], self.held):
                    units.append(t)
                    owner.append(u)
            self.owner = np.array(owner)
            self.texts = units
        else:
            self.texts = [concept_text(self.concepts[u], self.concepts, self.held, self.text) for u in self.uris]
        self.mat = self.model.encode(self.texts, normalize_embeddings=True, show_progress_bar=False)

    def scores(self, q):
        v = self.model.encode(preprocess_text(q), normalize_embeddings=True, show_progress_bar=False)
        s = np.einsum("ij,j->i", self.mat, v)  # `@` contends with torch threads, see matcher.py
        if self.text == "multi":
            best = {}
            for u, x in zip(self.owner, s):
                if x > best.get(u, -2):
                    best[u] = x
            return np.array([best.get(u, -2) for u in self.uris])
        return s

    def search(self, q, k):
        s = self.scores(q)
        return [self.uris[i] for i in np.argsort(-s)[:k]]


class Tfidf:
    def __init__(self, concepts, held, text="cur"):
        self.concepts, self.held, self.text = concepts, held, text

    def build(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.uris = list(self.concepts)
        self.texts = [concept_text(self.concepts[u], self.concepts, self.held, self.text) for u in self.uris]
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.mat = self.vec.fit_transform(self.texts)

    def scores(self, q):
        return (self.mat @ self.vec.transform([preprocess_text(q)]).T).toarray().ravel()

    def search(self, q, k):
        return [self.uris[i] for i in np.argsort(-self.scores(q))[:k]]


class Hybrid:
    def __init__(self, dense, lex):
        self.d, self.l = dense, lex

    def build(self):
        self.d.build()
        self.l.build()
        self.uris = self.d.uris

    def search(self, q, k):
        score = {}
        for sc in (self.d.scores(q), self.l.scores(q)):
            for r, i in enumerate(np.argsort(-sc)[:100]):
                score[i] = score.get(i, 0) + 1 / (60 + r)
        return [self.uris[i] for i in sorted(score, key=score.get, reverse=True)[:k]]


class Rerank:
    def __init__(self, dense, model="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.d, self.model_name = dense, model

    def build(self):
        from sentence_transformers import CrossEncoder

        self.d.build()
        self.ce = CrossEncoder(self.model_name)
        self.uris = self.d.uris

    def search(self, q, k):
        top = np.argsort(-self.d.scores(q))[:30]
        s = self.ce.predict([(q, self.d.docs[i]) for i in top], show_progress_bar=False)
        return [self.uris[top[i]] for i in np.argsort(-s)[:k]]


class Baseline:
    """The real ConceptMatcher. Hold-out is not applied (gold C is then optimistic for it)."""

    def __init__(self, concepts, chroma):
        from edam_mcp.ontology.matcher import ConceptMatcher

        loader = SimpleNamespace(concepts=concepts, get_concept=concepts.get)
        self.m = ConceptMatcher(loader)
        self.m.use_chromadb = chroma
        self.m.chroma_db = os.path.join(tempfile.mkdtemp(), "eval.db")

    def build(self):
        self.m._build_embeddings()

    def search(self, q, k):
        return [x.concept_uri for x in self.m.match_concepts(q, max_results=k, min_confidence=0.0)]


def make(spec, concepts, held):
    if spec == "baseline-memory":
        return Baseline(concepts, False)
    if spec == "baseline-chroma":
        return Baseline(concepts, True)
    kind, model, *text = spec.split(":")
    text = text[0] if text else "cur"
    if kind == "numpy":
        return Dense(concepts, held, model, text)
    if kind == "tfidf":
        return Tfidf(concepts, held, text)
    if kind == "hybrid":
        return Hybrid(Dense(concepts, held, model, text), Tfidf(concepts, held, text))
    if kind == "rerank":
        return Rerank(Dense(concepts, held, model, text))
    raise SystemExit(f"unknown strategy {spec}")


# ---------- metrics ----------
def score(strategy, rows, concepts):
    lat, rec, hit, rr, rel = [], {k: [] for k in KS}, {k: [] for k in KS}, [], []
    for q, gold in rows:
        t = time.perf_counter()
        res = strategy.search(q, max(KS))
        lat.append((time.perf_counter() - t) * 1000)
        for k in KS:
            top = set(res[:k])
            rec[k].append(len(top & gold) / len(gold))
            hit[k].append(bool(top & gold))
        rr.append(next((1 / (i + 1) for i, u in enumerate(res) if u in gold), 0.0))
        near = set(gold)
        for g in gold:
            near |= set(concepts[g]["parents"]) | set(concepts[g]["children"])
        rel.append(bool(set(res[:5]) & near))
    m = {f"R@{k}": np.mean(rec[k]) for k in KS}
    m["hit@5"] = np.mean(hit[5])
    m["MRR"] = np.mean(rr)
    m["near@5"] = np.mean(rel)
    m["p50ms"] = st.median(lat)
    m["p95ms"] = float(np.percentile(lat, 95))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", default=".cache/concepts.pkl")
    ap.add_argument("--strategies", required=True)
    ap.add_argument("--gold", default="A,C")
    ap.add_argument("--limit", type=int, default=0, help="subsample each gold set (0 = all)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # pickle is the repo's own OntologyLoader cache (same trust as loader.py), not external data
    allc = pickle.load(open(a.concepts, "rb"))
    concepts = {u: c for u, c in allc.items() if not c["deprecated"]}
    golds = {}
    held = {}
    for g in a.gold.split(","):
        if g == "C":
            golds["C"], held = gold_c(concepts)
        elif g == "A":
            golds["A"] = gold_a(concepts)
        elif g == "B":
            golds["B"] = gold_b(concepts)
    if a.limit:
        rng = np.random.default_rng(0)
        golds = {k: [v[i] for i in rng.permutation(len(v))[: a.limit]] for k, v in golds.items()}
    print({k: len(v) for k, v in golds.items()}, file=sys.stderr)

    rows = []
    # A/B search the full index; only C searches the index without its held-out synonyms
    groups = [({}, [g for g in golds if g != "C"]), (held, [g for g in golds if g == "C"])]
    for spec in a.strategies.split(","):
        for hidden, names in groups:
            if not names:
                continue
            strat = make(spec, concepts, hidden)
            t = time.perf_counter()
            strat.build()
            build = time.perf_counter() - t
            for g in names:
                data = golds[g]
                m = score(strat, data, concepts)
                rows.append(
                    {
                        "strategy": spec,
                        "gold": g,
                        "build_s": round(build, 1),
                        **{k: round(float(v), 3) for k, v in m.items()},
                    }
                )
                print(rows[-1], file=sys.stderr)
    if rows:
        cols = list(rows[0])
        out = ["\t".join(cols)] + ["\t".join(str(r[c]) for c in cols) for r in rows]
        print("\n".join(out))
        if a.out:
            open(a.out, "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
