"""ChromaDB cache must follow ontology changes (issue #65)."""

import sys
from unittest.mock import Mock

import numpy as np
import pytest

from edam_mcp.ontology.matcher import ConceptMatcher


def _concept(label, definition):
    return {"label": label, "definition": definition, "synonyms": [], "type": "Format"}


def test_chromadb_cache_follows_ontology_changes(tmp_path):
    model = Mock()
    model.encode.side_effect = lambda texts, **_: np.random.rand(len(texts), 4)
    loader = Mock()
    loader.concepts = {"a": _concept("A", "old text"), "b": _concept("B", "kept"), "c": _concept("C", "removed")}

    matcher = ConceptMatcher(loader)
    matcher.use_chromadb = True
    matcher.chroma_db = str(tmp_path / "default.db")
    matcher.embedding_model = model
    matcher._build_embeddings()

    loader.concepts = {"a": _concept("A", "new text"), "b": _concept("B", "kept"), "d": _concept("D", "added")}
    matcher._build_embeddings()

    import chromadb

    stored = chromadb.PersistentClient(path=matcher.chroma_db).get_collection("concept_embeddings").get()
    assert sorted(stored["ids"]) == ["a", "b", "d"]
    assert "new text" in dict(zip(stored["ids"], stored["documents"]))["a"]
    # Second build only embeds the changed and the new concept
    assert len(model.encode.call_args_list[1].args[0]) == 2


WORDS = ["alpha", "beta", "gamma"]


def _fake_model():
    """One-hot embedding on the first known word in the text, so matches are deterministic."""

    def vec(text):
        return np.eye(3)[next((i for i, w in enumerate(WORDS) if w in text.lower()), 0)]

    model = Mock()
    model.encode.side_effect = lambda x, **_: vec(x) if isinstance(x, str) else np.array([vec(t) for t in x])
    return model


def _matcher(tmp_path, use_chromadb, model):
    loader = Mock()
    loader.concepts = {w: _concept(w, f"{w} thing") for w in WORDS}
    loader.get_concept.side_effect = lambda uri: {**loader.concepts[uri], "parents": [], "children": []}
    matcher = ConceptMatcher(loader)
    matcher.use_chromadb = use_chromadb
    matcher.chroma_db = str(tmp_path / "default.db")
    matcher.embedding_model = model
    return matcher


@pytest.mark.parametrize("use_chromadb", [False, True])
def test_embeddings_built_once(tmp_path, use_chromadb):
    model = _fake_model()
    matcher = _matcher(tmp_path, use_chromadb, model)

    for _ in range(2):
        top = matcher.match_concepts("find the beta one", min_confidence=0.9)
        assert [m.concept_uri for m in top] == ["beta"]

    corpus_calls = [c for c in model.encode.call_args_list if not isinstance(c.args[0], str)]
    assert len(corpus_calls) == 1


def test_chromadb_reuse_skips_encoding(tmp_path):
    _matcher(tmp_path, True, _fake_model()).match_concepts("alpha")

    model = _fake_model()
    top = _matcher(tmp_path, True, model).match_concepts("gamma", min_confidence=0.9)

    assert [m.concept_uri for m in top] == ["gamma"]
    assert all(isinstance(c.args[0], str) for c in model.encode.call_args_list)


def test_chromadb_collection_has_no_default_embedding_function(tmp_path):
    import chromadb

    matcher = _matcher(tmp_path, True, _fake_model())
    matcher.match_concepts("alpha")

    config = chromadb.PersistentClient(path=matcher.chroma_db).get_collection("concept_embeddings").configuration_json
    assert config["embedding_function"]["type"] == "legacy"


def test_chromadb_missing_raises_clear_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "chromadb", None)  # makes `import chromadb` raise ImportError

    with pytest.raises(RuntimeError, match="chromadb not available"):
        _matcher(tmp_path, True, _fake_model()).match_concepts("alpha")


def test_chromadb_rebuilds_when_embedding_model_changes(tmp_path, monkeypatch):
    from edam_mcp.config import settings

    monkeypatch.setattr(settings, "embedding_model", "model-a")
    _matcher(tmp_path, True, _fake_model()).match_concepts("alpha")

    same = _fake_model()
    _matcher(tmp_path, True, same).match_concepts("alpha")
    assert all(isinstance(c.args[0], str) for c in same.encode.call_args_list)  # same model: cache reused

    monkeypatch.setattr(settings, "embedding_model", "model-b")
    changed = _fake_model()
    _matcher(tmp_path, True, changed).match_concepts("alpha")
    corpus = [c.args[0] for c in changed.encode.call_args_list if not isinstance(c.args[0], str)]
    assert len(corpus) == 1 and len(corpus[0]) == 3  # every concept re-encoded


def test_chromadb_rebuilds_unlabelled_legacy_collection(tmp_path):
    import chromadb

    chromadb.PersistentClient(path=str(tmp_path / "default.db")).create_collection("concept_embeddings")

    _matcher(tmp_path, True, _fake_model()).match_concepts("alpha")

    meta = chromadb.PersistentClient(path=str(tmp_path / "default.db")).get_collection("concept_embeddings").metadata
    assert meta["embedding_model"]
