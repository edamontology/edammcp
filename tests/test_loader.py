"""Deprecated EDAM classes are skipped unless EDAM_INCLUDE_DEPRECATED is set (issue #56)."""

import pickle

from edam_mcp.config import settings
from edam_mcp.ontology.loader import OntologyLoader

OWL = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#">
  <owl:Class rdf:about="http://edamontology.org/format_1"><rdfs:label>Live</rdfs:label></owl:Class>
  <owl:Class rdf:about="http://edamontology.org/format_2">
    <rdfs:label>Old</rdfs:label>
    <owl:deprecated>1.8</owl:deprecated>
    <owl:deprecated rdf:datatype="http://www.w3.org/2001/XMLSchema#boolean">true</owl:deprecated>
  </owl:Class>
</rdf:RDF>
"""


def test_deprecated_concepts_are_opt_in(tmp_path, monkeypatch):
    owl = tmp_path / "edam.owl"
    owl.write_text(OWL)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))

    def load():
        loader = OntologyLoader(ontology_url=str(owl))
        assert loader.load_ontology()
        return sorted(c["label"] for c in loader.concepts.values())

    assert load() == ["Live"]
    monkeypatch.setattr(settings, "include_deprecated", True)
    owl.unlink()  # second load must come from the cache, which kept both
    assert load() == ["Live", "Old"]


def test_legacy_cache_is_rebuilt_without_stale_concepts(tmp_path, monkeypatch):
    owl = tmp_path / "edam.owl"
    owl.write_text(OWL)
    monkeypatch.setattr(settings, "cache_dir", str(tmp_path / "cache"))
    # Pre-#56 cache: no deprecated flag, and a concept the current OWL no longer has
    concepts_path, types_path, ts_path = OntologyLoader()._cache_paths()
    gone = {"label": "Gone", "definition": None, "synonyms": [], "type": "Format", "parents": [], "children": []}
    with open(concepts_path, "wb") as f:
        pickle.dump({"http://edamontology.org/format_9": gone}, f)
    with open(types_path, "wb") as f:
        pickle.dump({"Format"}, f)
    open(ts_path, "w").close()

    for _ in range(2):  # the rebuilt cache must also load cleanly
        loader = OntologyLoader(ontology_url=str(owl))
        assert loader.load_ontology()
        assert [c["label"] for c in loader.concepts.values()] == ["Live"]


def test_related_follows_every_parent_and_skips_deprecated():
    op = "http://edamontology.org/operation_"
    loader = OntologyLoader()
    loader.concepts = {
        op + "1": {"deprecated": False, "parents": [], "children": [op + "2", op + "9"]},
        op + "2": {"deprecated": False, "parents": [op + "1"], "children": [op + "3"]},
        op + "3": {"deprecated": False, "parents": [op + "2", op + "4"], "children": []},
        op + "4": {"deprecated": False, "parents": [], "children": [op + "3"]},
        op + "9": {"deprecated": True, "parents": [op + "1"], "children": []},
    }
    assert loader.get_related(op + "1", descendant_depth=3) == {
        op + "1": ("exact", 0),
        op + "2": ("descendant", 1),
        op + "3": ("descendant", 2),
    }
    assert loader.get_related(op + "3", ancestor_depth=2) == {
        op + "3": ("exact", 0),
        op + "2": ("ancestor", 1),
        op + "4": ("ancestor", 1),
        op + "1": ("ancestor", 2),
    }
