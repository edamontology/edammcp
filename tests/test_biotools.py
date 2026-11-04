"""Tests for the Biotools query functionality."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from edam_mcp.models.biotools import (
    BiotoolsMatch,
    BiotoolsQueryExactRequest,
    BiotoolsQueryResponse,
    BiotoolsQueryTermsRequest,
    BiotoolsQueryTextRequest,
)
from edam_mcp.ontology import biotools_searcher
from edam_mcp.ontology.biotools_searcher import query_by_edam_terms, query_by_text
from edam_mcp.tools.biotools import (
    query_biotools_by_edam_terms,
    query_biotools_by_text,
    query_biotools_by_text_exact_match,
)
from edam_mcp.utils.context import MockContext

TOPIC = "http://edamontology.org/topic_3510"
DATA = "http://edamontology.org/data_1361"
FORMAT = "http://edamontology.org/format_1929"


def _tool(_id, topics=(), function=None):
    return {
        "biotoolsID": _id,
        "name": _id,
        "description": None,
        "topic": [{"uri": u, "term": "Phylogenetics"} for u in topics],
        "function": function or [],
    }


PAGES = {
    1: {
        "count": 4,
        "next": "?page=2",
        "list": [
            _tool("unannotated"),
            _tool("tool_a", topics=[TOPIC, TOPIC]),
        ],
    },
    2: {
        "count": 4,
        "next": None,
        "list": [
            _tool("tool_a", topics=[TOPIC]),  # duplicate across pages
            _tool(
                "tool_b",
                function=[
                    {
                        "input": [{"data": {"uri": DATA, "term": "Sequence motif"}, "format": [{"uri": FORMAT}]}],
                        "output": [{"format": [{"uri": FORMAT, "term": "FASTA"}]}],
                    }
                ],
            ),
        ],
    },
}


@pytest.fixture(autouse=True)
def fake_edam(monkeypatch):
    """Small in-memory EDAM so the tests never download or read the real ontology."""
    uris = [TOPIC, DATA, FORMAT, "http://edamontology.org/topic_0160"]
    concepts = {u: {"deprecated": False} for u in uris}
    concepts["http://edamontology.org/topic_0001"] = {"deprecated": True}
    monkeypatch.setattr(biotools_searcher.EDAM_LOADER, "concepts", concepts)
    monkeypatch.setattr(biotools_searcher, "_loaded", True)


def _fake_get(url, params, timeout):
    response = MagicMock()
    response.json.return_value = PAGES[params["page"]]
    return response


@pytest.fixture
def fake_api():
    with (
        patch("edam_mcp.ontology.biotools_searcher.requests.get", side_effect=_fake_get) as get,
        patch("edam_mcp.ontology.biotools_searcher.sleep"),
    ):
        yield get


EXACT_CASE_NAME = {
    "count": 1,
    "next": None,
    "list": [
        _tool("tool_a", topics=[TOPIC, TOPIC]),
    ],
}
EXACT_CASE_NAME_EMPTY = {
    "count": 0,
    "next": None,
    "list": [],
}

EXACT_CASE_ID = PAGES[2]["list"][1]


def _fake_get_exact(url, params, timeout):
    if "name" in params:
        if params["name"].find("bwa") == -1:
            value = EXACT_CASE_NAME
        else:
            value = EXACT_CASE_NAME_EMPTY
    else:
        if url.find("bwa") == -1:
            value = EXACT_CASE_ID
        else:
            value = {"detail": "Not found."}

    response = MagicMock()
    response.json.return_value = value
    response.status_code = 404 if "detail" in value else 200
    return response


@pytest.fixture
def fake_api_exact():
    with (
        patch("edam_mcp.ontology.biotools_searcher.requests.get", side_effect=_fake_get_exact) as get,
        patch("edam_mcp.ontology.biotools_searcher.sleep"),
    ):
        yield get


class TestQueryBiotools:
    """Test cases for the bio.tools query tools."""

    @pytest.mark.asyncio
    async def test_query_by_text(self, fake_api):
        request = BiotoolsQueryTextRequest(seed="a&b=c", max_results=5)
        result = await query_biotools_by_text(request, MockContext())

        assert isinstance(result, BiotoolsQueryResponse)
        assert result.matches == [
            # kept, so an agent can tell a registered tool without EDAM terms from a missing one
            BiotoolsMatch(name="unannotated", description="", biotools_link="https://bio.tools/unannotated"),
            BiotoolsMatch(
                name="tool_a",
                description="",
                biotools_link="https://bio.tools/tool_a",
                topic_terms={TOPIC: {"label": "Phylogenetics", "obsolete": False}},
            ),
            BiotoolsMatch(
                name="tool_b",
                description="",
                biotools_link="https://bio.tools/tool_b",
                input_data_terms={DATA: {"label": "Sequence motif", "obsolete": False}},
                input_format_terms={FORMAT: {"label": None, "obsolete": False}},
                output_format_terms={FORMAT: {"label": "FASTA", "obsolete": False}},
            ),
        ]
        assert result.total_matches == 4  # bio.tools count, not len(matches)
        # seed goes through params (so it is URL-encoded) and pagination advances
        assert [c.kwargs["params"]["page"] for c in fake_api.call_args_list] == [1, 2]
        assert fake_api.call_args.kwargs["params"]["q"] == "a&b=c"
        assert fake_api.call_args.kwargs["params"]["sort"] == "score"

    @pytest.mark.asyncio
    async def test_query_by_text_exact_match(self, fake_api_exact):
        request = BiotoolsQueryExactRequest(name="tool_a")
        result = await query_biotools_by_text_exact_match(request, MockContext())
        assert isinstance(result, BiotoolsQueryResponse)
        assert result.total_matches == 1
        assert result.matches[0].name == "tool_a"

        request = BiotoolsQueryExactRequest(biotools_id="tool_b")
        result = await query_biotools_by_text_exact_match(request, MockContext())
        assert isinstance(result, BiotoolsQueryResponse)
        assert result.total_matches == 1
        assert result.matches[0].name == "tool_b"
        assert fake_api_exact.call_args.args[0] == "https://bio.tools/api/t/tool_b"

        request = BiotoolsQueryExactRequest(biotools_id="bwa")
        result = await query_biotools_by_text_exact_match(request, MockContext())
        assert isinstance(result, BiotoolsQueryResponse)
        assert result.total_matches == 0

        request = BiotoolsQueryExactRequest(name="bwa")
        result = await query_biotools_by_text_exact_match(request, MockContext())
        assert isinstance(result, BiotoolsQueryResponse)
        assert result.total_matches == 0

    def test_stops_at_max_results(self, fake_api):
        matches, total = query_by_text("x", max_results=1)
        assert (len(matches), total) == (1, 4)
        assert fake_api.call_count == 1

    @pytest.mark.asyncio
    async def test_query_by_terms(self, fake_api):
        request = BiotoolsQueryTermsRequest(edam_terms=[TOPIC, DATA])
        result = await query_biotools_by_edam_terms(request, MockContext())

        assert [m.name for m in result.matches] == ["unannotated", "tool_a", "tool_b"]
        assert len(result.matches) == 3

        # AND: both filters go in the same request
        params = fake_api.call_args_list[0].kwargs["params"]
        assert (params["topicID"], params["dataTypeID"]) == ('"topic_3510"', '"data_1361"')

    def test_invalid_terms(self):
        with pytest.raises(ValidationError):
            BiotoolsQueryTermsRequest(edam_terms=[])
        for uri in [
            "http://edamontology.org/identifier_1",
            "https://example.com/topic_3510",
            "not-a-uri/topic_3510",
            "http://edamontology.org/topic_bad",
        ]:
            with pytest.raises(ValueError, match=f"Unknown or obsolete EDAM terms: {uri}"):
                query_by_edam_terms([uri])

        with pytest.raises(ValueError, match="one topic term"):
            query_by_edam_terms([TOPIC, "http://edamontology.org/topic_0160"])

    def test_obsolete_terms(self):
        obsolete = "http://edamontology.org/topic_0001"
        with pytest.raises(ValueError, match="Unknown or obsolete"):
            query_by_edam_terms([obsolete])
        # in results, obsolete and unknown terms are flagged, valid ones are not
        terms = biotools_searcher._terms(
            [{"uri": obsolete, "term": "a"}, {"uri": "http://edamontology.org/topic_9999", "term": "b"}, {"uri": TOPIC}]
        )
        assert [t["obsolete"] for t in terms.values()] == [True, True, False]

    def test_depth_bounds(self):
        with pytest.raises(ValidationError):
            BiotoolsQueryTermsRequest(edam_terms=[TOPIC], descendant_depth=4)
        with pytest.raises(ValidationError):
            BiotoolsQueryTermsRequest(edam_terms=[TOPIC], ancestor_depth=3)


NARROW = "http://edamontology.org/topic_0160"


@pytest.fixture
def hierarchy(monkeypatch):
    """TOPIC has the narrower NARROW as child; DATA is unrelated."""
    concepts = {
        TOPIC: {"deprecated": False, "parents": [], "children": [NARROW]},
        NARROW: {"deprecated": False, "parents": [TOPIC], "children": []},
        DATA: {"deprecated": False, "parents": [], "children": []},
    }
    monkeypatch.setattr(biotools_searcher.EDAM_LOADER, "concepts", concepts)


def _annotated(_id, topics=(), data=()):
    functions = [{"input": [{"data": {"uri": d, "term": "x"}, "format": []} for d in data]}] if data else None
    return _tool(_id, topics=topics, function=functions)


# bio.tools filter (topicID) to its tools; the fake ignores every other filter
BY_TOPIC = {
    '"topic_3510"': [_annotated("t_exact", topics=[TOPIC], data=[DATA])],
    '"topic_0160"': [_annotated("t_narrow", topics=[NARROW], data=[DATA]), _annotated("t_no_data", topics=[NARROW])],
}


def _fake_by_topic(url, params, timeout):
    value = BY_TOPIC[params["topicID"]]
    response = MagicMock()
    response.json.return_value = {"count": len(value), "next": None, "list": value}
    return response


@pytest.fixture
def fake_by_topic(hierarchy):
    with (
        patch("edam_mcp.ontology.biotools_searcher.requests.get", side_effect=_fake_by_topic) as get,
        patch("edam_mcp.ontology.biotools_searcher.sleep"),
    ):
        yield get


def test_descendant_depth_queries_each_candidate_and_checks_other_terms(fake_by_topic):
    matches, total = query_by_edam_terms([TOPIC, DATA], descendant_depth=1)

    # one query per topic candidate (the pivot), DATA stays a server-side filter
    queried = [c.kwargs["params"] for c in fake_by_topic.call_args_list]
    assert [p["topicID"] for p in queried] == ['"topic_3510"', '"topic_0160"']
    assert all(p["dataTypeID"] == '"data_1361"' for p in queried)

    # t_no_data lacks DATA and is dropped; exact matches rank first
    assert [m.name for m in matches] == ["t_exact", "t_narrow"]
    assert total == 3
    assert matches[1].matched_via == {
        TOPIC: {"term": NARROW, "direction": "descendant", "distance": 1},
        DATA: {"term": DATA, "direction": "exact", "distance": 0},
    }


def test_expansion_is_capped_by_max_calls(fake_by_topic, monkeypatch):
    monkeypatch.setattr(biotools_searcher, "MAX_CALLS", 1)
    query_by_edam_terms([TOPIC], descendant_depth=1)
    assert fake_by_topic.call_count == 1


def test_no_depth_leaves_matched_via_unset(fake_by_topic):
    matches, _ = query_by_edam_terms([TOPIC])
    assert fake_by_topic.call_count == 1
    assert all(m.matched_via is None for m in matches)
