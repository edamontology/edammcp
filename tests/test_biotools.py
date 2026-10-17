"""Tests for the Biotools query functionality."""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from edam_mcp.models.biotools import (
    BiotoolsMatch,
    BiotoolsQueryResponse,
    BiotoolsQueryTermsRequest,
    BiotoolsQueryTextRequest,
    BiotoolsQueryExactRequest
)
from edam_mcp.ontology.biotools_searcher import query_by_edam_terms, query_by_text, query_by_text_exact_match

from edam_mcp.tools.biotools import query_biotools_by_edam_terms, query_biotools_by_text, query_biotools_by_text_exact_match

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
    if("name" in params):
        if( params["name"].find("bwa") == -1 ):
            value = EXACT_CASE_NAME
        else:
            value = EXACT_CASE_NAME_EMPTY
    else:
        if( url.find("bwa") == -1 ):
            value = EXACT_CASE_ID
        else:
            value = {"detail": "Not found."}

    response = MagicMock()
    response.json.return_value = value
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
                topic_terms={TOPIC: {"label": "Phylogenetics", "obsolete": False } },
            ),
            BiotoolsMatch(
                name="tool_b",
                description="",
                biotools_link="https://bio.tools/tool_b",
                input_data_terms={DATA: {"label": "Sequence motif", "obsolete": False } },
                input_format_terms={FORMAT: {"label": None, "obsolete": False } },
                output_format_terms={FORMAT: {"label": "FASTA", "obsolete": False } },
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
        #params = fake_api.call_args_list[0].kwargs["params"]
        #assert params["dataTypeID"] == '"data_1361"'

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

        #with pytest.raises(ValueError, match="one topic term"):
        #    query_by_edam_terms([TOPIC, "http://edamontology.org/topic_0160"])
