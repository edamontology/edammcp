"""Tests for the mapping functionality."""

import logging
from unittest import TestCase

import responses
from pydantic_core._pydantic_core import ValidationError

from edam_mcp.models.requests import BiotoolsRequest
from edam_mcp.models.responses import ConceptMatch
from edam_mcp.ontology.biotools_scraper import BiotoolsScraper
from edam_mcp.tools.biotools_scraping import extract_edam_concepts_from_biotools


def mock_biotools_request(rsps: responses.RequestsMock):
    """Mock the request to EDAM"""
    url = "https://bio.tools/api/t/?q=multiqc&format=json"
    rsps_json = {
        "list": [
            {
                "name": "MultiQC",
                "description": "MultiQC aggregates results from multiple bioinformatics analyses across many samples into a single report. It searches a given directory for analysis logs and compiles a HTML report. It's a general use tool, perfect for summarising the output from numerous bioinformatics tools.",
                "biotoolsCURIE": "biotools:multiqc",
                "function": [
                    {
                        "output": [
                            {
                                "data": {
                                    "uri": "http://edamontology.org/data_0867",
                                    "term": "Sequence alignment report",
                                },
                                "format": [{"uri": "http://edamontology.org/format_2331", "term": "HTML"}],
                            }
                        ],
                    }
                ],
            }
        ]
    }
    rsps.add(method="GET", url=url, json=rsps_json, status=200)


class TestScrapingTool(TestCase):
    """Test cases for the scraping tool."""

    def setUp(self):
        """Set up mock objects:
        - context
        - BiotoolsRequest
        - ConceptMatch
        - BiotoolsScrapper
        """

        self.test_request = BiotoolsRequest(
            name="multiqc",
            biotools_curie="biotools:multiqc",
            ontology_type="output",
            max_results=1,
        )
        self.test_request_no_curie = BiotoolsRequest(
            name="multiqc",
            biotools_curie=None,
            ontology_type="output",
            max_results=1,
        )

        self.mock_match_html = ConceptMatch(
            concept_uri="http://edamontology.org/format_2331",
            concept_label="HTML",
            confidence=1.0,
            concept_type="Format",
            definition="HTML format.",
            synonyms=["Hypertext Markup Language"],
        )

        self.scrapper = BiotoolsScrapper()

        class MockContext:
            def __init__(self):
                self.log = logging.getLogger(__name__)

        self.mock_context = MockContext()

    async def test_get_concepts_from_biotools_with_curie(self):
        with responses.RequestsMock() as rsps:
            mock_biotools_request(rsps)
            matches = extract_edam_concepts_from_biotools(self.test_request, self.mock_context)
            print(matches)
            # The first match is the data URI, check the HTML
            assert matches[1].concept_uri == self.mock_match_html.concept_uri
            assert matches[1].concept_label == self.mock_match_html.concept_label
            assert matches[1].concept_type == self.mock_match_html.concept_type
            assert matches[1].definition == self.mock_match_html.definition
            assert matches[1].synonyms == self.mock_match_html.synonyms

    async def test_get_concepts_from_biotools_no_curie(self):
        with responses.RequestsMock() as rsps:
            mock_biotools_request(rsps)
            matches = extract_edam_concepts_from_biotools(self.test_request_no_curie, self.mock_context)
            # The first match is the data URI, check the HTML
            assert matches[1].concept_uri == self.mock_match_html.concept_uri
            assert matches[1].concept_label == self.mock_match_html.concept_label
            assert matches[1].concept_type == self.mock_match_html.concept_type
            assert matches[1].definition == self.mock_match_html.definition
            assert matches[1].synonyms == self.mock_match_html.synonyms

    def test_get_concepts_from_biotools_no_name(self):
        with self.assertRaises(ValidationError):
            BiotoolsRequest(
                name=None,
                biotools_curie="biotools:multiqc",
                ontology_type="output",
                max_results=1,
            )
