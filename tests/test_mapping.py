"""Tests for the mapping functionality."""

import logging
from unittest import TestCase
from unittest.mock import patch

import pytest
import responses

from edam_mcp.models.requests import MappingRequest
from edam_mcp.models.responses import ConceptMatch, MappingResponse
from edam_mcp.ontology.loader import OntologyLoader
from edam_mcp.ontology.matcher import ConceptMatcher
from edam_mcp.tools.mapping import map_to_edam_concept


def mock_edam_request(rsps: responses.RequestsMock):
    """Mock the request to EDAM"""
    url = "https://raw.githubusercontent.com/edamontology/edamontology/master/EDAM_dev.owl"
    resp = """<?xml version="1.0"?>
    <rdf:RDF xmlns="http://edamontology.org/"
        xml:base="http://edamontology.org/"
        xmlns:dc="http://purl.org/dc/elements/1.1/"
        xmlns:dcterms="http://purl.org/dc/terms/"
        xmlns:owl="http://www.w3.org/2002/07/owl#"
        xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        xmlns:skos="http://www.w3.org/2004/02/skos/core#"
        xmlns:xml="http://www.w3.org/XML/1998/namespace"
        xmlns:xsd="http://www.w3.org/2001/XMLSchema#"
        xmlns:doap="http://usefulinc.com/ns/doap#"
        xmlns:foaf="http://xmlns.com/foaf/0.1/"
        xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
        xmlns:oboInOwl="http://www.geneontology.org/formats/oboInOwl#"
        xmlns:oboLegacy="http://purl.obolibrary.org/obo/">

    <!-- http://www.geneontology.org/formats/oboInOwl#hasDefinition -->
    <owl:AnnotationProperty rdf:about="http://www.geneontology.org/formats/oboInOwl#hasDefinition"/>
    
    <!-- http://www.geneontology.org/formats/oboInOwl#hasExactSynonym -->
    <owl:AnnotationProperty rdf:about="http://www.geneontology.org/formats/oboInOwl#hasExactSynonym"/>

    <!-- http://edamontology.org/format_2331 -->
    <owl:Class rdf:about="http://edamontology.org/format_2331">
        <oboInOwl:hasDefinition>HTML format.</oboInOwl:hasDefinition>
        <oboInOwl:hasExactSynonym>Hypertext Markup Language</oboInOwl:hasExactSynonym>
        <rdfs:label>HTML</rdfs:label>
    </owl:Class>

    <!-- http://edamontology.org/format_3475 -->
    <owl:Class rdf:about="http://edamontology.org/format_3475">
        <oboInOwl:hasDefinition>Tabular data represented as tab-separated values in a text file.</oboInOwl:hasDefinition>
        <oboInOwl:hasExactSynonym>Tab-delimited</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>Tab-separated values</oboInOwl:hasExactSynonym>
        <oboInOwl:hasExactSynonym>tab</oboInOwl:hasExactSynonym>
        <rdfs:label>TSV</rdfs:label>
    </owl:Class>

    <!-- http://edamontology.org/data_0867 -->
    <owl:Class rdf:about="http://edamontology.org/data_0867">
        <rdfs:label>Sequence alignment report</rdfs:label>
    </owl:Class>

    </rdf:RDF>
    """
    rsps.add(method="GET", url=url, body=resp.encode("utf-8"), status=200, content_type="application/rdf+xml")


class TestMappingTool(TestCase):
    """Test cases for the mapping tool."""

    def setUp(self):
        """Set up mock objects:
        - context
        - MappingRequest
        - ConceptMatch
        - ConceptMatcher
        """

        class MockContext:
            def __init__(self):
                self.log = logging.getLogger(__name__)

        self.mock_context = MockContext()

        self.test_request = MappingRequest(
            description="HTML file",
            context="my context",
            max_results=1,
            min_confidence=0.5,
        )

        self.mock_match_html = ConceptMatch(
            concept_uri="http://edamontology.org/format_2331",
            concept_label="HTML",
            confidence=1.0,
            concept_type="Format",
            definition="HTML format.",
            synonyms=["Hypertext Markup Language"],
        )
        self.mock_match_tsv = ConceptMatch(
            concept_uri="http://edamontology.org/format_3475",
            concept_label="TSV",
            confidence=0.9,
            concept_type="Format",
            definition="Tabular data represented as tab-separated values in a text file.",
            synonyms=["Tab-delimited", "Tab-separated values", "tab"],
        )

        with responses.RequestsMock() as rsps:
            mock_edam_request(rsps)
            self.loader = OntologyLoader()
            self.loader.load_ontology()
            self.matcher = ConceptMatcher(self.loader)

    def test_load_ontology(self):
        """Test the OntologyLoader class"""
        assert self.loader.concepts["http://edamontology.org/format_2331"]
        assert self.loader.concepts["http://edamontology.org/format_2331"].get("label") == "HTML"

    @patch("edam_mcp.ontology.loader.OntologyLoader.load_ontology")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.find_exact_matches")
    async def test_map_to_edam_concept_exact_match(self, mock_exact_matches, mock_loader):
        """Test the mapping tool integration."""
        # Mock the ontology loader
        mock_loader.return_value = True

        # Mock the concept matcher
        mock_exact_matches.return_value = [self.mock_match_html]

        # Test the mapping function
        response = await map_to_edam_concept(
            self.test_request,
            self.mock_context,
        )

        assert isinstance(response, MappingResponse)
        assert response.matches[0] == self.mock_match_html
        assert response.total_matches == 1
        assert response.has_exact_match is True

    @patch("edam_mcp.ontology.loader.OntologyLoader.load_ontology")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.match_concepts")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.get_concepts_from_biotools")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.find_exact_matches")
    async def test_map_to_edam_concept_no_exact_match(
        self, mock_exact_matches, mock_biotools, mock_matcher, mock_loader
    ):
        """Test the mapping tool integration."""
        # Mock the ontology loader
        mock_loader.return_value = True

        # Mock the concept matcher
        mock_exact_matches.return_value = []
        mock_biotools.return_value = [self.mock_match_html]
        mock_matcher.return_value = [self.mock_match_tsv]

        # Test the mapping function
        response = await map_to_edam_concept(
            self.test_request,
            self.mock_context,
        )

        assert isinstance(response, MappingResponse)
        assert response.matches[0] == self.mock_match_html
        assert response.matches[1] == self.mock_match_tsv
        assert response.total_matches == 2
        assert response.has_exact_match is False

    @patch("edam_mcp.ontology.loader.OntologyLoader.load_ontology")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.match_concepts")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.get_concepts_from_biotools")
    @patch("edam_mcp.ontology.matcher.ConceptMatcher.find_exact_matches")
    async def test_map_to_edam_concept_max_results(self, mock_exact_matches, mock_biotools, mock_matcher, mock_loader):
        """Test the mapping tool integration."""
        # Mock the ontology loader
        mock_loader.return_value = True

        # Mock the concept matcher
        mock_exact_matches.return_value = []
        mock_biotools.return_value = [self.mock_match_html]
        mock_matcher.return_value = [self.mock_match_tsv]

        # Test the mapping function
        response = await map_to_edam_concept(
            self.test_request,
            self.mock_context,
        )

        assert isinstance(response, MappingResponse)
        assert response.matches[0] == self.mock_match_html
        assert response.total_matches == 1
        assert response.has_exact_match is False

    def test_match_concepts(self):
        matches = self.matcher.match_concepts("HTML file.")
        assert len(matches) == 1
        assert matches[0].concept_uri == self.mock_match_html.concept_uri
        assert matches[0].concept_label == self.mock_match_html.concept_label
        assert matches[0].concept_type == self.mock_match_html.concept_type
        assert matches[0].definition == self.mock_match_html.definition
        assert matches[0].synonyms == self.mock_match_html.synonyms
        assert matches[0].confidence == pytest.approx(0.63, 0.01)

    def test_find_exact_matches(self):
        matches = self.matcher.find_exact_matches("HTML")
        assert len(matches) == 1
        assert matches[0] == self.mock_match_html
