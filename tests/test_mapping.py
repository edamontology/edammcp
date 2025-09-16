"""Tests for the mapping functionality."""

import pytest
from unittest.mock import Mock, patch

from edam_mcp.models.requests import MappingRequest
from edam_mcp.models.responses import ConceptMatch, MappingResponse
from edam_mcp.tools.mapping import map_description_to_concepts


class TestMappingTool:
    """Test cases for the mapping tool."""
    
    @pytest.mark.asyncio
    async def test_mapping_request_validation(self):
        """Test that mapping requests are properly validated."""
        # Valid request
        request = MappingRequest(
            description="sequence alignment tool",
            context="bioinformatics",
            max_results=5,
            min_confidence=0.7
        )
        
        assert request.description == "sequence alignment tool"
        assert request.context == "bioinformatics"
        assert request.max_results == 5
        assert request.min_confidence == 0.7
    
    @pytest.mark.asyncio
    async def test_mapping_response_structure(self):
        """Test that mapping responses have the correct structure."""
        # Create a mock concept match
        match = ConceptMatch(
            concept_uri="http://edamontology.org/operation_0296",
            concept_label="Sequence alignment",
            confidence=0.85,
            concept_type="Operation",
            definition="Aligning biological sequences",
            synonyms=["alignment", "sequence alignment"]
        )
        
        response = MappingResponse(
            matches=[match],
            total_matches=1,
            has_exact_match=False,
            confidence_threshold=0.7
        )
        
        assert len(response.matches) == 1
        assert response.total_matches == 1
        assert response.has_exact_match is False
        assert response.confidence_threshold == 0.7
        assert response.matches[0].confidence == 0.85
    
    @pytest.mark.asyncio
    @patch('edam_mcp.tools.mapping.OntologyLoader')
    @patch('edam_mcp.tools.mapping.ConceptMatcher')
    async def test_mapping_tool_integration(self, mock_matcher, mock_loader):
        """Test the mapping tool integration."""
        # Mock the ontology loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = True
        mock_loader.return_value = mock_loader_instance
        
        # Mock the concept matcher
        mock_matcher_instance = Mock()
        mock_matcher_instance.find_exact_matches.return_value = []
        mock_matcher_instance.match_concepts.return_value = []
        mock_matcher.return_value = mock_matcher_instance
        
        # Test the mapping function
        response = await map_description_to_concepts(
            description="test description",
            context="test context",
            max_results=3,
            min_confidence=0.6
        )
        
        assert isinstance(response, MappingResponse)
        assert response.total_matches == 0
        assert response.has_exact_match is False 