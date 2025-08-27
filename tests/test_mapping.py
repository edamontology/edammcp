"""Tests for the mapping functionality."""

from unittest.mock import Mock, patch

from edam_mcp.models.responses import MappingResponse
from edam_mcp.tools.mapping import map_description_to_concepts


class TestMappingTool:
    """Test cases for the mapping tool."""

    @patch("edam_mcp.tools.mapping.OntologyLoader")
    @patch("edam_mcp.tools.mapping.ConceptMatcher")
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
            min_confidence=0.6,
        )

        assert isinstance(response, MappingResponse)
        assert response.total_matches == 0
        assert response.has_exact_match is False
