"""Tests for the keyword mapping functionality."""

from unittest.mock import Mock, patch

import pytest

from edam_mcp.models.requests import KeywordMappingRequest
from edam_mcp.models.responses import KeywordMatch, KeywordMappingResponse
from edam_mcp.tools.keywords_mapper import map_keywords_to_edam_concepts


class TestKeywordMapperTool:
    """Test cases for the keyword mapper tool."""

    @pytest.mark.asyncio
    async def test_keyword_mapping_request_validation(self):
        """Test that keyword mapping requests are properly validated."""
        # Valid request with substring mode
        request = KeywordMappingRequest(
            keywords=["alignment", "sequence", "protein"],
            match_mode="substring",
            threshold=0.3,
            max_results=10,
        )

        assert request.keywords == ["alignment", "sequence", "protein"]
        assert request.match_mode == "substring"
        assert request.threshold == 0.3
        assert request.max_results == 10

    @pytest.mark.asyncio
    async def test_keyword_mapping_request_list_embeddings_mode(self):
        """Test keyword mapping request with list_embeddings mode."""
        request = KeywordMappingRequest(
            keywords=["alignment", "protein"],
            match_mode="list_embeddings",
            threshold=0.5,
            max_results=5,
        )

        assert request.match_mode == "list_embeddings"
        assert request.threshold == 0.5

    @pytest.mark.asyncio
    async def test_keyword_mapping_request_joined_embedding_mode(self):
        """Test keyword mapping request with joined_embedding mode."""
        request = KeywordMappingRequest(
            keywords=["sequence", "analysis"],
            match_mode="joined_embedding",
            threshold=0.4,
        )

        assert request.match_mode == "joined_embedding"
        assert request.threshold == 0.4
        assert request.max_results is None

    @pytest.mark.asyncio
    async def test_keyword_mapping_response_structure(self):
        """Test that keyword mapping responses have the correct structure."""
        # Create a mock keyword match
        match = KeywordMatch(
            concept_uri="http://edamontology.org/operation_0496",
            concept_label="Global alignment",
            confidence=0.85,
            concept_type="Operation",
            definition="Alignment of two sequences",
            synonyms=["global alignment", "Needleman-Wunsch"],
            match_type="list_embeddings",
            matched_keywords=["alignment", "global"],
        )

        response = KeywordMappingResponse(
            matches=[match],
            total_matches=1,
            match_mode="list_embeddings",
            threshold=0.3,
        )

        assert len(response.matches) == 1
        assert response.total_matches == 1
        assert response.match_mode == "list_embeddings"
        assert response.threshold == 0.3
        assert response.matches[0].confidence == 0.85
        assert response.matches[0].match_type == "list_embeddings"
        assert "alignment" in response.matches[0].matched_keywords

    @pytest.mark.asyncio
    async def test_keyword_mapping_response_multiple_matches(self):
        """Test keyword mapping response with multiple matches."""
        matches = [
            KeywordMatch(
                concept_uri="http://edamontology.org/operation_0496",
                concept_label="Global alignment",
                confidence=0.85,
                concept_type="Operation",
                definition="Alignment of two sequences",
                synonyms=["global alignment"],
                match_type="substring",
                matched_keywords=["alignment"],
            ),
            KeywordMatch(
                concept_uri="http://edamontology.org/operation_0292",
                concept_label="Sequence alignment",
                confidence=0.75,
                concept_type="Operation",
                definition="Alignment of sequences",
                synonyms=["alignment"],
                match_type="substring",
                matched_keywords=["alignment", "sequence"],
            ),
        ]

        response = KeywordMappingResponse(
            matches=matches,
            total_matches=2,
            match_mode="substring",
            threshold=None,
        )

        assert len(response.matches) == 2
        assert response.total_matches == 2
        assert response.match_mode == "substring"
        assert response.threshold is None

    @pytest.mark.asyncio
    @patch("edam_mcp.tools.keywords_mapper.OntologyLoader")
    @patch("edam_mcp.tools.keywords_mapper.KeywordMatcher")
    async def test_keyword_mapper_tool_substring_mode(
        self, mock_matcher, mock_loader
    ):
        """Test the keyword mapper tool with substring mode."""
        # Mock context
        mock_context = Mock()
        mock_context.log = Mock()

        # Mock the ontology loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = True
        mock_loader.return_value = mock_loader_instance

        # Mock the keyword matcher
        mock_matcher_instance = Mock()
        mock_matcher_instance.match_by_substring.return_value = []
        mock_matcher.return_value = mock_matcher_instance

        # Create request
        request = KeywordMappingRequest(
            keywords=["alignment", "sequence"],
            match_mode="substring",
            threshold=0.3,
            max_results=5,
        )

        # Test the mapping function
        response = await map_keywords_to_edam_concepts(request, mock_context)

        assert isinstance(response, KeywordMappingResponse)
        assert response.total_matches == 0
        assert response.match_mode == "substring"
        assert response.threshold is None  # Threshold not used for substring mode
        mock_matcher_instance.match_by_substring.assert_called_once_with(
            ["alignment", "sequence"]
        )

    @pytest.mark.asyncio
    @patch("edam_mcp.tools.keywords_mapper.OntologyLoader")
    @patch("edam_mcp.tools.keywords_mapper.KeywordMatcher")
    async def test_keyword_mapper_tool_list_embeddings_mode(
        self, mock_matcher, mock_loader
    ):
        """Test the keyword mapper tool with list_embeddings mode."""
        # Mock context
        mock_context = Mock()
        mock_context.log = Mock()

        # Mock the ontology loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = True
        mock_loader.return_value = mock_loader_instance

        # Mock the keyword matcher with sample results (using actual KeywordMatch objects)
        mock_match = KeywordMatch(
            concept_uri="http://edamontology.org/operation_0496",
            concept_label="Global alignment",
            confidence=0.85,
            concept_type="Operation",
            definition="Alignment of two sequences",
            synonyms=["global alignment"],
            match_type="list_embeddings",
            matched_keywords=["alignment"],
        )

        mock_matcher_instance = Mock()
        mock_matcher_instance.match_by_list_embeddings.return_value = [mock_match]
        mock_matcher.return_value = mock_matcher_instance

        # Create request
        request = KeywordMappingRequest(
            keywords=["alignment", "protein"],
            match_mode="list_embeddings",
            threshold=0.4,
            max_results=10,
        )

        # Test the mapping function
        response = await map_keywords_to_edam_concepts(request, mock_context)

        assert isinstance(response, KeywordMappingResponse)
        assert response.total_matches == 1
        assert response.match_mode == "list_embeddings"
        assert response.threshold == 0.4
        mock_matcher_instance.match_by_list_embeddings.assert_called_once_with(
            ["alignment", "protein"], threshold=0.4
        )

    @pytest.mark.asyncio
    @patch("edam_mcp.tools.keywords_mapper.OntologyLoader")
    @patch("edam_mcp.tools.keywords_mapper.KeywordMatcher")
    async def test_keyword_mapper_tool_joined_embedding_mode(
        self, mock_matcher, mock_loader
    ):
        """Test the keyword mapper tool with joined_embedding mode."""
        # Mock context
        mock_context = Mock()
        mock_context.log = Mock()

        # Mock the ontology loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = True
        mock_loader.return_value = mock_loader_instance

        # Mock the keyword matcher
        mock_matcher_instance = Mock()
        mock_matcher_instance.match_by_joined_embedding.return_value = []
        mock_matcher.return_value = mock_matcher_instance

        # Create request
        request = KeywordMappingRequest(
            keywords=["sequence", "analysis", "tool"],
            match_mode="joined_embedding",
            threshold=0.5,
        )

        # Test the mapping function
        response = await map_keywords_to_edam_concepts(request, mock_context)

        assert isinstance(response, KeywordMappingResponse)
        assert response.match_mode == "joined_embedding"
        assert response.threshold == 0.5
        mock_matcher_instance.match_by_joined_embedding.assert_called_once_with(
            ["sequence", "analysis", "tool"], threshold=0.5
        )

    @pytest.mark.asyncio
    @patch("edam_mcp.tools.keywords_mapper.OntologyLoader")
    @patch("edam_mcp.tools.keywords_mapper.KeywordMatcher")
    async def test_keyword_mapper_max_results_limit(self, mock_matcher, mock_loader):
        """Test that max_results properly limits the number of returned matches."""
        # Mock context
        mock_context = Mock()
        mock_context.log = Mock()

        # Mock the ontology loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = True
        mock_loader.return_value = mock_loader_instance

        # Create multiple mock matches using actual KeywordMatch objects
        mock_matches = [
            KeywordMatch(
                concept_uri=f"http://edamontology.org/operation_{i:04d}",
                concept_label=f"Test operation {i}",
                confidence=1.0 - (i * 0.1),
                concept_type="Operation",
                definition=f"Test definition {i}",
                synonyms=[],
                match_type="substring",
                matched_keywords=["alignment"],
            )
            for i in range(10)
        ]

        mock_matcher_instance = Mock()
        mock_matcher_instance.match_by_substring.return_value = mock_matches
        mock_matcher.return_value = mock_matcher_instance

        # Create request with max_results limit
        request = KeywordMappingRequest(
            keywords=["alignment"],
            match_mode="substring",
            max_results=3,
        )

        # Test the mapping function
        response = await map_keywords_to_edam_concepts(request, mock_context)

        assert response.total_matches == 3
        assert len(response.matches) == 3

    @pytest.mark.asyncio
    @patch("edam_mcp.tools.keywords_mapper.OntologyLoader")
    async def test_keyword_mapper_ontology_load_failure(self, mock_loader):
        """Test error handling when ontology loading fails."""
        # Mock context
        mock_context = Mock()
        mock_context.log = Mock()

        # Mock the ontology loader to fail
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = False
        mock_loader.return_value = mock_loader_instance

        # Create request
        request = KeywordMappingRequest(
            keywords=["alignment"],
            match_mode="substring",
        )

        # Test that RuntimeError is raised
        with pytest.raises(RuntimeError, match="Failed to load EDAM ontology"):
            await map_keywords_to_edam_concepts(request, mock_context)

    @pytest.mark.asyncio
    @patch("edam_mcp.tools.keywords_mapper.OntologyLoader")
    @patch("edam_mcp.tools.keywords_mapper.KeywordMatcher")
    async def test_keyword_mapper_invalid_match_mode(self, mock_matcher, mock_loader):
        """Test error handling for invalid match mode."""
        # Mock context
        mock_context = Mock()
        mock_context.log = Mock()

        # Mock the ontology loader
        mock_loader_instance = Mock()
        mock_loader_instance.load_ontology.return_value = True
        mock_loader.return_value = mock_loader_instance

        mock_matcher_instance = Mock()
        mock_matcher.return_value = mock_matcher_instance

        # Create request with invalid match_mode (bypassing Pydantic validation for testing)
        request = KeywordMappingRequest(
            keywords=["alignment"],
            match_mode="substring",
        )
        # Manually override the match_mode to test error handling
        request.match_mode = "invalid_mode"

        # Test that ValueError is raised
        with pytest.raises(
            ValueError, match="Invalid match_mode: invalid_mode"
        ):
            await map_keywords_to_edam_concepts(request, mock_context)

    @pytest.mark.asyncio
    async def test_keyword_match_extends_concept_match(self):
        """Test that KeywordMatch properly extends ConceptMatch."""
        # Create a KeywordMatch
        match = KeywordMatch(
            concept_uri="http://edamontology.org/operation_0496",
            concept_label="Global alignment",
            confidence=0.85,
            concept_type="Operation",
            definition="Alignment of two sequences",
            synonyms=["global alignment"],
            match_type="list_embeddings",
            matched_keywords=["alignment", "global"],
        )

        # Verify base ConceptMatch fields
        assert match.concept_uri == "http://edamontology.org/operation_0496"
        assert match.concept_label == "Global alignment"
        assert match.confidence == 0.85
        assert match.concept_type == "Operation"
        assert match.definition == "Alignment of two sequences"
        assert match.synonyms == ["global alignment"]

        # Verify KeywordMatch specific fields
        assert match.match_type == "list_embeddings"
        assert match.matched_keywords == ["alignment", "global"]
