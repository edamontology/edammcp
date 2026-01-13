import pytest

from edam_mcp.tools.mapping import map_description_to_concepts
from edam_mcp.models.requests import SuggestionRequest
from edam_mcp.tools.suggestion import suggest_concepts, validate_suggestion_request


class TestValidationTool:
    """Test cases for the validation tool."""

    @pytest.mark.asyncio
    async def test_validation_basic(self):
        """Test basic mapping functionality."""
        request = SuggestionRequest(
            description="A tool for MSA",
            concept_type="Operation",
            parent_concept="Alignment",
            rationale = None
        )

        assert request.description=="A tool for MSA"
        assert request.concept_type=="Operation"
        assert request.parent_concept=="Alignment"
        assert request.rationale is None
