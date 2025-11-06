"""MCP tool for suggesting new EDAM concepts."""

from fastmcp.server import Context

# Import needed for the mapping attempt
from ..config import settings
from ..models.mapping import MappingRequest
from ..models.suggestion import SuggestionRequest, SuggestionResponse
from ..ontology import ConceptMatcher, ConceptSuggester, OntologyLoader
from ..utils.context import MockContext
from .mapping import map_to_edam_concept


async def suggest_new_concept(request: SuggestionRequest, context: Context) -> SuggestionResponse:
    """Suggest new EDAM concepts when no suitable existing concept is found.

    This tool generates suggestions for new concepts that could be integrated
    into the EDAM ontology. It first attempts to map the description to existing
    concepts, and if no suitable match is found, it generates suggestions for
    new concepts with appropriate placement in the ontology hierarchy.

    Args:
        request: Suggestion request containing description and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Suggestion response with proposed new concepts.
    """
    try:
        # Log the request
        context.info(f"Suggesting concepts for: {request.description[:100]}...")

        # Initialize ontology components
        ontology_loader = OntologyLoader()
        if not ontology_loader.load_ontology():
            raise RuntimeError("Failed to load EDAM ontology")

        concept_matcher = ConceptMatcher(ontology_loader)
        concept_suggester = ConceptSuggester(ontology_loader, concept_matcher)

        # First attempt to map to existing concepts
        context.info("Attempting to map to existing concepts...")
        mapping_response = await map_to_edam_concept(
            MappingRequest(
                description=request.description,
                context=request.rationale,
                max_results=5,
                min_confidence=0.7,
            ),
            context,
        )

        # Check if we found good matches
        if mapping_response.matches and mapping_response.matches[0].confidence >= 0.8:
            context.info("Found high-confidence existing concept matches")
            return SuggestionResponse(
                suggestions=[],
                total_suggestions=0,
                mapping_attempted=True,
                mapping_failed_reason="High-confidence existing matches found",
            )

        # Generate suggestions for new concepts
        context.info("Generating suggestions for new concepts...")
        suggestions = concept_suggester.suggest_concepts(
            description=request.description,
            concept_type=request.concept_type,
            parent_concept=request.parent_concept,
            rationale=request.rationale,
            max_suggestions=settings.max_suggestions,
        )

        context.info(f"Generated {len(suggestions)} concept suggestions")

        return SuggestionResponse(
            suggestions=suggestions,
            total_suggestions=len(suggestions),
            mapping_attempted=True,
            mapping_failed_reason=None,
        )

    except Exception as e:
        context.error(f"Error in concept suggestion: {e}")
        raise


# Alternative function signature for direct use
async def suggest_concepts_for_description(
    description: str,
    concept_type: str | None = None,
    parent_concept: str | None = None,
    rationale: str | None = None,
    max_suggestions: int = 5,
) -> SuggestionResponse:
    """Alternative interface for suggesting new concepts.

    Args:
        description: Description of the concept to suggest.
        concept_type: Type of concept (Operation, Data, Format, Topic, Identifier).
        parent_concept: Suggested parent concept.
        rationale: Rationale for the suggestion.
        max_suggestions: Maximum number of suggestions to generate.

    Returns:
        Suggestion response with proposed new concepts.
    """
    request = SuggestionRequest(
        description=description,
        concept_type=concept_type,
        parent_concept=parent_concept,
        rationale=rationale,
    )

    mock_context = MockContext()

    return await suggest_new_concept(request, mock_context)
