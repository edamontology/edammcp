"""MCP tool for mapping descriptions to EDAM concepts."""

import logging

from fastmcp.server import Context

from ..models.requests import MappingRequest
from ..models.responses import MappingResponse
from ..ontology import ConceptMatcher, OntologyLoader

logger = logging.getLogger(__name__)


async def map_to_edam_concept(request: MappingRequest, context: Context) -> MappingResponse:
    """Map a description to existing EDAM concepts.

    This tool takes a description (metadata, free text) and finds the most
    appropriate mappings to concepts in the EDAM ontology. It returns matches
    with confidence scores, indicating how well each concept matches the description.

    Args:
        request: Mapping request containing description and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Mapping response with matched concepts and confidence scores.
    """
    try:
        # Log the request
        context.log.info(f"Mapping description: {request.description[:100]}...")

        # Initialize ontology components
        ontology_loader = OntologyLoader()
        if not ontology_loader.load_ontology():
            raise RuntimeError("Failed to load EDAM ontology")

        concept_matcher = ConceptMatcher(ontology_loader)

        # First try exact matches
        exact_matches = concept_matcher.find_exact_matches(request.description)

        if exact_matches:
            context.log.info(f"Found {len(exact_matches)} exact matches")
            return MappingResponse(
                matches=exact_matches,
                total_matches=len(exact_matches),
                has_exact_match=True,
                confidence_threshold=request.min_confidence,
            )

        # Extract matches from bio.tools
        context.log.info("Looking for ontology terms on bio.tools...")
        matches = concept_matcher.get_concepts_from_biotools(
            tool_name=request.name,
            tool_curie=request.biotools_curie,
            max_results=request.max_results,
            ontology_type=request.ontology_type,
        )
        context.log.info(f"Found {len(matches)} terms on bio.tools.")

        # Perform semantic matching
        context.log.info("Performing semantic matching...")
        matches += concept_matcher.match_concepts(
            description=request.description,
            context=request.context,
            max_results=request.max_results,
            min_confidence=request.min_confidence,
        )

        context.log.info(f"Found {len(matches)} semantic matches")

        return MappingResponse(
            matches=matches,
            total_matches=len(matches),
            has_exact_match=False,
            confidence_threshold=request.min_confidence,
        )

    except Exception as e:
        context.log.error(f"Error in concept mapping: {e}")
        raise


# Alternative function signature for direct use
async def map_description_to_concepts(
    description: str,
    name: str | None = None,
    biotools_curie: str | None = None,
    context: str | None = None,
    ontology_type: str | None = None,
    max_results: int = 5,
    min_confidence: float = 0.5,
) -> MappingResponse:
    """Alternative interface for mapping descriptions to concepts.

    Args:
        description: Text description to map.
        context: Additional context information.
        max_results: Maximum number of results to return.
        min_confidence: Minimum confidence threshold.

    Returns:
        Mapping response with matched concepts.
    """
    request = MappingRequest(
        description=description,
        name=name,
        biotools_curie=biotools_curie,
        context=context,
        ontology_type=ontology_type,
        max_results=max_results,
        min_confidence=min_confidence,
    )

    # Create a mock context for standalone use
    class MockContext:
        def __init__(self):
            self.log = logging.getLogger(__name__)

    mock_context = MockContext()

    return await map_to_edam_concept(request, mock_context)
