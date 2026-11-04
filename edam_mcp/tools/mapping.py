"""MCP tool for mapping descriptions to EDAM concepts."""

import time

from fastmcp.server import Context

from ..config import settings
from ..models.mapping import MappingRequest, MappingResponse
from ..ontology import ConceptMatcher, OntologyLoader
from ..utils.context import MockContext

_matcher: ConceptMatcher | None = None
_matcher_built = 0.0


def get_matcher() -> ConceptMatcher:
    """Return the shared matcher, rebuilt after settings.cache_ttl so ontology updates are picked up."""
    # ponytail: module global, not thread-safe; fine for FastMCP's single event loop
    global _matcher, _matcher_built
    if _matcher is None or time.monotonic() - _matcher_built > settings.cache_ttl:
        ontology_loader = OntologyLoader()
        if not ontology_loader.load_ontology():
            raise RuntimeError("Failed to load EDAM ontology")
        _matcher, _matcher_built = ConceptMatcher(ontology_loader), time.monotonic()
    return _matcher


async def map_to_edam_concept(
    request: MappingRequest, context: Context, matcher: ConceptMatcher | None = None
) -> MappingResponse:
    """Map a description to existing EDAM concepts.

    This tool takes a description (metadata, free text) and finds the most
    appropriate mappings to concepts in the EDAM ontology. It returns matches
    with confidence scores, indicating how well each concept matches the description.

    Args:
        request: Mapping request containing description and parameters.
        context: MCP context for logging and progress reporting.
        matcher: Matcher to use; defaults to the shared one.

    Returns:
        Mapping response with matched concepts and confidence scores.
    """
    try:
        # Log the request
        context.info(f"Mapping description: {request.description[:100]}...")
        min_confidence = request.min_confidence if request.min_confidence is not None else settings.similarity_threshold

        concept_matcher = matcher or get_matcher()

        # First try exact matches
        exact_matches = concept_matcher.find_exact_matches(request.description)

        if exact_matches:
            context.info(f"Found {len(exact_matches)} exact matches")
            return MappingResponse(
                matches=exact_matches,
                total_matches=len(exact_matches),
                has_exact_match=True,
                confidence_threshold=min_confidence,
            )

        # Perform semantic matching
        context.info("Performing semantic matching...")
        matches = concept_matcher.match_concepts(
            description=request.description,
            context=request.context,
            max_results=request.max_results,
            min_confidence=min_confidence,
        )

        context.info(f"Found {len(matches)} semantic matches")

        return MappingResponse(
            matches=matches,
            total_matches=len(matches),
            has_exact_match=False,
            confidence_threshold=min_confidence,
        )

    except Exception as e:
        context.error(f"Error in concept mapping: {e}")
        raise


# Alternative function signature for direct use
async def map_description_to_concepts(
    description: str,
    context: str | None = None,
    max_results: int = 5,
    min_confidence: float | None = None,
) -> MappingResponse:
    """Alternative interface for mapping descriptions to concepts.

    Args:
        description: Text description to map.
        context: Additional context information.
        max_results: Maximum number of results to return.
        min_confidence: Minimum confidence threshold (defaults to settings.similarity_threshold).

    Returns:
        Mapping response with matched concepts.
    """
    request = MappingRequest(
        description=description,
        context=context,
        max_results=max_results,
        min_confidence=min_confidence,
    )

    mock_context = MockContext()

    return await map_to_edam_concept(request, mock_context)
