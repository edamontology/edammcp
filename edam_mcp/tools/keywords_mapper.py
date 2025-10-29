"""MCP tool for mapping keywords to EDAM concepts."""

import logging

from fastmcp.server import Context

from ..models.requests import KeywordMappingRequest
from ..models.responses import KeywordMappingResponse
from ..ontology.keywords_matcher import KeywordMatcher
from ..ontology.loader import OntologyLoader

logger = logging.getLogger(__name__)


async def map_keywords_to_edam_concepts(request: KeywordMappingRequest, context: Context) -> KeywordMappingResponse:
    """Map keywords to EDAM concepts using various matching strategies.

    This tool takes a list of keywords and finds matching concepts in the EDAM
    ontology using one of three matching modes:
    - 'substring': Simple substring search in term descriptions
    - 'list_embeddings': Semantic matching with separate embeddings for each keyword
    - 'joined_embedding': Semantic matching with a single embedding from joined keywords

    Args:
        request: Keyword mapping request containing keywords and matching parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Keyword mapping response with matched concepts and metadata.

    Raises:
        RuntimeError: If ontology loading fails.
        ValueError: If invalid match_mode is provided.
    """
    try:
        # Log the request
        context.log.info(f"Mapping {len(request.keywords)} keywords using {request.match_mode} mode")
        context.log.debug(f"Keywords: {request.keywords}")

        # Initialize ontology components
        ontology_loader = OntologyLoader()
        if not ontology_loader.load_ontology():
            raise RuntimeError("Failed to load EDAM ontology")

        keyword_matcher = KeywordMatcher(ontology_loader)

        # Perform matching based on the selected mode
        matches = []

        if request.match_mode == "substring":
            context.log.info("Performing substring matching...")
            matches = keyword_matcher.match_by_substring(request.keywords)

        elif request.match_mode == "list_embeddings":
            context.log.info(f"Performing list embeddings matching (threshold: {request.threshold})...")
            matches = keyword_matcher.match_by_list_embeddings(request.keywords, threshold=request.threshold)

        elif request.match_mode == "joined_embedding":
            context.log.info(f"Performing joined embedding matching (threshold: {request.threshold})...")
            matches = keyword_matcher.match_by_joined_embedding(request.keywords, threshold=request.threshold)

        else:
            raise ValueError(
                f"Invalid match_mode: {request.match_mode}. "
                "Must be 'substring', 'list_embeddings', or 'joined_embedding'"
            )

        # Apply max_results limit if specified
        if request.max_results is not None and request.max_results > 0:
            matches = matches[: request.max_results]
            context.log.info(f"Limited results to top {request.max_results} matches")

        context.log.info(f"Found {len(matches)} total matches")

        # Determine if threshold applies (only for embedding modes)
        threshold_used = request.threshold if request.match_mode in ["list_embeddings", "joined_embedding"] else None

        return KeywordMappingResponse(
            matches=matches,
            total_matches=len(matches),
            match_mode=request.match_mode,
            threshold=threshold_used,
        )

    except Exception as e:
        context.log.error(f"Error in keyword mapping: {e}")
        raise


# Alternative function signature for direct use
async def map_keywords_to_concepts(
    keywords: list[str],
    match_mode: str = "substring",
    threshold: float = 0.3,
    max_results: int | None = None,
) -> KeywordMappingResponse:
    """Alternative interface for mapping keywords to concepts.

    Args:
        keywords: List of keywords to match.
        match_mode: Matching algorithm ('substring', 'list_embeddings', 'joined_embedding').
        threshold: Similarity threshold for embedding modes (default: 0.3).
        max_results: Maximum number of results to return (optional).

    Returns:
        Keyword mapping response with matched concepts.

    Raises:
        ValueError: If invalid match_mode is provided.
    """
    request = KeywordMappingRequest(
        keywords=keywords,
        match_mode=match_mode,
        threshold=threshold,
        max_results=max_results,
    )

    # Create a mock context for standalone use
    class MockContext:
        def __init__(self):
            self.log = logging.getLogger(__name__)

    mock_context = MockContext()

    return await map_keywords_to_edam_concepts(request, mock_context)
