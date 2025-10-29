"""MCP tool for mapping descriptions to EDAM concepts."""

import logging

from fastmcp.server import Context

from ..models.requests import BiotoolsRequest
from ..models.responses import BiotoolsResponse
from ..ontology.biotools_scrapper import BiotoolsScrapper

logger = logging.getLogger(__name__)


async def extract_edam_concepts_from_biotools(request: BiotoolsRequest, context: Context) -> BiotoolsResponse:
    """Extract EDAM concepts from bio.tools.

    This tool takes the name of a tool and optionally the bio.tools CURIE
    and finds the available EDAM ontology concepts. It returns found concepts.

    Args:
        request: Biotools request containing tool name and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Biotools response with found concepts.
    """
    try:
        # Log the request
        context.log.info(f"Extracting bio.tools ontologies for tools: {request.name}")

        biotools_scrapper = BiotoolsScrapper()

        # Extract matches from bio.tools
        context.log.info("Looking for ontology terms on bio.tools...")
        matches = biotools_scrapper.get_concepts_from_biotools(
            tool_name=request.name,
            tool_curie=request.biotools_curie,
            ontology_type=request.ontology_type,
        )
        context.log.info(f"Found {len(matches)} terms on bio.tools.")

        return BiotoolsResponse(
            matches=matches[: request.max_results],
            total_matches=len(matches),
        )

    except Exception as e:
        context.log.error(f"Error in concept scrapping: {e}")
        raise


# Alternative function signature for direct use
async def exctract_concepts_from_biotools(
    name: str | None = None,
    biotools_curie: str | None = None,
    ontology_type: str | None = None,
    max_results: int = 5,
) -> BiotoolsResponse:
    """Alternative interface for exctracting concepts from bio.tools.

    Args:
        name: Name of the tool.
        biotools_curie: Biotools CURIE of the tool.
        ontology_type: Type of ontology to extract.
        max_results: Maximum number of results to return.

    Returns:
        Biotools response with found concepts.
    """
    request = BiotoolsRequest(
        name=name,
        biotools_curie=biotools_curie,
        ontology_type=ontology_type,
        max_results=max_results,
    )

    # Create a mock context for standalone use
    class MockContext:
        def __init__(self):
            self.log = logging.getLogger(__name__)

    mock_context = MockContext()

    return await extract_edam_concepts_from_biotools(request, mock_context)
