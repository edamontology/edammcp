from fastmcp.server import Context

from ..config import settings
from ..models.query import BiotoolsQueryRequest, BiotoolsQueryResponse
from ..ontology import BiotoolsSearcher

async def query_biotools(
    request: BiotoolsQueryRequest, 
    context: Context,
) -> BiotoolsQueryResponse:
    """Search for tools in Biotools repository.

    This tool takes a seed and performs a query in the Biotools database. It returns matches
    with name, description, biotools database url, and list of concepts in the four EDAM ontology categories (topic, operation, format and data).

    Args:
        request: Biotools query request containing search seed and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Mapping response with matched concepts and confidence scores.
    """
    context.info(request)
    try:
        # Log the request
        context.info(f"Query keyword: {request.seed[:100]}...")

        searcher = BiotoolsSearcher()
        matches = searcher.query(
            seed=request.seed,
            max_results=request.max_results,
        )

        context.info(f"Found {len(matches)} tool matches")

        return BiotoolsQueryResponse(
            matches=matches,
            total_matches=len(matches),
        )

    except Exception as e:
        context.error(f"Error in biotools query: {e}")
        raise