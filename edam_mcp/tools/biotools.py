import asyncio

from fastmcp.server import Context

from ..models.biotools import BiotoolsQueryResponse, BiotoolsQueryTermsRequest, BiotoolsQueryTextRequest, BiotoolsQueryExactRequest
from ..ontology import biotools_searcher


async def query_biotools_by_text(
    request: BiotoolsQueryTextRequest,
    context: Context,
) -> BiotoolsQueryResponse:
    """Search for tools in Biotools repository by keywords or tool name.

    This tool takes keywords or a tool name as seed and performs a query in the Biotools database, ranked by relevance. It returns matches
    with name, description, biotools database url, and EDAM terms (URI to label) for topics, operations, and input and output data and formats.

    Args:
        request: Biotools query request containing search seed and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Biotools query response with matching tools and their associated EDAM terms.
    """
    try:
        context.info(f"Query keyword: {request.seed[:100]}...")
        matches, total = await asyncio.to_thread(biotools_searcher.query_by_text, request.seed, request.max_results)
        context.info(f"Found {total} tools, returning {len(matches)}")
        return BiotoolsQueryResponse(matches=matches, total_matches=total)

    except Exception as e:
        context.error(f"Error in biotools query: {e}")
        raise


async def query_biotools_by_text_exact_match(
    request: BiotoolsQueryExactRequest,
    context: Context,
) -> BiotoolsQueryResponse:
    """Search for tools in Biotools repository by seraching exactly the tool name or the biotools tool identifier.

    This tool takes a tool name and/or the biotools id of the tool and retrieves the corresponding tool information. It returns the match
    with name, description, biotools database url, and EDAM terms (URI to label) for topics, operations, and input and output data and formats.

    Args:
        request: Biotools query request containing tool name and/or biotools id and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Biotools query response with matching tools and their associated EDAM terms.
    """
    try:
        context.info(f"Query information - Tool name: {request.name} | Tool identifier: {request.biotools_id}")

        matches, total = await asyncio.to_thread(biotools_searcher.query_by_text_exact_match, request.name, request.biotools_id)
        context.info(f"Found {total} tools, returning {len(matches)}")
        return BiotoolsQueryResponse(matches=matches, total_matches=total)

    except Exception as e:
        context.error(f"Error in biotools query: {e}")
        raise


async def query_biotools_by_edam_terms(
    request: BiotoolsQueryTermsRequest,
    context: Context,
) -> BiotoolsQueryResponse:
    """Search for tools in Biotools repository by EDAM terms.

    This tool takes EDAM concept URIs (at most one per category) and searches the Biotools database for tools annotated with all of them. It returns matches
    with name, description, biotools database url, and EDAM terms (URI to label) for topics, operations, and input and output data and formats.

    Args:
        request: Biotools query request containing EDAM term URIs and parameters.
        context: MCP context for logging and progress reporting.

    Returns:
        Biotools query response with matching tools and their associated EDAM terms.
    """
    try:
        context.info(f"Query EDAM terms: {request.edam_terms[:5]}...")
        matches, total = await asyncio.to_thread(
            biotools_searcher.query_by_edam_terms, request.edam_terms, request.max_results
        )
        context.info(f"Found {total} tools, returning {len(matches)}")
        return BiotoolsQueryResponse(matches=matches, total_matches=total)

    except Exception as e:
        context.error(f"Error in biotools query: {e}")
        raise
