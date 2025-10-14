"""Main entry point for the EDAM MCP server."""

import logging
import sys

from fastmcp import FastMCP
from fastmcp.server import Context

from .config import settings
from .models.requests import MappingRequest, SuggestionRequest
from .models.responses import MappingResponse, SuggestionResponse
from .tools import map_to_edam_concept, suggest_new_concept

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def create_server() -> FastMCP:
    """Create and configure the FastMCP server.

    Returns:
        Configured FastMCP server instance.
    """
    # Create server
    mcp = FastMCP("edam-mcp")

    # Register tools using decorators
    @mcp.tool
    async def map_to_edam_concept_tool(request: MappingRequest, context: Context) -> MappingResponse:
        return await map_to_edam_concept(request, context)

    @mcp.tool
    async def suggest_new_concept_tool(request: SuggestionRequest, context: Context) -> SuggestionResponse:
        return await suggest_new_concept(request, context)

    return mcp


def main() -> None:
    """Main entry point for running the server."""
    try:
        # Create server
        mcp = create_server()

        # Run server
        mcp.run()

    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
