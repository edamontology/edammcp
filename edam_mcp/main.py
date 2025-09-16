"""Main entry point for the EDAM MCP server."""

import asyncio
import logging
import sys
from typing import Optional

from fastmcp import FastMCP
from fastmcp.server import Context

from .config import settings
from .tools import map_to_edam_concept, suggest_new_concept
from .models.requests import MappingRequest, SuggestionRequest
from .models.responses import MappingResponse, SuggestionResponse

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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


async def main() -> None:
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


def run_server() -> None:
    """Run the server synchronously."""
    asyncio.run(main())


if __name__ == "__main__":
    run_server() 