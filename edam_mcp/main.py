"""Main entry point for the EDAM MCP server."""

import logging
import sys

from fastmcp import FastMCP
from fastmcp.server import Context

from .config import settings
from .models.mapping import MappingRequest, MappingResponse
from .models.segmentation import SegmentationRequest, SegmentationResponse
from .models.suggestion import SuggestionRequest, SuggestionResponse
from .models.workflow import WorkflowSummaryRequest, WorkflowSummaryResponse
from .tools import map_to_edam_concept, map_to_edam_operation, suggest_new_concept
from .tools.segment_text import segment_text
from .tools.workflow import get_workflow_summary

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

    @mcp.tool
    async def get_workflow_summary_tool(request: WorkflowSummaryRequest, context: Context) -> WorkflowSummaryResponse:
        """Get comprehensive summary of the EDAM mapping workflow for copilot planning."""
        return await get_workflow_summary(request, context)

    @mcp.tool
    async def segment_text_tool(request: SegmentationRequest, context: Context) -> SegmentationResponse:
        """Segment text into topic and keywords using NLP (spaCy)."""
        return await segment_text(request, context)

    @mcp.tool
    async def map_to_edam_concept_tool(request: MappingRequest, context: Context) -> MappingResponse:
        return await map_to_edam_concept(request, context)

    @mcp.tool
    async def map_to_edam_operation_tool(request: MappingRequest, context: Context) -> MappingResponse:
        return await map_to_edam_operation(request, context)

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
