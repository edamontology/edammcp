#!/usr/bin/env python3
"""Basic usage example for the EDAM MCP server."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from edam_mcp.models.mapping import MappingRequest
    from edam_mcp.tools.mapping import map_to_edam_operation
    from edam_mcp.utils.context import MockContext
except ImportError as e:
    print(f"Error importing edam_mcp: {e}")
    print("Make sure you have installed the package in development mode:")
    print("  uv sync --dev")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_operation_mapping():
    """Example of mapping descriptions to EDAM operations concepts."""
    print("=== EDAM Operation Mapping Example ===\n")

    # Example descriptions to map
    descriptions = [
        "The Spectra package defines an efficient infrastructure for storing "
        "and handling mass spectrometry spectra and functionality to subset, "
        "process, visualize and compare spectra data. It provides different implementations "
        "(backends) to store mass spectrometry data. These comprise backends tuned for fast "
        "data access and processing and backends for very large data sets ensuring a small memory footprint.",
    ]

    for description in descriptions:
        print(f"Mapping: {description}")

        try:
            request = MappingRequest(
                description=description,
                context="bioinformatics",
                max_results=5,
                min_confidence=0.5,
            )

            mock_context = MockContext()

            response = await map_to_edam_operation(request, context=mock_context)

            if response.matches:
                print(f"  Found {response.total_matches} matches:")
                for match in response.matches:
                    print(f"    - {match.concept_label} (confidence: {match.confidence:.2f})")
                    print(f"      URI: {match.concept_uri}")
                    print(f"      Type: {match.concept_type}")
                    assert match.concept_type == "Operation"
            else:
                print("  No matches found")

        except Exception as e:
            print(f"  Error: {e}")


async def main():
    """Run the examples."""
    print("EDAM MCP Server - Bioconductor Spectra - Mapping")
    print("=" * 50)

    # Run mapping example
    await example_operation_mapping()


if __name__ == "__main__":
    asyncio.run(main())
