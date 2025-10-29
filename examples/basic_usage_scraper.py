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
    from edam_mcp.tools.mapping import map_description_to_concepts
except ImportError as e:
    print(f"Error importing edam_mcp: {e}")
    print("Make sure you have installed the package in development mode:")
    print("  uv sync --dev")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_scraping():
    """Example of scraping EDAM concepts from bio.tools."""
    print("=== EDAM Concept Scraping Example ===\n")

    biotools_tests = [
        {
            "description": "sequence alignment tool",
            "name": "clustalo",
            "biotools_curie": "biotools:clustalo",
            "ontology_type": "operation",
        },
        {
            "description": "Sequence alignment report in html format",
            "name": "multiqc",
            "biotools_curie": None,
            "ontology_type": "output",
        },
    ]

    for test in biotools_tests:
        print(f"Mapping with bio.tools: {test.get('name')}")

        try:
            response = await map_description_to_concepts(
                description=test.get("description"),
                name=test.get("name"),
                biotools_curie=test.get("biotools_curie"),
                ontology_type=test.get("ontology_type"),
                max_results=3,
                min_confidence=0.5,
            )

            if response.matches:
                print(f"  Found {response.total_matches} matches:")
                for match in response.matches:
                    print(f"    - {match.concept_label} (confidence: {match.confidence:.2f})")
                    print(f"      URI: {match.concept_uri}")
                    print(f"      Type: {match.concept_type}")
            else:
                print("  No matches found")

        except Exception as e:
            print(f"  Error: {e}")


async def main():
    """Run the examples."""
    print("EDAM MCP Server - Basic Usage Examples - Scraping")
    print("=" * 50)

    # Run mapping example
    await example_scraping()

    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
