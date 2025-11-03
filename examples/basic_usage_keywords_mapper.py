#!/usr/bin/env python3
"""Basic usage example for keyword mapping to EDAM concepts."""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from edam_mcp.tools.keywords_mapper import map_keywords_to_concepts
except ImportError as e:
    print(f"Error importing edam_mcp: {e}")
    print("Make sure you have installed the package in development mode:")
    print("  uv sync --dev")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def example_substring_matching():
    """Example of keyword matching using substring mode."""
    print("=== Substring Matching Example ===\n")

    keywords = ["sequence", "alignment", "protein"]
    print(f"Keywords: {keywords}")

    try:
        response = await map_keywords_to_concepts(
            keywords=keywords,
            match_mode="substring",
            max_results=5,
        )

        if response.matches:
            print(f"  Found {response.total_matches} matches:")
            for match in response.matches:
                print(f"    - {match.concept_label}")
                print(f"      URI: {match.concept_uri}")
                print(f"      Type: {match.concept_type}")
                print(f"      Matched keywords: {', '.join(match.matched_keywords)}")
                print()
        else:
            print("  No matches found")

    except Exception as e:
        print(f"  Error: {e}")

    print()


async def example_list_embeddings_matching():
    """Example of keyword matching using list embeddings mode."""
    print("=== List Embeddings Matching Example ===\n")

    # Example keyword sets
    keyword_sets = [
        ["genome", "assembly", "denovo"],
        ["protein", "structure", "prediction"],
        ["RNA", "sequencing", "expression"],
    ]

    for keywords in keyword_sets:
        print(f"Keywords: {keywords}")

        try:
            response = await map_keywords_to_concepts(
                keywords=keywords,
                match_mode="list_embeddings",
                threshold=0.3,
                max_results=3,
            )

            if response.matches:
                print(f"  Found {response.total_matches} matches (threshold: {response.threshold}):")
                for match in response.matches:
                    print(f"    - {match.concept_label}")
                    print(f"      URI: {match.concept_uri}")
                    print(f"      Type: {match.concept_type}")
                    if match.confidence:
                        print(f"      Confidence: {match.confidence:.3f}")
                    print(f"      Matched keywords: {', '.join(match.matched_keywords)}")
                    print()
            else:
                print("  No matches found")

        except Exception as e:
            print(f"  Error: {e}")

        print()


async def example_joined_embedding_matching():
    """Example of keyword matching using joined embedding mode."""
    print("=== Joined Embedding Matching Example ===\n")

    # Example keyword sets that work better when combined
    keyword_sets = [
        ["mass", "spectrometry", "proteomics"],
        ["gene", "ontology", "enrichment"],
        ["molecular", "docking", "simulation"],
    ]

    for keywords in keyword_sets:
        print(f"Keywords: {keywords}")

        try:
            response = await map_keywords_to_concepts(
                keywords=keywords,
                match_mode="joined_embedding",
                threshold=0.35,
                max_results=3,
            )

            if response.matches:
                print(f"  Found {response.total_matches} matches (threshold: {response.threshold}):")
                for match in response.matches:
                    print(f"    - {match.concept_label}")
                    print(f"      URI: {match.concept_uri}")
                    print(f"      Type: {match.concept_type}")
                    if match.confidence:
                        print(f"      Confidence: {match.confidence:.3f}")
                    if match.definition:
                        print(f"      Definition: {match.definition[:100]}...")
                    print()
            else:
                print("  No matches found")

        except Exception as e:
            print(f"  Error: {e}")

        print()


async def example_comparing_modes():
    """Example comparing different matching modes on the same keywords."""
    print("=== Comparing Matching Modes ===\n")

    keywords = ["phylogenetic", "tree", "visualization"]
    print(f"Keywords: {keywords}\n")

    modes = ["substring", "list_embeddings", "joined_embedding"]

    for mode in modes:
        print(f"Mode: {mode}")

        try:
            response = await map_keywords_to_concepts(
                keywords=keywords,
                match_mode=mode,
                threshold=0.3 if mode != "substring" else 0.0,
                max_results=3,
            )

            if response.matches:
                print(f"  Found {response.total_matches} matches:")
                for i, match in enumerate(response.matches[:3], 1):
                    conf_str = f" (confidence: {match.confidence:.3f})" if match.confidence else ""
                    print(f"    {i}. {match.concept_label}{conf_str}")
            else:
                print("  No matches found")

        except Exception as e:
            print(f"  Error: {e}")

        print()


async def main():
    """Run the examples."""
    print("EDAM MCP Server - Basic Usage Examples - Keywords Mapper")
    print("=" * 60)
    print()

    # Run all examples
    await example_substring_matching()
    await example_list_embeddings_matching()
    await example_joined_embedding_matching()
    await example_comparing_modes()

    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
