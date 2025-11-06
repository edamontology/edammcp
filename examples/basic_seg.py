#!/usr/bin/env python3
"""Basic usage example for the EDAM MCP server."""

import asyncio
import logging
import sys
from pathlib import Path
from edam_mcp.utils.context import MockContext

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from edam_mcp.tools.segment_text import segment_text
except ImportError as e:
    print(f"Error importing edam_mcp: {e}")
    print("Make sure you have installed the package in development mode:")
    print("  uv sync --dev")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_seg():
    """Example of segmentation of text in MCP"""
    print("=== EDAM Text Segmentation Example ===\n")

    try:
        mystr = "The document introduces the chromVAR R package, a computational tool designed to analyze sparse chromatin accessibility data, which can be derived from single cell or bulk ATAC-seq or DNase-seq experiments. The core functionality of chromVAR is to identify transcription factor motifs or other genomic features associated with cell-to-cell or sample-to-sample variability in chromatin accessibility. This enables the inference of transcription factor (TF) activity from the underlying chromatin landscape. The article provides links to the primary publication detailing the method, as well as critical third-party benchmarking, which highlights chromVAR’s strengths and limitations in the context of single-cell clustering and annotation, and compares it to other tools such as SnapATAC. Installation instructions are provided, noting dependencies, recommended companion packages (motifmatchr, JASPAR2016, chromVARmotifs), and common issues on different operating systems (Windows, Mac, Unix). Recommendations for parallelization using the BiocParallel package are described, including code snippets for setting up parallel processing with various backends. The document includes an illustrative quickstart guide comprising reading peak and fragment data, running quality control filters, motif matching (using JASPAR motifs), and calculating deviation scores, which represent per-sample motif accessibility. Users are directed to further documentation and community resources. The overall focus is on enabling end users in genomics to analyze regulatory heterogeneity and TF activity from high-dimensional sparse chromatin datasets, emphasizing robust installation, compatibility, and workflow integration."
        response = await segment_text(mystr, MockContext())
        return response 
    except Exception as e:
            print(f"  Error: {e}")


async def main():
    """Run the examples."""
    print("EDAM MCP Server - Basic Usage Examples - Segmentation")
    print("=" * 50)

    # Run mapping example
    await test_seg()

    print("Examples completed!")


if __name__ == "__main__":
    asyncio.run(main())
