# EDAM MCP Server

An MCP (Model Context Protocol) server for EDAM ontology mapping, concept suggestion, and text segmentation. This server provides tools to:

1. **Map descriptions to EDAM concepts**: Find the most appropriate EDAM ontology concepts for metadata or free text descriptions with confidence scores
2. **Suggest new concepts**: Generate suggestions for new EDAM concepts when no suitable match exists
3. **Segment text**: Extract topics and keywords from text for improved mapping accuracy

Documentation [here](https://edamontology.github.io/edammcp/).

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone git@github.com:edamontology/edammcp.git
cd edammcp

# Install with uv (recommended)
uv sync --dev

# Or install manually
uv sync
uv pip install -e .
```

### Basic Usage

```bash
# Test the basic structure (fast)
uv run python examples/simple_test.py

# Run the full example (downloads ML models on first run)

# Run the mapper
uv run python examples/basic_usage_mapper.py

# Run the suggester
uv run python examples/basic_usage_suggester.py

# Run the segmentation example
uv run python examples/basic_seg.py

# Start the MCP server
uv run edam-mcp
```

### Example Output

For examples on how to run the functions, please check [basic-usage.md](/docs/examples/basic-usage.md).

## Features

- **Ontology Mapping**: Semantic search and matching of descriptions to existing EDAM concepts
- **Confidence Scoring**: Provide confidence levels for mapping results
- **Concept Suggestion**: Generate suggestions for new EDAM concepts when no match is found
- **Hierarchical Placement**: Suggest appropriate placement within the EDAM ontology hierarchy
- **Text Segmentation**: Segment and analyze text to extract topics and keywords for mapping

## Usage

### Running the MCP Server

```bash
# Run the server directly
uv run python -m edam_mcp.main

# Or use the installed script
uv run edam-mcp
```

### Using with MCP Clients

The server exposes three main tools:

1. **`map_to_edam_concept`** - Maps descriptions to existing EDAM concepts
   - **Input**: Description text, context, confidence threshold
   - **Output**: List of matched concepts with confidence scores
   - **Example**: "sequence alignment tool" → "Sequence alignment" (confidence: 0.85)

2. **`suggest_new_concept`** - Suggests new concepts when no match is found
   - **Input**: Description text, concept type, parent concept
   - **Output**: List of suggested new concepts with hierarchical placement
   - **Example**: "quantum protein folding" → "Quantum Protein Folding" (suggested as child of "Sequence alignment")

3. **`get_workflow_summary`** - Get comprehensive summary of the EDAM mapping workflow for copilot planning
   - **Input**: (No parameters required)
   - **Output**: Workflow summary with available functions and their descriptions

### MCP Client Integration

```bash
# Install in Claude Desktop or other compatible MCP clients
# Add to your MCP configuration file:
{
  "mcpServers": {
    "edam-mcp": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/edamontology/edammcp", "edam-mcp"],
      "env": {
        "EDAM_SIMILARITY_THRESHOLD": "0.7"
      }
    }
  }
}
```

Alternately, if you plan to develop the code and test it from your MCP client at the same time:

```bash
# Install in Claude Desktop or other compatible MCP clients
# Add to your MCP configuration file:

{
  "mcpServers": {
    "edam-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/repo/edammcp",
        "run",
        "edam-mcp"
      ],
      "env": {
        "EDAM_SIMILARITY_THRESHOLD": "0.7"
      }
    }
  }
}
```

## Project Structure

```
edam_mcp/
├── __init__.py
├── main.py                 # Main server entry point
├── config.py              # Configuration management
├── models/                # Pydantic models (organized by functionality)
│   ├── __init__.py
│   ├── mapping.py         # Mapping request/response models
│   ├── suggestion.py      # Suggestion request/response models
│   ├── segmentation.py   # Text segmentation models
│   └── workflow.py        # Workflow summary models
├── ontology/              # EDAM ontology handling
│   ├── __init__.py
│   ├── loader.py          # Ontology loading and parsing
│   ├── matcher.py         # Concept matching logic
│   └── suggester.py       # New concept suggestion logic
├── tools/                 # MCP tools
│   ├── __init__.py
│   ├── mapping.py         # Mapping tool implementation
│   ├── suggestion.py      # Suggestion tool implementation
│   ├── segment_text.py    # Text segmentation tool
│   └── workflow.py        # Workflow summary tool
└── utils/                 # Utility functions
    ├── __init__.py
    ├── context.py         # Context utilities
    ├── text_processing.py # Text preprocessing
    └── similarity.py      # Similarity calculation
```

## Development

### Setting up the development environment

```bash
# Install development dependencies
uv sync --dev

# Run tests
uv run pytest

# Lint & Format code
uv run pre-commit install # Run only first time
uv run pre-commit run --all-files
```

### Adding new tools

1. Create a new tool function in the appropriate module under `tools/`
2. Create corresponding request/response models in `models/` (organized by functionality)
3. Register the tool in `main.py` using the `@mcp.tool` decorator
4. Export models in `models/__init__.py`
5. Write tests for the new functionality

## Configuration

The server can be configured through environment variables:

- `EDAM_ONTOLOGY_URL`: URL to the EDAM ontology file (default: official EDAM OWL file)
- `EDAM_SIMILARITY_THRESHOLD`: Minimum confidence threshold for mappings (default: 0.7)
- `EDAM_MAX_SUGGESTIONS`: Maximum number of suggestions to return (default: 5)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
