# EDAM MCP Server

An MCP (Model Context Protocol) server for EDAM ontology mapping and concept suggestion. This server provides tools to:

1. **Map descriptions to EDAM concepts**: Given metadata or free text descriptions, find the most appropriate EDAM ontology concepts with confidence scores
2. **Suggest new concepts**: When no suitable concept exists, suggest new concepts that could be integrated into the EDAM ontology

Documentation [here](https://edamontology.github.io/edammcp/).

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
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
uv run python examples/basic_usage_mapper.py
uv run python examples/basic_usagesuggester.py

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

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd edammcp

# Install with uv (recommended)
uv sync

# Or install in development mode
uv sync --dev

# Or install with specific extras
uv sync --extra dev
```

## Usage

### Running the MCP Server

```bash
# Run the server directly
uv run python -m edam_mcp.main

# Or use the installed script
uv run edam-mcp
```

### Using with MCP Clients

The server exposes two main tools:

1. **`map_to_edam_concept`** - Maps descriptions to existing EDAM concepts
   - **Input**: Description text, context, confidence threshold
   - **Output**: List of matched concepts with confidence scores
   - **Example**: "sequence alignment tool" → "Sequence alignment" (confidence: 0.85)

2. **`suggest_new_concept`** - Suggests new concepts when no match is found
   - **Input**: Description text, concept type, parent concept
   - **Output**: List of suggested new concepts with hierarchical placement
   - **Example**: "quantum protein folding" → "Quantum Protein Folding" (suggested as child of "Sequence alignment")

### MCP Client Integration

```bash
# Install in Claude Desktop
# Add to your MCP configuration file:
{
  "mcpServers": {
    "edam-mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "edam_mcp.main"],
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
├── models/                # Pydantic models
│   ├── __init__.py
│   ├── requests.py        # Request models
│   └── responses.py       # Response models
├── ontology/              # EDAM ontology handling
│   ├── __init__.py
│   ├── loader.py          # Ontology loading and parsing
│   ├── matcher.py         # Concept matching logic
│   └── suggester.py       # New concept suggestion logic
├── tools/                 # MCP tools
│   ├── __init__.py
│   ├── mapping.py         # Mapping tool implementation
│   └── suggestion.py      # Suggestion tool implementation
└── utils/                 # Utility functions
    ├── __init__.py
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
2. Register the tool in `main.py`
3. Add corresponding request/response models in `models/`
4. Write tests for the new functionality

## Configuration

The server can be configured through environment variables:

- `EDAM_ONTOLOGY_URL`: URL to the EDAM ontology file (default: official EDAM OWL file)
- `SIMILARITY_THRESHOLD`: Minimum confidence threshold for mappings (default: 0.7)
- `MAX_SUGGESTIONS`: Maximum number of suggestions to return (default: 5)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
