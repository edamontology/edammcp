# Development Setup

This guide will help you set up a local development environment for the EDAM MCP Server.

## 🛠️ Prerequisites

### Required Software

- **Python 3.12+**: The project requires Python 3.12 or higher
- **uv**: Fast Python package manager (recommended)
- **Git**: Version control system
- **Make**: Build automation (optional, for convenience)

### System Requirements

- **Memory**: At least 2GB RAM (4GB+ recommended)
- **Storage**: At least 1GB free space for dependencies and models
- **Network**: Internet connection for downloading dependencies and ML models

## 🚀 Quick Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/edammcp.git
cd edammcp
```

### 2. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install Dependencies

```bash
# Install all dependencies including development tools
uv sync --dev

# Install the package in development mode
uv pip install -e .
```

### 4. Verify Installation

```bash
# Test basic functionality
uv run python examples/simple_test.py

# Run tests
uv run pytest

# Check code formatting
uv run black --check edam_mcp/
```

## 🔧 Development Environment

### IDE Setup

#### VS Code (Recommended)

1. **Install Extensions**:
   - Python
   - Pylance
   - Python Test Explorer
   - Python Docstring Generator

2. **Workspace Settings** (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreter": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": false,
    "python.linting.ruffEnabled": true,
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

#### PyCharm

1. **Configure Interpreter**:
   - Go to Settings → Project → Python Interpreter
   - Add new interpreter from `.venv/bin/python`

2. **Configure Testing**:
   - Go to Settings → Tools → Python Integrated Tools
   - Set default test runner to pytest

### Environment Variables

Create a `.env` file in the project root:

```bash
# Development settings
EDAM_LOG_LEVEL=DEBUG
EDAM_SIMILARITY_THRESHOLD=0.6
EDAM_MAX_SUGGESTIONS=10

# Optional: Use local ontology file for faster development
# EDAM_ONTOLOGY_URL=file:///path/to/local/edam.owl
```

## 🧪 Testing Setup

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_mapping.py

# Run with coverage
uv run pytest --cov=edam_mcp --cov-report=html

# Run tests in parallel
uv run pytest -n auto
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── test_mapping.py          # Mapping tool tests
├── test_suggestion.py       # Suggestion tool tests
├── test_ontology.py         # Ontology tests
├── test_models.py           # Model validation tests
└── test_utils.py            # Utility function tests
```

### Writing Tests

```python
import pytest
from edam_mcp.tools.mapping import map_description_to_concepts

@pytest.mark.asyncio
async def test_mapping_basic():
    """Test basic mapping functionality."""
    response = await map_description_to_concepts(
        description="sequence alignment",
        max_results=1
    )
    
    assert response.total_matches > 0
    assert response.matches[0].confidence > 0.5
```

## 🔍 Code Quality

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
# Install pre-commit
uv run pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

### Code Formatting

```bash
# Format code with black
uv run black edam_mcp/ tests/ examples/

# Sort imports with isort
uv run isort edam_mcp/ tests/ examples/

# Check formatting
uv run black --check edam_mcp/
uv run isort --check-only edam_mcp/
```

### Linting

```bash
# Run ruff linter
uv run ruff check edam_mcp/

# Fix issues automatically
uv run ruff check --fix edam_mcp/

# Run mypy type checking
uv run mypy edam_mcp/
```

## 📚 Documentation

### Building Documentation

```bash
# Install documentation dependencies
uv sync --extra dev

# Build documentation
uv run mkdocs build

# Serve documentation locally
uv run mkdocs serve

# Deploy to GitHub Pages
uv run mkdocs gh-deploy
```

### Documentation Structure

```
docs/
├── mkdocs.yml              # MkDocs configuration
├── index.md                # Home page
├── getting-started/        # User guides
├── developer/              # Developer documentation
├── examples/               # Usage examples
├── contributing/           # Contribution guides
└── stylesheets/            # Custom CSS
```

## 🐛 Debugging

### Debug Configuration

#### VS Code Debug Configuration (`.vscode/launch.json`):

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "EDAM_LOG_LEVEL": "DEBUG"
            }
        },
        {
            "name": "Python: MCP Server",
            "type": "python",
            "request": "launch",
            "module": "edam_mcp.main",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "EDAM_LOG_LEVEL": "DEBUG"
            }
        }
    ]
}
```

### Logging

```python
import logging

# Set up logging for development
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use in your code
logger = logging.getLogger(__name__)
logger.debug("Debug message")
logger.info("Info message")
logger.warning("Warning message")
logger.error("Error message")
```

## 🚀 Running the Server

### Development Mode

```bash
# Run with debug logging
EDAM_LOG_LEVEL=DEBUG uv run python -m edam_mcp.main

# Run with custom configuration
EDAM_SIMILARITY_THRESHOLD=0.5 uv run python -m edam_mcp.main
```

### Testing with MCP Clients

1. **Claude Desktop**:
   ```json
   {
     "mcpServers": {
       "edam-mcp-dev": {
         "command": "uv",
         "args": ["run", "python", "-m", "edam_mcp.main"],
         "env": {
           "EDAM_LOG_LEVEL": "DEBUG"
         }
       }
     }
   }
   ```

2. **ChatGPT**:
   - Use the same configuration as above

## 🔄 Development Workflow

### 1. Feature Development

```bash
# Create feature branch
git checkout -b feature/new-tool

# Make changes
# ... edit files ...

# Run tests
uv run pytest

# Format code
uv run black edam_mcp/
uv run isort edam_mcp/

# Check types
uv run mypy edam_mcp/

# Commit changes
git add .
git commit -m "feat: add new tool for concept validation"
```

### 2. Pull Request Process

1. **Create PR** on GitHub
2. **Run CI checks** (automated)
3. **Code review** by maintainers
4. **Address feedback** if needed
5. **Merge** when approved

### 3. Release Process

```bash
# Update version
# Edit pyproject.toml version

# Build package
uv run python -m build

# Test installation
uv run pip install dist/*.whl

# Create release on GitHub
# Upload built packages
```

## 🆘 Troubleshooting

### Common Issues

#### 1. Import Errors

```bash
# Reinstall in development mode
uv pip install -e .

# Clear Python cache
find . -type d -name "__pycache__" -delete
find . -type f -name "*.pyc" -delete
```

#### 2. ML Model Download Issues

```bash
# Clear model cache
rm -rf ~/.cache/torch/
rm -rf ~/.cache/huggingface/

# Use smaller model for development
export EDAM_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

#### 3. Memory Issues

```bash
# Reduce memory usage
export EDAM_MAX_SUGGESTIONS=3
export EDAM_SIMILARITY_THRESHOLD=0.8

# Use smaller ontology for development
export EDAM_ONTOLOGY_URL="file:///path/to/small-ontology.owl"
```

#### 4. Test Failures

```bash
# Run tests with more output
uv run pytest -v -s

# Run specific test
uv run pytest tests/test_mapping.py::test_specific_function -v -s

# Check test coverage
uv run pytest --cov=edam_mcp --cov-report=term-missing
```

### Getting Help

- **GitHub Issues**: Create an issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the docs folder for detailed guides
- **Code Examples**: Look at the examples folder for usage patterns 