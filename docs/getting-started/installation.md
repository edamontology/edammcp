# Installation Guide

This guide covers different ways to install and set up the EDAM MCP Server.

## 🚀 Quick Installation

### Prerequisites

- **Python 3.12+**: The server requires Python 3.12 or higher
- **uv** (recommended): Fast Python package manager
- **Git**: For cloning the repository

### Option 1: Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/edammcp.git
cd edammcp

# Install with uv
uv sync --dev

# Install in development mode
uv pip install -e .
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/your-username/edammcp.git
cd edammcp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Option 3: Using conda

```bash
# Clone the repository
git clone https://github.com/your-username/edammcp.git
cd edammcp

# Create conda environment
conda create -n edammcp python=3.12
conda activate edammcp

# Install dependencies
pip install -e .
```

## 🔧 Development Installation

For development work, install with all development dependencies:

```bash
# Install development dependencies
uv sync --dev

# Install in development mode
uv pip install -e .

# Verify installation
uv run python examples/simple_test.py
```

## 📦 Package Installation

### From PyPI (when available)

```bash
pip install edam-mcp-server
```

### From Source

```bash
# Clone and install
git clone https://github.com/your-username/edammcp.git
cd edammcp
pip install .
```

## 🐳 Docker Installation

### Using Docker

```bash
# Build the image
docker build -t edam-mcp-server .

# Run the container
docker run -p 8000:8000 edam-mcp-server
```

### Using Docker Compose

```yaml
# docker-compose.yml
version: "3.8"
services:
  edam-mcp:
    build: .
    ports:
      - "8000:8000"
    environment:
      - EDAM_LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
```

```bash
# Run with docker-compose
docker-compose up -d
```

## 🔍 Verification

### Test Basic Functionality

```bash
# Test basic imports and server creation
uv run python examples/simple_test.py
```

Expected output:

```
✅ Basic imports successful
✅ Server creation successful
✅ Configuration loading successful
```

### Test Full Functionality

```bash
# Test with ML models (downloads on first run)

# Run the mapper
uv run python examples/basic_usage_mapper.py

# Run the suggester
uv run python examples/basic_usage_suggester.py
```

Expected output:

```
✅ Ontology loading successful
✅ Concept mapping successful
✅ Concept suggestion successful
```

### Test MCP Server

```bash
# Start the MCP server
uv run python -m edam_mcp.main
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# EDAM Ontology Configuration
EDAM_ONTOLOGY_URL=https://raw.githubusercontent.com/edamontology/edamontology/master/EDAM_dev.owl

# Matching Configuration
EDAM_SIMILARITY_THRESHOLD=0.7
EDAM_MAX_SUGGESTIONS=5

# Model Configuration
EDAM_EMBEDDING_MODEL=all-MiniLM-L6-v2

# Cache Configuration
EDAM_CACHE_TTL=3600

# Logging Configuration
EDAM_LOG_LEVEL=INFO
```

### Configuration File

You can also use a configuration file:

```python
# config.py
from edam_mcp.config import Settings

settings = Settings(
    ontology_url="https://raw.githubusercontent.com/edamontology/edamontology/master/EDAM_dev.owl",
    similarity_threshold=0.7,
    max_suggestions=5,
    embedding_model="all-MiniLM-L6-v2",
    cache_ttl=3600,
    log_level="INFO"
)
```

## 🚀 First Run

### 1. Start the Server

```bash
# Basic start
uv run python -m edam_mcp.main

# With custom configuration
EDAM_LOG_LEVEL=DEBUG uv run python -m edam_mcp.main
```

### 2. Test with MCP Client

Configure your MCP client (e.g., Claude Desktop):

```json
{
  "mcpServers": {
    "edam-mcp": {
      "command": "uv",
      "args": ["run", "python", "-m", "edam_mcp.main"],
      "env": {
        "EDAM_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### 3. Test Tools

Once connected, test the available tools:

- `map_to_edam_concept`: Map descriptions to EDAM concepts
- `suggest_new_concept`: Suggest new concepts

## 🔧 Troubleshooting

### Common Issues

#### 1. Python Version Issues

```bash
# Check Python version
python --version

# Should be 3.12 or higher
# If not, install Python 3.12+
```

#### 2. Dependency Issues

```bash
# Clear and reinstall dependencies
rm -rf .venv/
uv sync --dev
uv pip install -e .
```

#### 3. ML Model Download Issues

```bash
# Clear model cache
rm -rf ~/.cache/torch/
rm -rf ~/.cache/huggingface/

# Use smaller model
export EDAM_EMBEDDING_MODEL="all-MiniLM-L6-v2"
```

#### 4. Memory Issues

```bash
# Reduce memory usage
export EDAM_MAX_SUGGESTIONS=3
export EDAM_SIMILARITY_THRESHOLD=0.8
```

#### 5. Network Issues

```bash
# Use local ontology file
export EDAM_ONTOLOGY_URL="file:///path/to/local/edam.owl"
```

### Getting Help

- **GitHub Issues**: Create an issue for bugs
- **Discussions**: Use GitHub Discussions for questions
- **Documentation**: Check the docs folder
- **Examples**: Look at the examples folder

## 📊 System Requirements

### Minimum Requirements

- **CPU**: 2 cores
- **RAM**: 2GB
- **Storage**: 1GB free space
- **Network**: Internet connection

### Recommended Requirements

- **CPU**: 4+ cores
- **RAM**: 4GB+
- **Storage**: 2GB+ free space
- **Network**: Stable internet connection

### Performance Notes

- **First Run**: ~5 seconds (ML model download)
- **Subsequent Runs**: <1 second
- **Memory Usage**: ~500MB with models loaded
- **Concurrent Requests**: Full async support

## 🔄 Updates

### Updating the Installation

```bash
# Pull latest changes
git pull origin main

# Update dependencies
uv sync --dev

# Reinstall package
uv pip install -e .
```

### Checking for Updates

```bash
# Check for outdated packages
uv pip list --outdated

# Update specific package
uv pip install --upgrade package-name
```

## 🧹 Uninstallation

### Remove Package

```bash
# Remove installed package
pip uninstall edam-mcp-server

# Or with uv
uv pip uninstall edam-mcp-server
```

### Clean Environment

```bash
# Remove virtual environment
rm -rf .venv/

# Remove cached models
rm -rf ~/.cache/torch/
rm -rf ~/.cache/huggingface/

# Remove project files
rm -rf edammcp/
```

## 📚 Next Steps

After installation:

1. **Read the [Quick Start Guide](quickstart.md)** to get running quickly
2. **Check the [Configuration Guide](configuration.md)** for advanced setup
3. **Explore [Examples](../examples/basic-usage.md)** for usage patterns
4. **Review [API Documentation](../developer/api.md)** for detailed reference
5. **Join the [Community](../contributing/development-setup.md)** for support

