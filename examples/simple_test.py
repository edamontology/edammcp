#!/usr/bin/env python3
"""Simple test script to verify the basic structure without heavy dependencies."""

import importlib
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_basic_imports():
    """Test basic imports without heavy dependencies."""
    print("Testing basic imports...")

    try:
        # Test config
        from edam_mcp.config import settings

        print("✅ Config imported successfully")
        print(f"   Similarity threshold: {settings.similarity_threshold}")

        requests_spec = importlib.util.find_spec("edam_mcp.models.requests")
        responses_spec = importlib.util.find_spec("edam_mcp.models.responses")

        if requests_spec is None:
            print("Module 'edam_mcp.models.requests' is not available")
            return False
        if responses_spec is None:
            print("Module 'edam_mcp.models.responses' is not available")
            return False

        requests_mod = importlib.import_module("edam_mcp.models.requests")
        responses_mod = importlib.import_module("edam_mcp.models.responses")

        for cls_name in ("MappingRequest", "SuggestionRequest"):
            if not hasattr(requests_mod, cls_name):
                print(f"Class '{cls_name}' not found in edam_mcp.models.requests")
                return False

        for cls_name in ("ConceptMatch", "MappingResponse", "SuggestionResponse"):
            if not hasattr(responses_mod, cls_name):
                print(f"Class '{cls_name}' not found in edam_mcp.models.responses")
                return False

        mapping_request = getattr(requests_mod, "MappingRequest")

        print("✅ Request models imported successfully")

        print("✅ Response models imported successfully")

        # Test utils
        from edam_mcp.utils.text_processing import preprocess_text

        print("✅ Text processing utilities imported successfully")

        # Test basic functionality
        test_text = "sequence alignment tool"
        processed = preprocess_text(test_text)
        print(f"✅ Text processing works: '{test_text}' -> '{processed}'")

        # Test model creation
        _ = mapping_request(
            description="test description",
            context="test context",
            max_results=5,
            min_confidence=0.7,
        )
        print("✅ Request model creation works")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_server_creation():
    """Test server creation without running it."""
    print("\nTesting server creation...")

    try:
        from edam_mcp.main import create_server

        server = create_server()
        print("✅ Server creation successful")
        print(f"   Server name: {server.name}")
        print(f"   Server type: {type(server).__name__}")

        return True

    except Exception as e:
        print(f"❌ Server creation failed: {e}")
        return False


def main():
    """Run the simple tests."""
    print("🧪 Simple EDAM MCP Test")
    print("=" * 40)

    # Test basic imports
    if not test_basic_imports():
        print("\n❌ Basic imports failed")
        return

    # Test server creation
    if not test_server_creation():
        print("\n❌ Server creation failed")
        return

    print("\n🎉 All basic tests passed!")
    print("\nNote: This test doesn't include the heavy ML dependencies.")
    print("To test the full functionality, you'll need to run the main example.")


if __name__ == "__main__":
    main()
