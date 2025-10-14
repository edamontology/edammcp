#!/usr/bin/env python3
"""Install the package in development mode."""

import subprocess
import sys


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        _ = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"stdout: {e.stdout}")
        if e.stderr:
            print(f"stderr: {e.stderr}")
        return False


def main():
    """Main installation function."""
    print("🚀 Installing EDAM MCP in development mode")
    print("=" * 50)

    # Install dependencies
    if not run_command(["uv", "sync", "--dev"], "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)

    # Install the package in development mode
    if not run_command(["uv", "pip", "install", "-e", "."], "Installing package in development mode"):
        print("❌ Failed to install package in development mode")
        sys.exit(1)

    # Test the installation
    test_script = """
import sys
try:
    import edam_mcp
    print("✅ edam_mcp package imported successfully")
    print(f"   Version: {edam_mcp.__version__}")
except ImportError as e:
    print(f"❌ Failed to import edam_mcp: {e}")
    sys.exit(1)
"""

    print("🔄 Testing package installation...")
    try:
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Package test failed: {e}")
        print(e.stderr)
        sys.exit(1)

    print("\n🎉 Development installation completed!")
    print("\nYou can now run:")
    print("  uv run python examples/basic_usage_mapper.py")
    print("  uv run python examples/basic_usage_suggester.py")
    print("  uv run python -m edam_mcp.main")
    print("  make example")
    print("  make run")


if __name__ == "__main__":
    main()
