#!/usr/bin/env python3
"""Setup script for development environment."""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
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
    """Main setup function."""
    print("🚀 Setting up EDAM MCP development environment with uv")
    print("=" * 60)
    
    # Check if uv is installed
    if not run_command(["uv", "--version"], "Checking uv installation"):
        print("❌ uv is not installed. Please install uv first:")
        print("   curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    
    # Install dependencies
    if not run_command(["uv", "sync", "--dev"], "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests to verify installation
    if not run_command(["uv", "run", "pytest", "--version"], "Verifying pytest installation"):
        print("❌ Failed to verify pytest installation")
        sys.exit(1)
    
    # Format code
    if not run_command(["uv", "run", "black", "--version"], "Verifying black installation"):
        print("❌ Failed to verify black installation")
        sys.exit(1)
    
    print("\n🎉 Development environment setup completed!")
    print("\nNext steps:")
    print("1. Run tests: uv run pytest")
    print("2. Format code: uv run black edam_mcp/")
    print("3. Start the server: uv run python -m edam_mcp.main")
    print("4. Run examples: uv run python examples/basic_usage.py")


if __name__ == "__main__":
    main() 