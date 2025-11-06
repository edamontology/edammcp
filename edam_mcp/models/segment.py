
"""Segmentation classes for MCP tools."""

from pydantic import BaseModel, Field


class ReadyForMapping(BaseModel):
    """Represents a segmented textual input."""

    top_concept: str = Field(..., description="Very brief assertion on topic of input text")

    chunks: list[str] = Field(..., description="List of fragments identified by NLP")

