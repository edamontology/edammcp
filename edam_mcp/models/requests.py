"""Request models for MCP tools."""

from typing import Optional

from pydantic import BaseModel, Field


class MappingRequest(BaseModel):
    """Request model for mapping descriptions to EDAM concepts."""
    
    description: str = Field(
        ...,
        description="Text description or metadata to map to EDAM concepts",
        min_length=1,
        max_length=10000
    )
    
    context: Optional[str] = Field(
        None,
        description="Additional context about the description (e.g., tool name, domain)",
        max_length=2000
    )
    
    max_results: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Maximum number of concept matches to return"
    )
    
    min_confidence: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for matches"
    )


class SuggestionRequest(BaseModel):
    """Request model for suggesting new EDAM concepts."""
    
    description: str = Field(
        ...,
        description="Description of the concept that needs to be suggested",
        min_length=1,
        max_length=10000
    )
    
    concept_type: Optional[str] = Field(
        None,
        description="Type of concept (e.g., 'Operation', 'Data', 'Format', 'Topic')",
        pattern="^(Operation|Data|Format|Topic|Identifier)$"
    )
    
    parent_concept: Optional[str] = Field(
        None,
        description="Suggested parent concept URI or label",
        max_length=500
    )
    
    rationale: Optional[str] = Field(
        None,
        description="Rationale for why this concept should be added",
        max_length=2000
    ) 