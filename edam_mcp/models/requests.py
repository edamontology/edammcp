"""Request models for MCP tools."""

from pydantic import BaseModel, Field


class MappingRequest(BaseModel):
    """Request model for mapping descriptions to EDAM concepts."""

    description: str = Field(
        ...,
        description="Text description or metadata to map to EDAM concepts",
        min_length=1,
        max_length=10000,
    )
    name: str | None = Field(
        None,
        description="The name of the tool being processed.",
        min_length=1,
        max_length=20,
    )
    biotools_curie: str | None = Field(
        None,
        description="The biotoolsCURIE of the tool being processed.",
        min_length=1,
        max_length=20,
    )
    context: str | None = Field(
        None,
        description="Additional context about the description (e.g., tool name, domain)",
        max_length=2000,
    )
    ontology_type: str | None = Field(
        None,
        description="What ontology terms to retrieve from bio.tools. Can be one of [operation, input, output, topic]",
        pattern="^(operation|input|output|topic)$",
    )
    max_results: int | None = Field(5, ge=1, le=20, description="Maximum number of concept matches to return")
    min_confidence: float | None = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold for matches")


class SuggestionRequest(BaseModel):
    """Request model for suggesting new EDAM concepts."""

    description: str = Field(
        ...,
        description="Description of the concept that needs to be suggested",
        min_length=1,
        max_length=10000,
    )
    concept_type: str | None = Field(
        None,
        description="Type of concept (e.g., 'Operation', 'Data', 'Format', 'Topic')",
        pattern="^(Operation|Data|Format|Topic|Identifier)$",
    )
    parent_concept: str | None = Field(None, description="Suggested parent concept URI or label", max_length=500)
    rationale: str | None = Field(
        None,
        description="Rationale for why this concept should be added",
        max_length=2000,
    )
