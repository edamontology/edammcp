"""Request models for MCP tools."""

from pydantic import BaseModel, Field


class MappingRequest(BaseModel):
    """Request model for mapping descriptions to EDAM concepts."""

    description: str = Field(
        ...,
        description=(
            "REQUIRED: A description of the concept, operation, data type, format, or topic "
            "to map to EDAM ontology terms. "
            "You MUST extract this from the user's question or context. "
            "Look for descriptive text after colons, dashes, or in the question itself. "
            "Examples: 'graphical inspection of spectral data', 'mass spectrometry data', "
            "'quantitative analysis without internal standards'. "
            "If the user's question contains a description (e.g., after '—' or ':'), "
            "use that text as the description. "
            "This field is REQUIRED and cannot be empty."
        ),
        min_length=1,
        max_length=10000,
    )

    context: str | None = Field(
        None,
        description=(
            "OPTIONAL: Additional context about the description to help with mapping. "
            "Examples: 'tool name', 'domain', 'related terms'. "
            "This field is OPTIONAL and can be empty."
        ),
        max_length=2000,
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
