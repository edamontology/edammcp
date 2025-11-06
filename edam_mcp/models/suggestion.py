"""Models for EDAM concept suggestion functionality."""

from pydantic import BaseModel, Field


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


class SuggestedConcept(BaseModel):
    """Represents a suggested new EDAM concept."""

    suggested_label: str = Field(..., description="Suggested label for the new concept")

    suggested_uri: str = Field(..., description="Suggested URI for the new concept")

    concept_type: str = Field(..., description="Type of the suggested concept")

    definition: str = Field(..., description="Definition for the suggested concept")

    parent_concept: str | None = Field(None, description="Suggested parent concept URI")

    rationale: str = Field(..., description="Rationale for suggesting this concept")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the suggestion quality")


class SuggestionResponse(BaseModel):
    """Response model for concept suggestions."""

    suggestions: list[SuggestedConcept] = Field(..., description="List of suggested concepts")

    total_suggestions: int = Field(..., description="Total number of suggestions generated")

    mapping_attempted: bool = Field(..., description="Whether concept mapping was attempted first")

    mapping_failed_reason: str | None = Field(None, description="Reason why mapping failed (if applicable)")

