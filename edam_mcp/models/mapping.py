"""Models for EDAM concept mapping functionality."""

from pydantic import BaseModel, Field


class MappingRequest(BaseModel):
    """Request model for mapping descriptions to EDAM concepts."""

    description: str = Field(
        ...,
        description="Text description or metadata to map to EDAM concepts",
        min_length=1,
        max_length=10000,
    )

    context: str | None = Field(
        None,
        description="Additional context about the description (e.g., tool name, domain)",
        max_length=2000,
    )

    max_results: int | None = Field(5, ge=1, le=20, description="Maximum number of concept matches to return")

    min_confidence: float | None = Field(0.5, ge=0.0, le=1.0, description="Minimum confidence threshold for matches")


class ConceptMatch(BaseModel):
    """Represents a matched EDAM concept with confidence score."""

    concept_uri: str = Field(..., description="URI of the matched EDAM concept")

    concept_label: str = Field(..., description="Human-readable label of the concept")

    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the match (0.0 to 1.0)")

    concept_type: str = Field(
        ...,
        description="Type of the concept (Operation, Data, Format, Topic, Identifier)",
    )

    definition: str | None = Field(None, description="Definition of the concept")

    synonyms: list[str] = Field(default_factory=list, description="List of synonyms for the concept")


class MappingResponse(BaseModel):
    """Response model for concept mapping results."""

    matches: list[ConceptMatch] = Field(..., description="List of matched concepts ordered by confidence")

    total_matches: int = Field(..., description="Total number of matches found")

    has_exact_match: bool = Field(..., description="Whether an exact match was found")

    confidence_threshold: float = Field(..., description="Confidence threshold used for filtering")
