"""Response models for MCP tools."""

from pydantic import BaseModel, Field


class ConceptMatch(BaseModel):
    """Represents a matched EDAM concept with confidence score."""

    concept_uri: str = Field(..., description="URI of the matched EDAM concept")

    concept_label: str = Field(..., description="Human-readable label of the concept")

    confidence: float | None = Field(..., ge=0.0, le=1.0, description="Confidence score for the match (0.0 to 1.0)")

    concept_type: str = Field(
        ...,
        description="Type of the concept (Operation, Data, Format, Topic, Identifier)",
    )

    definition: str | None = Field(None, description="Definition of the concept")

    synonyms: list[str] = Field(default_factory=list, description="List of synonyms for the concept")


class KeywordMatch(ConceptMatch):
    """Represents a keyword match result, extending ConceptMatch with matching metadata."""

    match_type: str = Field(
        ..., description="Type of matching algorithm used (e.g., substring, list_embeddings, joined_embedding)"
    )

    matched_keywords: list[str] = Field(..., description="List of keywords that matched this concept")


class MappingResponse(BaseModel):
    """Response model for concept mapping results."""

    matches: list[ConceptMatch] = Field(..., description="List of matched concepts ordered by confidence")

    total_matches: int = Field(..., description="Total number of matches found")

    has_exact_match: bool = Field(..., description="Whether an exact match was found")

    confidence_threshold: float = Field(..., description="Confidence threshold used for filtering")


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


class KeywordMappingResponse(BaseModel):
    """Response model for keyword mapping results."""

    matches: list[KeywordMatch] = Field(..., description="List of matched concepts ordered by confidence")

    total_matches: int = Field(..., description="Total number of matches found")

    match_mode: str = Field(
        ..., description="Matching algorithm used (substring, list_embeddings, or joined_embedding)"
    )

    threshold: float | None = Field(None, description="Similarity threshold used for filtering (if applicable)")
