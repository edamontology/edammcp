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


class MappingResponse(BaseModel):
    """Response model for concept mapping results."""

    matches: list[ConceptMatch] = Field(..., description="List of matched concepts ordered by confidence")
    total_matches: int = Field(..., description="Total number of matches found")
    has_exact_match: bool = Field(..., description="Whether an exact match was found")
    confidence_threshold: float = Field(..., description="Confidence threshold used for filtering")


class BiotoolsResponse(BaseModel):
    """Response model for bio.tools concept results."""

    matches: list[ConceptMatch] = Field(..., description="List of obtained concepts")
    total_matches: int = Field(..., description="Total number of concepts found")


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


class BioToolsInfo(BaseModel):
    """Model for information retrieved from bio.tools."""

    name: str = Field(..., description="The name of the tool")
    biotools_curie: str = Field(..., description="The biotools CURIE of the tool")
    description: str = Field(..., description="The description of the tool")
    operation: list[dict[str, str]] = Field(
        ..., description="A list of ontology terms describing the operation of the tool"
    )
    input: list[dict[str, dict[str, str] | list[dict[str, str]]]] = Field(
        ..., description="A list of ontology terms describing the input(s) of the tool"
    )
    output: list[dict[str, dict[str, str] | list[dict[str, str]]]] = Field(
        ..., description="A list of ontology terms describing the output(s) of the tool"
    )
    topic: list[dict[str, str]] = Field(..., description="A list of ontology terms describing the topic of the tool")
