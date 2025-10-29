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

    context: str | None = Field(
        None,
        description="Additional context about the description (e.g., tool name, domain)",
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


class KeywordMappingRequest(BaseModel):
    """Request model for mapping keywords to EDAM concepts."""

    keywords: list[str] = Field(
        ...,
        description="List of keywords to match against EDAM ontology terms",
        min_length=1,
    )

    match_mode: str = Field(
        ...,
        description="Matching algorithm: 'substring', 'list_embeddings', or 'joined_embedding'",
        pattern="^(substring|list_embeddings|joined_embedding)$",
    )

    threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for embedding-based matching (default: 0.3)",
    )

    max_results: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Maximum number of results to return (optional)",
    )
