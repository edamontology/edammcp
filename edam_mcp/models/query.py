"""Models for Biotools Query functionality."""

from pydantic import BaseModel, Field

class BiotoolsQueryRequest(BaseModel):
    """Request model for biotools query."""

    seed: str = Field(
        ...,
        description="Search keyword to search in tool names and descriptions.",
        min_length=1,
        max_length=10000,
    )

    max_results: int | None = Field(10, ge=1, le=100, description="Maximum number of query hits to return")

class BiotoolsHit(BaseModel):
    """Represents a query hit item (tool) with the name, description and the edam terms found for the ontology categories."""

    name: str = Field(..., description="Tool name")

    description: str = Field(..., description="Tool description")

    biotools_link: str = Field(..., description="Tool link in biotools")

    topic_terms: list[str] = Field(default_factory=list, description="List of uris of the EDAM terms beloging to topic category")

    operation_terms: list[str] = Field(default_factory=list, description="List of uris of the EDAM terms beloging to operation category")

    data_terms: list[str] = Field(default_factory=list, description="List of uris of the EDAM terms beloging to data category")

    format_terms: list[str] = Field(default_factory=list, description="List of uris of the EDAM terms beloging to format category")

class BiotoolsQueryResponse(BaseModel):
    """Response model for biotools query results."""

    matches: list[BiotoolsHit] = Field(..., description="List of matched tools in biotools database")

    total_matches: int = Field(..., description="Total number of matches found")