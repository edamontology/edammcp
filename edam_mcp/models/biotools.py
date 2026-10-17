"""Models for Biotools Query functionality."""

from pydantic import BaseModel, Field, model_validator


class BiotoolsQueryTextRequest(BaseModel):
    """Request model for biotools query by text."""

    seed: str = Field(
        ...,
        description="Keywords or tool name to search in bio.tools",
        min_length=1,
        max_length=200,
    )

    max_results: int = Field(10, ge=1, le=100, description="Maximum number of matches to return")

class BiotoolsQueryExactRequest(BaseModel):
    """Request model for biotools query by text with exact match."""

    name: str = Field(
        None,
        description="Tool name to search in bio.tools",
        min_length=1,
        max_length=200,
    )

    biotools_id: str = Field(
        None,
        description="BioTools normalized identifier to search in bio.tools",
        min_length=1,
        max_length=200,
    )

    @model_validator(mode='after')
    def check_input_presence(self):
        flag = (self.name is not None) or (self.biotools_id is not None)
        if(not flag):
            raise ValueError("You must fill either the name or the biotools ID")
        return self

class BiotoolsQueryTermsRequest(BaseModel):
    """Request model for biotools query by EDAM terms."""

    edam_terms: list[str] = Field(
        ...,
        min_length=1,
        max_length=4,
        description="EDAM term URIs that every returned tool must be annotated with, at most one per category (topic, operation, data, format)",
    )

    max_results: int = Field(10, ge=1, le=100, description="Maximum number of matches to return")


class BiotoolsMatch(BaseModel):
    """Represents a query match item (tool) with the name, description and the edam terms found for the ontology categories."""

    name: str = Field(..., description="Tool name")

    description: str = Field(..., description="Tool description")

    biotools_link: str = Field(..., description="Tool link in biotools")

    topic_terms: dict[str, dict] = Field(default_factory=dict, description="EDAM topic terms of the tool, URI to label")

    operation_terms: dict[str, dict] = Field(
        default_factory=dict, description="EDAM operation terms of the tool functions, URI to label"
    )

    input_data_terms: dict[str, dict] = Field(
        default_factory=dict, description="EDAM data terms of the tool inputs, URI to label"
    )

    input_format_terms: dict[str, dict] = Field(
        default_factory=dict, description="EDAM format terms of the tool inputs, URI to label"
    )

    output_data_terms: dict[str, dict] = Field(
        default_factory=dict, description="EDAM data terms of the tool outputs, URI to label"
    )

    output_format_terms: dict[str, dict] = Field(
        default_factory=dict, description="EDAM format terms of the tool outputs, URI to label"
    )


class BiotoolsQueryResponse(BaseModel):
    """Response model for biotools query results."""

    matches: list[BiotoolsMatch] = Field(..., description="List of matched tools in biotools database")

    total_matches: int = Field(
        ...,
        description="Total number of tools bio.tools reports for the query, including those beyond max_results",
    )
