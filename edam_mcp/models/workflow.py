"""Workflow models for the EDAM MCP mapping workflow."""

from pydantic import BaseModel, Field


class WorkflowSummaryRequest(BaseModel):
    """Request model for workflow summary (no parameters needed)."""

    pass


class WorkflowFunction(BaseModel):
    """Model describing a workflow function."""

    name: str = Field(..., description="Function name")
    description: str = Field(..., description="Function description and purpose")
    input_format: dict = Field(..., description="Expected input parameters and types")
    output_format: dict = Field(..., description="Expected output structure and types")
    configurable_options: list[str] = Field(default_factory=list, description="List of configurable options/parameters")
    dependencies: list[str] = Field(default_factory=list, description="Other workflow functions this depends on")
    existing_implementation: str | None = Field(
        None,
        description="Name of existing tool/function that provides partial or full implementation (if any)",
    )


class WorkflowSummaryResponse(BaseModel):
    """Response model containing the complete workflow summary."""

    workflow_name: str = Field(..., description="Name of the workflow")
    workflow_description: str = Field(..., description="Overall workflow description")
    workflow_steps: list[str] = Field(..., description="Ordered list of workflow step names")
    functions: list[WorkflowFunction] = Field(..., description="Detailed description of all available functions")
    configurable_options: dict = Field(..., description="Global configurable options and their defaults")
    workflow_flow: str = Field(..., description="Textual description of the workflow flow")
