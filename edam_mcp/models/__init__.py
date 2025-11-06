"""Pydantic models for EDAM MCP tools, organized by functionality."""

from .mapping import ConceptMatch, MappingRequest, MappingResponse
from .segmentation import SegmentationRequest, SegmentationResponse
from .suggestion import SuggestedConcept, SuggestionRequest, SuggestionResponse
from .workflow import WorkflowFunction, WorkflowSummaryRequest, WorkflowSummaryResponse

__all__ = [
    # Mapping models
    "MappingRequest",
    "MappingResponse",
    "ConceptMatch",
    # Suggestion models
    "SuggestionRequest",
    "SuggestionResponse",
    "SuggestedConcept",
    # Segmentation models
    "SegmentationRequest",
    "SegmentationResponse",
    # Workflow models
    "WorkflowSummaryRequest",
    "WorkflowSummaryResponse",
    "WorkflowFunction",
]
