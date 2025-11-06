"""Models for text segmentation functionality."""

from pydantic import BaseModel, Field


class SegmentationRequest(BaseModel):
    """Request model for text segmentation."""

    text: str = Field(
        ...,
        description="Raw text to be segmented into topic and keywords",
        min_length=1,
    )


class SegmentationResponse(BaseModel):
    """Represents a segmented textual input by topic and keywords."""

    topic: str = Field(..., description="Brief assertion on the topic of the input text")

    keywords: list[str] = Field(..., description="List of keywords derived from the input text")

