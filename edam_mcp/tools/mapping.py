"""MCP tool for mapping descriptions to EDAM concepts."""

import logging
from typing import Optional

from fastmcp.server import Context

from ..models.requests import MappingRequest
from ..models.responses import MappingResponse, ConceptMatch
from ..ontology import OntologyLoader, ConceptMatcher

logger = logging.getLogger(__name__)


async def map_to_edam_concept(
    request: MappingRequest,
    context: Context
) -> MappingResponse:
    """Map a description to existing EDAM concepts.
    
    This tool takes a description (metadata, free text) and finds the most
    appropriate mappings to concepts in the EDAM ontology. It returns matches
    with confidence scores, indicating how well each concept matches the description.
    
    Args:
        request: Mapping request containing description and parameters.
        context: MCP context for logging and progress reporting.
        
    Returns:
        Mapping response with matched concepts and confidence scores.
    """
    try:
        # Log the request
        context.log.info(f"Mapping description: {request.description[:100]}...")
        
        # Initialize ontology components
        ontology_loader = OntologyLoader()
        if not ontology_loader.load_ontology():
            raise RuntimeError("Failed to load EDAM ontology")
        
        concept_matcher = ConceptMatcher(ontology_loader)
        
        # First try exact matches
        exact_matches = concept_matcher.find_exact_matches(request.description)
        
        if exact_matches:
            context.log.info(f"Found {len(exact_matches)} exact matches")
            return MappingResponse(
                matches=exact_matches,
                total_matches=len(exact_matches),
                has_exact_match=True,
                confidence_threshold=request.min_confidence
            )
        
        # Perform semantic matching
        context.log.info("Performing semantic matching...")
        matches = concept_matcher.match_concepts(
            description=request.description,
            context=request.context,
            max_results=request.max_results,
            min_confidence=request.min_confidence
        )
        
        context.log.info(f"Found {len(matches)} semantic matches")
        
        return MappingResponse(
            matches=matches,
            total_matches=len(matches),
            has_exact_match=False,
            confidence_threshold=request.min_confidence
        )
        
    except Exception as e:
        context.log.error(f"Error in concept mapping: {e}")
        raise


# Alternative function signature for direct use
async def map_description_to_concepts(
    description: str,
    context: Optional[str] = None,
    max_results: int = 5,
    min_confidence: float = 0.5
) -> MappingResponse:
    """Alternative interface for mapping descriptions to concepts.
    
    Args:
        description: Text description to map.
        context: Additional context information.
        max_results: Maximum number of results to return.
        min_confidence: Minimum confidence threshold.
        
    Returns:
        Mapping response with matched concepts.
    """
    request = MappingRequest(
        description=description,
        context=context,
        max_results=max_results,
        min_confidence=min_confidence
    )
    
    # Create a mock context for standalone use
    class MockContext:
        def __init__(self):
            self.log = logging.getLogger(__name__)
    
    mock_context = MockContext()
    
    return await map_to_edam_concept(request, mock_context) 