"""MCP tools for EDAM ontology operations."""

from .keywords_mapper import map_keywords_to_edam_concepts
from .mapping import map_to_edam_concept
from .suggestion import suggest_new_concept

__all__ = ["map_to_edam_concept", "suggest_new_concept", "map_keywords_to_edam_concepts"]
