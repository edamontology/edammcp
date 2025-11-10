"""MCP tools for EDAM ontology operations."""

from .mapping import map_to_edam_concept, map_to_edam_operation
from .suggestion import suggest_new_concept

__all__ = ["map_to_edam_concept", "map_to_edam_operation", "suggest_new_concept"]
