"""MCP tools for EDAM ontology operations."""

from .mapping import map_to_edam_concept
from .suggestion import suggest_new_concept
from .query_biotools import query_biotools

__all__ = ["map_to_edam_concept", "suggest_new_concept", "query_biotools"]
