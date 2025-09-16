"""EDAM ontology loading and parsing functionality."""

import logging
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import requests
from rdflib import Graph, Namespace, URIRef, RDF
from rdflib.namespace import RDFS, SKOS

from ..config import settings

logger = logging.getLogger(__name__)

# EDAM namespaces
EDAM = Namespace("http://edamontology.org/")
OWL = Namespace("http://www.w3.org/2002/07/owl#")


class OntologyLoader:
    """Handles loading and parsing of the EDAM ontology."""
    
    def __init__(self, ontology_url: Optional[str] = None):
        """Initialize the ontology loader.
        
        Args:
            ontology_url: URL to the EDAM ontology file. Defaults to settings.
        """
        self.ontology_url = ontology_url or settings.edam_ontology_url
        self.graph: Optional[Graph] = None
        self.concepts: Dict[str, Dict] = {}
        self.concept_types: Set[str] = set()
        
    def load_ontology(self) -> bool:
        """Load the EDAM ontology from the configured URL.
        
        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            logger.info(f"Loading EDAM ontology from {self.ontology_url}")
            
            # Download ontology file
            response = requests.get(self.ontology_url, timeout=30)
            response.raise_for_status()
            
            # Parse RDF/OWL content
            self.graph = Graph()
            self.graph.parse(data=response.content, format="xml")
            
            # Extract concepts
            self._extract_concepts()
            
            logger.info(f"Successfully loaded {len(self.concepts)} concepts")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            return False
    
    def _extract_concepts(self) -> None:
        """Extract concept information from the loaded graph."""
        if not self.graph:
            return
            
        for concept_uri in self.graph.subjects(RDF.type, OWL.Class):
            if not str(concept_uri).startswith(str(EDAM)):
                continue
                
            concept_data = self._extract_concept_data(concept_uri)
            if concept_data:
                self.concepts[str(concept_uri)] = concept_data
                self.concept_types.add(concept_data["type"])
    
    def _extract_concept_data(self, concept_uri: URIRef) -> Optional[Dict]:
        """Extract data for a single concept.
        
        Args:
            concept_uri: URI of the concept to extract data for.
            
        Returns:
            Dictionary containing concept data or None if invalid.
        """
        try:
            # Get label
            label = self._get_literal_value(concept_uri, RDFS.label)
            if not label:
                return None
                
            # Get definition
            definition = self._get_literal_value(concept_uri, SKOS.definition)
            
            # Get synonyms
            synonyms = self._get_literal_values(concept_uri, SKOS.altLabel)
            
            # Determine concept type from URI
            concept_type = self._determine_concept_type(str(concept_uri))
            
            return {
                "uri": str(concept_uri),
                "label": label,
                "definition": definition,
                "synonyms": synonyms,
                "type": concept_type,
                "parents": self._get_parent_concepts(concept_uri),
                "children": self._get_child_concepts(concept_uri)
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract data for {concept_uri}: {e}")
            return None
    
    def _get_literal_value(self, subject: URIRef, predicate: URIRef) -> Optional[str]:
        """Get a single literal value for a subject-predicate pair."""
        for obj in self.graph.objects(subject, predicate):
            return str(obj)
        return None
    
    def _get_literal_values(self, subject: URIRef, predicate: URIRef) -> List[str]:
        """Get all literal values for a subject-predicate pair."""
        return [str(obj) for obj in self.graph.objects(subject, predicate)]
    
    def _determine_concept_type(self, uri: str) -> str:
        """Determine the concept type from its URI."""
        if "operation" in uri.lower():
            return "Operation"
        elif "data" in uri.lower():
            return "Data"
        elif "format" in uri.lower():
            return "Format"
        elif "topic" in uri.lower():
            return "Topic"
        elif "identifier" in uri.lower():
            return "Identifier"
        else:
            return "Unknown"
    
    def _get_parent_concepts(self, concept_uri: URIRef) -> List[str]:
        """Get parent concepts for a given concept."""
        parents = []
        for parent in self.graph.objects(concept_uri, RDFS.subClassOf):
            if str(parent).startswith(str(EDAM)):
                parents.append(str(parent))
        return parents
    
    def _get_child_concepts(self, concept_uri: URIRef) -> List[str]:
        """Get child concepts for a given concept."""
        children = []
        for child in self.graph.subjects(RDFS.subClassOf, concept_uri):
            if str(child).startswith(str(EDAM)):
                children.append(str(child))
        return children
    
    def get_concept(self, uri: str) -> Optional[Dict]:
        """Get concept data by URI."""
        return self.concepts.get(uri)
    
    def get_concepts_by_type(self, concept_type: str) -> List[Dict]:
        """Get all concepts of a specific type."""
        return [
            concept for concept in self.concepts.values()
            if concept["type"] == concept_type
        ]
    
    def search_concepts(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search concepts by label, definition, or synonyms.
        
        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            
        Returns:
            List of matching concepts.
        """
        query_lower = query.lower()
        matches = []
        
        for concept in self.concepts.values():
            # Check label
            if query_lower in concept["label"].lower():
                matches.append(concept)
                continue
                
            # Check definition
            if concept["definition"] and query_lower in concept["definition"].lower():
                matches.append(concept)
                continue
                
            # Check synonyms
            for synonym in concept["synonyms"]:
                if query_lower in synonym.lower():
                    matches.append(concept)
                    break
        
        return matches[:max_results]
    
    def get_concept_hierarchy(self, concept_uri: str) -> List[str]:
        """Get the full hierarchy path for a concept."""
        hierarchy = []
        current_uri = concept_uri
        
        while current_uri:
            concept = self.get_concept(current_uri)
            if concept:
                hierarchy.insert(0, concept["label"])
                current_uri = concept["parents"][0] if concept["parents"] else None
            else:
                break
        
        return hierarchy 