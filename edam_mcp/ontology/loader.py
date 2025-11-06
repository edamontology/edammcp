"""EDAM ontology loading and parsing functionality."""

import logging
import os
import pickle
import time

from rdflib import RDF, Graph, Namespace, URIRef
from rdflib.namespace import OWL, RDFS, SKOS

from ..config import settings

logger = logging.getLogger(__name__)

# EDAM namespaces
EDAM = Namespace("http://edamontology.org/")
OBOINOWL = Namespace("http://www.geneontology.org/formats/oboInOwl#")


class OntologyLoader:
    """Handles loading and parsing of the EDAM ontology."""

    def __init__(self, ontology_url: str | None = None):
        """Initialize the ontology loader.

        Args:
            ontology_url: URL to the EDAM ontology file. Defaults to settings.
        """
        self.ontology_url = ontology_url or settings.ontology_url
        self.graph: Graph | None = None
        self.concepts: dict[str, dict] = {}
        self.concept_types: set[str] = set()
        self.cache_dir = settings.cache_dir
        self.cache_ttl = settings.cache_ttl

    def _cache_paths(self):
        os.makedirs(self.cache_dir, exist_ok=True)
        return (
            os.path.join(self.cache_dir, "concepts.pkl"),
            os.path.join(self.cache_dir, "concept_types.pkl"),
            os.path.join(self.cache_dir, "cache.timestamp"),
        )

    def _is_cache_fresh(self):
        _, _, ts_path = self._cache_paths()
        if not os.path.exists(ts_path):
            return False
        cache_time = os.path.getmtime(ts_path)
        return (time.time() - cache_time) < self.cache_ttl

    def _load_cache(self):
        concepts_path, types_path, _ = self._cache_paths()
        with open(concepts_path, "rb") as f:
            self.concepts = pickle.load(f)
        with open(types_path, "rb") as f:
            self.concept_types = pickle.load(f)

    def _save_cache(self):
        concepts_path, types_path, ts_path = self._cache_paths()
        with open(concepts_path, "wb") as f:
            pickle.dump(self.concepts, f)
        with open(types_path, "wb") as f:
            pickle.dump(self.concept_types, f)
        with open(ts_path, "w") as f:
            f.write(str(time.time()))

    def load_ontology(self) -> bool:
        """Load the EDAM ontology from the configured URL.

        Returns:
            True if loading was successful, False otherwise.
        """
        try:
            if self._is_cache_fresh():
                logger.info("Loading concepts from cache")
                try:
                    self._load_cache()
                    return True
                except Exception as cache_exc:
                    logger.warning(f"Failed to load cache: {cache_exc}. Falling back to loading from ontology URL.")

            logger.info(f"Loading EDAM ontology from {self.ontology_url}")

            # Parse RDF/OWL content
            self.graph = Graph()
            self.graph.bind("edam", EDAM)
            self.graph.bind("owl", OWL)
            self.graph.bind("skos", SKOS)

            self.graph.parse(self.ontology_url)

            # Extract concepts
            self._extract_concepts()

            logger.info(f"Successfully loaded {len(self.concepts)} concepts")

            self._save_cache()
            return True

        except Exception as e:
            logger.error(f"Failed to load ontology: {e}")
            return False

    def _extract_concepts(self) -> None:
        """Extract concept information from the loaded graph."""
        if not self.graph:
            logger.error("No EDAM OWL file loaded.")
            return

        obsolete_count = 0
        for concept_uri in self.graph.subjects(RDF.type, OWL.Class):
            if not str(concept_uri).startswith(str(EDAM)):
                continue

            concept_data = self._extract_concept_data(concept_uri)
            if concept_data:
                if not concept_data.get("obsolete", False):
                    self.concepts[str(concept_uri)] = concept_data
                    self.concept_types.add(concept_data["type"])
                else:
                    obsolete_count += 1

        logger.info(f"Skipped {obsolete_count} obsolete concepts")

    def _extract_concept_data(self, concept_uri: URIRef) -> dict | None:
        """Extract data for a single concept.

        Args:
            concept_uri: URI of the concept to extract data for.

        Returns:
            Dictionary containing concept data or None if invalid.
        """
        try:
            label = self._get_literal_value(concept_uri, RDFS.label)
            if not label:
                return None

            definition = self._get_literal_value(concept_uri, SKOS.definition) or self._get_literal_value(
                concept_uri, OBOINOWL.hasDefinition
            )

            synonyms = self._get_literal_values(concept_uri, SKOS.altLabel) + self._get_literal_values(
                concept_uri, OBOINOWL.hasExactSynonym
            )

            # Determine concept type from URI
            concept_type = self._determine_concept_type(str(concept_uri))

            # Determine if a concept is obsolete
            obsolete = self._get_literal_value(concept_uri, OWL.deprecated)

            return {
                "uri": str(concept_uri),
                "label": label,
                "definition": definition,
                "synonyms": synonyms,
                "type": concept_type,
                "parents": self._get_parent_concepts(concept_uri),
                "children": self._get_child_concepts(concept_uri),
                "obsolete": obsolete == "true",
            }

        except Exception as e:
            logger.warning(f"Failed to extract data for {concept_uri}: {e}")
            return None

    def _get_literal_value(self, subject: URIRef, predicate: URIRef) -> str | None:
        """Get a single literal value for a subject-predicate pair."""
        for obj in self.graph.objects(subject, predicate):
            return str(obj)
        return None

    def _get_literal_values(self, subject: URIRef, predicate: URIRef) -> list[str]:
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

    def _get_parent_concepts(self, concept_uri: URIRef) -> list[str]:
        """Get parent concepts for a given concept."""
        parents = []
        for parent in self.graph.objects(concept_uri, RDFS.subClassOf):
            if str(parent).startswith(str(EDAM)):
                parents.append(str(parent))
        return parents

    def _get_child_concepts(self, concept_uri: URIRef) -> list[str]:
        """Get child concepts for a given concept."""
        children = []
        for child in self.graph.subjects(RDFS.subClassOf, concept_uri):
            if str(child).startswith(str(EDAM)):
                children.append(str(child))
        return children

    def get_concept(self, uri: str) -> dict | None:
        """Get concept data by URI."""
        return self.concepts.get(uri)

    def get_concepts_by_type(self, concept_type: str) -> list[dict]:
        """Get all concepts of a specific type."""
        return [concept for concept in self.concepts.values() if concept["type"] == concept_type]

    def search_concepts(self, query: str, max_results: int = 10) -> list[dict]:
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

    def get_concept_hierarchy(self, concept_uri: str) -> list[str]:  # TODO: currently  not used
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
