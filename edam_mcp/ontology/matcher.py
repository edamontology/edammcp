"""Concept matching functionality for mapping descriptions to EDAM concepts."""

import logging

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from ..config import settings
from ..models.responses import BioToolsInfo, ConceptMatch
from ..utils.text_processing import preprocess_text
from .loader import OntologyLoader

logger = logging.getLogger(__name__)


class ConceptMatcher:
    """Handles semantic matching of descriptions to EDAM concepts."""

    def __init__(self, ontology_loader: OntologyLoader):
        """Initialize the concept matcher.

        Args:
            ontology_loader: Loaded ontology instance.
        """
        self.ontology_loader = ontology_loader
        self.embedding_model = None
        self.concept_embeddings: dict[str, np.ndarray] = {}

    def match_concepts(
        self,
        description: str,
        context: str | None = None,
        max_results: int = 5,
        min_confidence: float = 0.5,
    ) -> list[ConceptMatch]:
        """Match a description to EDAM concepts.

        Args:
            description: Text description to match.
            context: Additional context information.
            max_results: Maximum number of matches to return.
            min_confidence: Minimum confidence threshold.

        Returns:
            List of concept matches ordered by confidence.
        """
        # Build embeddings if not already built
        if not self.concept_embeddings:
            self._build_embeddings()

        # Preprocess input text
        processed_description = preprocess_text(description)

        # Add context if provided
        if context:
            processed_description += " " + preprocess_text(context)

        # Generate embedding for the description
        description_embedding = self.embedding_model.encode(processed_description, show_progress_bar=False)

        # Calculate similarities
        similarities = self._calculate_similarities(description_embedding)

        # Filter and sort results
        matches = []
        for uri, similarity in similarities:
            if similarity >= min_confidence:
                concept = self.ontology_loader.get_concept(uri)
                if concept:
                    match = ConceptMatch(
                        concept_uri=uri,
                        concept_label=concept["label"],
                        confidence=float(similarity),
                        concept_type=concept["type"],
                        definition=concept["definition"],
                        synonyms=concept["synonyms"],
                    )
                    matches.append(match)
            else:
                break

        return matches[:max_results]

    def get_concepts_from_biotools(
        self,
        tool_name: str,
        tool_curie: str | None = None,
        ontology_type: str = "operation",
    ) -> list[ConceptMatch]:
        """Get EDAM ontology concepts from bio.tools.

        Args:
            tool_name (str): The name of the tool.
            tool_curie (str|None): The biotoolsCURIE of the tool.
            ontology_type (str): What ontology terms to retrieve. Can be one of [operation, input, output, topic]

        Returns:
            List of concept matches.
        """
        tool_info = self._get_biotools_ontology(tool_name, tool_curie)
        try:
            ontology_terms = getattr(tool_info, ontology_type)
        except AttributeError:
            # Info not found
            logger.error(f"Bio.tools information for tool {tool_name} not found.")
            return []

        matches = []
        if ontology_type in ["operation", "topic"]:
            for term in ontology_terms:
                concept = self.ontology_loader.get_concept(term.get("uri"))
                match = ConceptMatch(
                    concept_uri=term.get("uri"),
                    concept_label=concept["label"],
                    confidence=None,
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                )
                matches.append(match)
        elif ontology_type in ["input", "output"]:
            for io in ontology_terms:
                data_term = io.get("data")
                format_terms = io.get("format")
                concept = self.ontology_loader.get_concept(data_term.get("uri"))
                match = ConceptMatch(
                    concept_uri=data_term.get("uri"),
                    concept_label=concept["label"],
                    confidence=None,
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                )
                matches.append(match)
                for term in format_terms:
                    concept = self.ontology_loader.get_concept(term.get("uri"))
                    match = ConceptMatch(
                        concept_uri=term.get("uri"),
                        concept_label=concept["label"],
                        confidence=None,
                        concept_type=concept["type"],
                        definition=concept["definition"],
                        synonyms=concept["synonyms"],
                    )
                    matches.append(match)

        return matches

    def _build_embeddings(self) -> None:
        """Build embeddings for all concepts in the ontology."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.embedding_model)

        logger.info("Building concept embeddings...")

        for uri, concept in self.ontology_loader.concepts.items():
            # Create text representation for embedding
            text_parts = [concept["label"]]

            if concept["definition"]:
                text_parts.append(concept["definition"])

            if concept["synonyms"]:
                text_parts.extend(concept["synonyms"])

            text = " ".join(text_parts)
            processed_text = preprocess_text(text)

            # Generate embedding
            embedding = self.embedding_model.encode(processed_text, show_progress_bar=False)
            self.concept_embeddings[uri] = embedding

        logger.info(f"Built embeddings for {len(self.concept_embeddings)} concepts")

    def _calculate_similarities(self, description_embedding: np.ndarray) -> list[tuple[str, float]]:
        """Calculate cosine similarities between description and all concepts.

        Args:
            description_embedding: Embedding of the description.

        Returns:
            List of (concept_uri, similarity) tuples.
        """
        similarities = []

        for uri, concept_embedding in self.concept_embeddings.items():
            # Calculate cosine similarity
            similarity = self._cosine_similarity(description_embedding, concept_embedding)
            similarities.append((uri, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        except ZeroDivisionError:
            return 0.0

    def find_exact_matches(self, description: str) -> list[ConceptMatch]:
        """Find exact text matches in concept labels and synonyms.

        Args:
            description: Description to find exact matches for.

        Returns:
            List of exact matches.
        """
        description_lower = description.lower()
        matches = []

        for uri, concept in self.ontology_loader.concepts.items():
            # Check label
            if description_lower == concept["label"].lower():
                match = ConceptMatch(
                    concept_uri=uri,
                    concept_label=concept["label"],
                    confidence=1.0,
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                )
                matches.append(match)
                continue

            # Check synonyms
            for synonym in concept["synonyms"]:
                if description_lower == synonym.lower():
                    match = ConceptMatch(
                        concept_uri=uri,
                        concept_label=concept["label"],
                        confidence=1.0,
                        concept_type=concept["type"],
                        definition=concept["definition"],
                        synonyms=concept["synonyms"],
                    )
                    matches.append(match)
                    break

        return matches

    def get_concept_neighbors(self, concept_uri: str, max_distance: int = 2) -> list[ConceptMatch]:
        """Get neighboring concepts in the ontology hierarchy.

        Args:
            concept_uri: URI of the target concept.
            max_distance: Maximum distance to search.

        Returns:
            List of neighboring concepts.
        """
        neighbors = []
        visited = set()
        queue = [(concept_uri, 0)]

        while queue:
            current_uri, distance = queue.pop(0)

            if current_uri in visited or distance > max_distance:
                continue

            visited.add(current_uri)
            concept = self.ontology_loader.get_concept(current_uri)

            if concept and distance > 0:  # Exclude the original concept
                match = ConceptMatch(
                    concept_uri=current_uri,
                    concept_label=concept["label"],
                    confidence=1.0 - (distance * 0.2),  # Decrease confidence with distance
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                )
                neighbors.append(match)

            # Add parent and child concepts
            if distance < max_distance:
                for parent_uri in concept["parents"]:
                    queue.append((parent_uri, distance + 1))

                for child_uri in concept["children"]:
                    queue.append((child_uri, distance + 1))

        return neighbors

    def _get_biotools_ontology(self, tool_name: str, biotools_curie: str | None) -> BioToolsInfo | None:
        """
        Given a specific entry of the tools list associated to the module, return the biotools input ontology ID.
        Get the associated ontology terms from a tool in bio.tools.

        Args:
            tool_name (str): The name of the tool to get the bio.tools information for.
            biotools_curie (str|None): The biotools CURIE to get the bio.tools information for.

        Returns:
            BioToolsInfo: The information extracted from bio.tools for the tool.
        """

        url = f"https://bio.tools/api/t/?q={tool_name}&format=json"
        try:
            response = requests.get(url)
            response.raise_for_status()

            data = response.json()
            data_list = data.get("list", [])

        except requests.exceptions.RequestException:
            logger.error(f"Could not find the bio.tools entry for the tool {tool_name}.")
            return None

        selected_tool = []
        try:
            if biotools_curie:
                selected_tool = [
                    tool for tool in data_list if tool.get("biotoolsCURIE", None).lower() == biotools_curie.lower()
                ][0]
            else:
                selected_tool = [tool for tool in data_list if tool.get("name", None).lower() == tool_name.lower()][0]
        except IndexError:
            pass

        if not selected_tool:
            logger.error(f"The tool '{tool_name}' with biotools CUIRE '{biotools_curie}' was not found.")
            return None

        tool_functions = selected_tool.get("function", [])

        operations = []
        inputs = []
        outputs = []
        for function in tool_functions:
            operations += function.get("operation", "")
            inputs += function.get("input", "")
            outputs += function.get("output", "")

        tool_info = BioToolsInfo(
            name=selected_tool.get("name", ""),
            biotools_curie=selected_tool.get("biotoolsCURIE", ""),
            description=selected_tool.get("description", ""),
            operation=operations,
            input=inputs,
            output=outputs,
            topic=selected_tool.get("topic", []),
        )

        return tool_info
