"""Concept matching functionality for mapping descriptions to EDAM concepts."""

import logging
import os

import numpy as np

from ..config import settings
from ..models.responses import ConceptMatch
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
        self.use_chromadb = settings.use_chromadb
        self.chroma_db = os.path.join(settings.cache_dir, "default.db")
        # Don't build embeddings immediately - do it lazily when needed

    def _build_embeddings(self) -> None:
        """Build embeddings for all concepts in the ontology and store them in ChromaDB or in-memory."""
        # Lazy import of sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence_transformers not available. Install with: pip install sentence-transformers")
            return

        if self.embedding_model is None:
            # If none is defined 'all-MiniLM-L6-v2' is set
            self.embedding_model = SentenceTransformer(settings.embedding_model)

        if self.use_chromadb:
            try:
                import chromadb
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            except ImportError:
                logger.error("chromadb not available. Install with: pip install chromadb")
                return
            client = chromadb.PersistentClient(path=self.chroma_db)
            # Further details at: https://docs.trychroma.com/docs/collections/configure#hnsw-index-configuration
            embedding_function = SentenceTransformerEmbeddingFunction(settings.embedding_model)
            collection = client.get_or_create_collection(
                name="concept_embeddings",
                embedding_function=embedding_function,
                configuration={"hnsw": {"space": "cosine", "ef_construction": 200}},
            )
            logger.info("Building concept embeddings and storing in ChromaDB...")
        else:
            logger.info("Building concept embeddings and storing in memory...")

        for uri, concept in self.ontology_loader.concepts.items():
            if self.use_chromadb:
                existing = collection.get(ids=[uri])
                # ChromaDB returns empty dict if not found
                if existing and existing.get("ids"):
                    logger.debug(f"Embedding for {uri} already exists in ChromaDB, skipping.")
                    continue

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

            if self.use_chromadb:
                collection.add(
                    ids=[uri],
                    embeddings=[embedding.tolist()],
                    documents=[processed_text],
                    metadatas=[
                        {
                            "label": concept["label"],
                            "definition": concept.get("definition"),
                            "synonyms": (
                                ", ".join(concept["synonyms"])
                                if isinstance(concept.get("synonyms"), list)
                                else concept.get("synonyms")
                            ),
                        }
                    ],
                )
            else:
                self.concept_embeddings[uri] = embedding

        if self.use_chromadb:
            logger.info(f"Stored embeddings for {len(self.ontology_loader.concepts)} concepts in ChromaDB")
        else:
            logger.info(f"Built embeddings for {len(self.concept_embeddings)} concepts")

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
        similarities = self._calculate_similarities(description_embedding, max_results * 2)

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

        # Sort by confidence and limit results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]

    def _calculate_similarities(self, description_embedding: np.ndarray, max_results: int) -> list[tuple[str, float]]:
        """Calculate cosine similarities between description and all concepts.

        Args:
            description_embedding: Embedding of the description.

        Returns:
            List of (concept_uri, similarity) tuples.
        """
        similarities = []

        if self.use_chromadb:
            try:
                import chromadb
            except ImportError:
                logger.error("chromadb not available. Install with: pip install chromadb")
                return []

            client = chromadb.PersistentClient(path=self.chroma_db)
            collection = client.get_or_create_collection(name="concept_embeddings")
            # Use ChromaDB's default query for similarity search
            query_results = collection.query(
                query_embeddings=[description_embedding],
                n_results=max_results,
            )
            ids = query_results.get("ids", [[]])[0]
            distances = query_results.get("distances", [[]])[0]
            # Convert distances to similarity scores (assuming distances are normalized between 0 and 1)
            similarity_scores = [1.0 - d for d in distances]
            similarities = list(zip(ids, similarity_scores))
        else:
            for uri, concept_embedding in self.concept_embeddings.items():
                similarity = self._cosine_similarity(description_embedding, concept_embedding)
                similarities.append((uri, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

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
