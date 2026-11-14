"""Concept matching functionality for mapping descriptions to EDAM concepts."""

import logging
import os

import numpy as np

from ..config import settings
from ..models.mapping import ConceptMatch
from ..utils.text_processing import preprocess_text
from .loader import OntologyLoader

logger = logging.getLogger(__name__)


def _normalize(x: np.ndarray) -> np.ndarray:
    """Scale vectors (last axis) to unit length; zero vectors stay zero."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(norm == 0, 1, norm)


class ConceptMatcher:
    """Handles semantic matching of descriptions to EDAM concepts."""

    def __init__(self, ontology_loader: OntologyLoader):
        """Initialize the concept matcher.

        Args:
            ontology_loader: Loaded ontology instance.
        """
        self.ontology_loader = ontology_loader
        self.embedding_model = None
        # In-memory index: row i of _matrix is the unit-norm embedding of _uris[i]
        self._uris: list[str] = []
        self._matrix: np.ndarray | None = None
        self.use_chromadb = settings.use_chromadb
        self.chroma_db = os.path.join(settings.cache_dir, "default.db")
        self._ready = False
        self._collection = None
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
            except ImportError as e:
                raise RuntimeError("chromadb not available. Install with: pip install chromadb") from e
            client = chromadb.PersistentClient(path=self.chroma_db)
            # Further details at: https://docs.trychroma.com/docs/collections/configure#hnsw-index-configuration
            # Embeddings are always passed explicitly, so no embedding function is needed
            # Vectors from another model (or an unlabelled older DB) are invalid: start over
            if any(
                c.name == "concept_embeddings" and (c.metadata or {}).get("embedding_model") != settings.embedding_model
                for c in client.list_collections()
            ):
                logger.info("Embedding model changed, rebuilding ChromaDB collection")
                client.delete_collection("concept_embeddings")
            collection = self._collection = client.get_or_create_collection(
                name="concept_embeddings",
                embedding_function=None,
                metadata={"embedding_model": settings.embedding_model},
                configuration={"hnsw": {"space": "cosine", "ef_construction": 200}},
            )
            logger.info("Building concept embeddings and storing in ChromaDB...")
        else:
            logger.info("Building concept embeddings and storing in memory...")

        concepts = self.ontology_loader.concepts
        uris = list(concepts)

        # Create text representation for embedding
        texts = []
        for uri in uris:
            concept = concepts[uri]
            text_parts = [concept["label"]]

            if concept["definition"]:
                text_parts.append(concept["definition"])

            if concept["synonyms"]:
                text_parts.extend(concept["synonyms"])

            texts.append(preprocess_text(" ".join(text_parts)))

        if self.use_chromadb:
            # One lookup for all stored ids and texts: drop concepts removed from EDAM,
            # then only embed the ones that are missing or whose text changed
            stored = collection.get(include=["documents"])
            stored_texts = dict(zip(stored["ids"], stored["documents"]))
            removed = [uri for uri in stored_texts if uri not in concepts]
            if removed:
                collection.delete(ids=removed)
            changed = [i for i, uri in enumerate(uris) if stored_texts.get(uri) != texts[i]]
            uris = [uris[i] for i in changed]
            texts = [texts[i] for i in changed]

        # Encode all texts in one batched call instead of one call per concept
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False) if texts else []

        if self.use_chromadb:
            # One transaction per chunk instead of one per concept (slow on spinning disks)
            step = client.get_max_batch_size()
            for i in range(0, len(uris), step):
                collection.upsert(
                    ids=uris[i : i + step],
                    embeddings=embeddings[i : i + step],
                    documents=texts[i : i + step],
                    metadatas=[
                        {
                            "label": concepts[uri]["label"],
                            "definition": concepts[uri].get("definition"),
                            "synonyms": ", ".join(concepts[uri]["synonyms"]),
                        }
                        for uri in uris[i : i + step]
                    ],
                )
            logger.info(f"Stored embeddings for {len(uris)} new/changed concepts in ChromaDB")
        else:
            self._uris, self._matrix = uris, _normalize(np.asarray(embeddings, dtype=np.float32))
            logger.info(f"Built embeddings for {len(self._uris)} concepts")
        self._ready = True

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
        if not self._ready:
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
            # Use ChromaDB's default query for similarity search
            query_results = self._collection.query(
                query_embeddings=[description_embedding],
                n_results=max_results,
            )
            ids = query_results.get("ids", [[]])[0]
            distances = query_results.get("distances", [[]])[0]
            # Convert distances to similarity scores (assuming distances are normalized between 0 and 1)
            similarity_scores = [1.0 - d for d in distances]
            similarities = list(zip(ids, similarity_scores))
        elif self._uris:
            # Exact cosine search: one matrix-vector product over all concepts.
            # einsum, not `@`: BLAS threads contend with torch's thread pool right after encode() (~60 ms vs <1 ms)
            query = _normalize(np.asarray(description_embedding, dtype=np.float32))
            scores = np.einsum("ij,j->i", self._matrix, query)
            top = np.argpartition(-scores, min(max_results, len(scores)) - 1)[:max_results]
            similarities = [(self._uris[i], float(scores[i])) for i in top]

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities

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

    def get_related(self, concept_uri: str, descendant_depth: int = 0, ancestor_depth: int = 0) -> list[ConceptMatch]:
        """Get narrower and broader concepts of concept_uri, ranked exact, descendant, ancestor.

        Args:
            concept_uri: URI of the target concept.
            descendant_depth: Steps down to children.
            ancestor_depth: Steps up to parents.

        Returns:
            Related concepts, excluding concept_uri itself, with direction and distance set.
        """
        related = []
        for uri, (direction, distance) in self.ontology_loader.get_related(
            concept_uri, descendant_depth, ancestor_depth
        ).items():
            concept = self.ontology_loader.get_concept(uri)
            if uri == concept_uri or concept is None:
                continue
            related.append(
                ConceptMatch(
                    concept_uri=uri,
                    concept_label=concept["label"],
                    confidence=1.0 - (distance * 0.2),  # Decrease confidence with distance
                    concept_type=concept["type"],
                    definition=concept["definition"],
                    synonyms=concept["synonyms"],
                    direction=direction,
                    distance=distance,
                )
            )
        return related
