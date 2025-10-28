"""Keyword matching functionality for EDAM ontology terms."""

import logging
from dataclasses import dataclass

import numpy as np

from ..config import settings
from ..utils.text_processing import preprocess_text
from .loader import OntologyLoader

logger = logging.getLogger(__name__)


@dataclass
class Match:
    """Represents a keyword match result."""

    term_id: str
    label: str
    score: float
    match_type: str
    matched_keywords: list[str]


class KeywordMatcher:
    """Handles keyword matching against EDAM ontology terms using multiple search modes."""

    def __init__(self, ontology_loader: OntologyLoader):
        """Initialize the keyword matcher.

        Args:
            ontology_loader: Loaded ontology instance.
        """
        self.ontology_loader = ontology_loader
        self.embedding_model = None
        self.term_descriptions: dict[str, str] = {}
        self.term_embeddings: dict[str, np.ndarray] = {}
        
        # Prepare term descriptions immediately
        self._prepare_term_descriptions()

    def _prepare_term_descriptions(self) -> None:
        """Index ontology term descriptions for matching.
        
        Creates a dictionary mapping term URIs to their text descriptions,
        which combines label, definition, and synonyms.
        """
        logger.info("Preparing term descriptions for keyword matching...")
        
        for uri, concept in self.ontology_loader.concepts.items():
            # Build description from label, definition, and synonyms
            text_parts = [concept["label"]]
            
            if concept["definition"]:
                text_parts.append(concept["definition"])
            
            if concept["synonyms"]:
                text_parts.extend(concept["synonyms"])
            
            # Join and preprocess the text
            description = " ".join(text_parts)
            self.term_descriptions[uri] = preprocess_text(description)
        
        logger.info("Prepared descriptions for %d terms", len(self.term_descriptions))

    def _prepare_embeddings(self) -> None:
        """Create embeddings for all term descriptions.
        
        Lazily loads the embedding model and generates embeddings for all
        term descriptions to enable semantic similarity matching.
        """
        # Lazy import of sentence_transformers
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error(
                "sentence_transformers not available. "
                "Install with: pip install sentence-transformers"
            )
            return

        # Initialize embedding model if not already done
        if self.embedding_model is None:
            logger.info("Loading embedding model: %s", settings.embedding_model)
            self.embedding_model = SentenceTransformer(settings.embedding_model)

        logger.info("Building term embeddings for semantic matching...")

        # Generate embeddings for all term descriptions
        for uri, description in self.term_descriptions.items():
            embedding = self.embedding_model.encode(
                description, 
                show_progress_bar=False
            )
            self.term_embeddings[uri] = embedding

        logger.info("Built embeddings for %d terms", len(self.term_embeddings))

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.
            
        Returns:
            Cosine similarity score between -1 and 1.
        """
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        
        # Compute dot product
        return float(np.dot(vec1_norm, vec2_norm))

    def match_by_substring(self, keywords: list[str]) -> list[Match]:
        """Match keywords using substring search in term descriptions.

        Args:
            keywords: List of keywords to match.

        Returns:
            List of Match objects sorted by score (match count).
        """
        logger.info("Matching keywords by substring: %s", keywords)
        
        # Handle empty keyword list
        if not keywords:
            logger.warning("Empty keyword list provided")
            return []
        
        # Preprocess keywords for case-insensitive matching
        processed_keywords = [preprocess_text(kw) for kw in keywords]
        
        # Dictionary to store term URI -> (match_count, matched_keywords)
        term_matches: dict[str, tuple[int, list[str]]] = {}
        
        # Search for each keyword in term descriptions
        for uri, description in self.term_descriptions.items():
            match_count = 0
            matched_keywords = []
            
            # Check if each keyword appears in the description
            for original_kw, processed_kw in zip(keywords, processed_keywords):
                if processed_kw.lower() in description.lower():
                    match_count += 1
                    matched_keywords.append(original_kw)
            
            # Store if any matches found
            if match_count > 0:
                term_matches[uri] = (match_count, matched_keywords)
        
        # Build Match objects
        matches = []
        for uri, (match_count, matched_keywords) in term_matches.items():
            concept = self.ontology_loader.concepts.get(uri)
            if concept:
                match = Match(
                    term_id=uri,
                    label=concept["label"],
                    score=float(match_count),
                    match_type="substring",
                    matched_keywords=matched_keywords
                )
                matches.append(match)
        
        # Sort by score (match count) in descending order
        matches.sort(key=lambda m: m.score, reverse=True)
        
        logger.info("Found %d matches using substring search", len(matches))
        return matches

    def match_by_list_embeddings(
        self, 
        keywords: list[str], 
        threshold: float = 0.3
    ) -> list[Match]:
        """Match keywords using separate embeddings for each keyword.
        
        Creates a separate embedding for each keyword and computes similarity
        with term embeddings. Uses max aggregation to find the best match
        for each term.

        Args:
            keywords: List of keywords to match.
            threshold: Minimum similarity score threshold (default: 0.3).

        Returns:
            List of Match objects sorted by score (descending).
        """
        logger.info("Matching keywords by list embeddings: %s (threshold: %.2f)", keywords, threshold)
        
        # Handle empty keyword list
        if not keywords:
            logger.warning("Empty keyword list provided")
            return []
        
        # Ensure embeddings are prepared
        if not self.term_embeddings:
            logger.info("Embeddings not yet prepared, initializing...")
            self._prepare_embeddings()
        
        # Check if embedding model is available
        if self.embedding_model is None:
            logger.error("Embedding model not available, cannot perform semantic matching")
            return []
        
        # Create separate embeddings for each keyword
        logger.debug("Creating embeddings for %d keywords", len(keywords))
        keyword_embeddings = self.embedding_model.encode(
            keywords,
            show_progress_bar=False
        )
        
        # Dictionary to store term URI -> (max_score, best_matched_keyword)
        term_matches: dict[str, tuple[float, str]] = {}
        
        # For each term, find the maximum similarity across all keyword embeddings
        for uri, term_embedding in self.term_embeddings.items():
            max_similarity = -1.0
            best_keyword = ""
            
            # Compare term embedding with each keyword embedding
            for keyword, keyword_embedding in zip(keywords, keyword_embeddings):
                # Compute cosine similarity
                similarity = self._cosine_similarity(term_embedding, keyword_embedding)
                
                # Track the maximum similarity
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_keyword = keyword
            
            # Store if similarity meets threshold
            if max_similarity >= threshold:
                term_matches[uri] = (max_similarity, best_keyword)
        
        # Build Match objects
        matches = []
        for uri, (score, matched_keyword) in term_matches.items():
            concept = self.ontology_loader.concepts.get(uri)
            if concept:
                match = Match(
                    term_id=uri,
                    label=concept["label"],
                    score=score,
                    match_type="list_embeddings",
                    matched_keywords=[matched_keyword]
                )
                matches.append(match)
        
        # Sort by score in descending order
        matches.sort(key=lambda m: m.score, reverse=True)
        
        logger.info("Found %d matches using list embeddings", len(matches))
        return matches

    def match_by_joined_embedding(
        self, 
        keywords: list[str], 
        threshold: float = 0.3
    ) -> list[Match]:
        """Match keywords using a single embedding from joined keywords.
        
        Joins all keywords into a single string and creates one embedding
        for the entire query. This approach captures the semantic meaning
        of keywords as a phrase or context.

        Args:
            keywords: List of keywords to match.
            threshold: Minimum similarity score threshold (default: 0.3).

        Returns:
            List of Match objects sorted by score (descending).
        """
        logger.info("Matching keywords by joined embedding: %s (threshold: %.2f)", keywords, threshold)
        
        # Handle empty keyword list
        if not keywords:
            logger.warning("Empty keyword list provided")
            return []
        
        # Ensure embeddings are prepared
        if not self.term_embeddings:
            logger.info("Embeddings not yet prepared, initializing...")
            self._prepare_embeddings()
        
        # Check if embedding model is available
        if self.embedding_model is None:
            logger.error("Embedding model not available, cannot perform semantic matching")
            return []
        
        # Join keywords into a single string
        joined_keywords = " ".join(keywords)
        logger.debug("Creating embedding for joined query: '%s'", joined_keywords)
        
        # Create single embedding for the joined keywords
        query_embedding = self.embedding_model.encode(
            joined_keywords,
            show_progress_bar=False
        )
        
        # Dictionary to store term URI -> similarity_score
        term_matches: dict[str, float] = {}
        
        # Compute cosine similarity with all term embeddings
        for uri, term_embedding in self.term_embeddings.items():
            similarity = self._cosine_similarity(term_embedding, query_embedding)
            
            # Store if similarity meets threshold
            if similarity >= threshold:
                term_matches[uri] = similarity
        
        # Build Match objects
        matches = []
        for uri, score in term_matches.items():
            concept = self.ontology_loader.concepts.get(uri)
            if concept:
                match = Match(
                    term_id=uri,
                    label=concept["label"],
                    score=score,
                    match_type="joined_embedding",
                    matched_keywords=keywords  # All keywords contributed to the match
                )
                matches.append(match)
        
        # Sort by score in descending order
        matches.sort(key=lambda m: m.score, reverse=True)
        
        logger.info("Found %d matches using joined embedding", len(matches))
        return matches
