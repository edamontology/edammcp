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

        Args:
            keywords: List of keywords to match.
            threshold: Minimum similarity score threshold.

        Returns:
            List of Match objects sorted by score.
        """
        # To be implemented
        raise NotImplementedError("match_by_list_embeddings not yet implemented")

    def match_by_joined_embedding(
        self, 
        keywords: list[str], 
        threshold: float = 0.3
    ) -> list[Match]:
        """Match keywords using a single embedding from joined keywords.

        Args:
            keywords: List of keywords to match.
            threshold: Minimum similarity score threshold.

        Returns:
            List of Match objects sorted by score.
        """
        # To be implemented
        raise NotImplementedError("match_by_joined_embedding not yet implemented")
