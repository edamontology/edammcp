"""Concept matching functionality for mapping descriptions to EDAM concepts."""

import logging
from typing import Dict, List, Optional, Tuple

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
        self.concept_embeddings: Dict[str, np.ndarray] = {}
        self._model_available = False
        # Don't build embeddings immediately - do it lazily when needed
    
    def _build_embeddings(self) -> None:
        """Build embeddings for all concepts in the ontology."""
        # Lazy import of sentence_transformers with better error handling
        try:
            from sentence_transformers import SentenceTransformer
            import warnings
            # Suppress sentence-transformers warnings temporarily
            st_logger = logging.getLogger('sentence_transformers')
            original_level = st_logger.level
            st_logger.setLevel(logging.ERROR)
        except ImportError:
            logger.error("sentence_transformers not available. Install with: pip install sentence-transformers")
            return
        
        if self.embedding_model is None:
            try:
                # Try loading the model with the correct name format
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.embedding_model = SentenceTransformer(settings.embedding_model)
                self._model_available = True
                logger.info(f"Successfully loaded embedding model: {settings.embedding_model}")
            except Exception as e:
                logger.error(f"Failed to load embedding model '{settings.embedding_model}': {e}")
                # Try alternative model names as fallbacks
                fallback_models = [
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "paraphrase-MiniLM-L6-v2",
                    "all-MiniLM-L12-v2"
                ]
                
                for fallback_model in fallback_models:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            self.embedding_model = SentenceTransformer(fallback_model)
                        self._model_available = True
                        logger.info(f"Successfully loaded fallback model: {fallback_model}")
                        break
                    except Exception:
                        # Silently continue to next fallback without logging warnings
                        continue
                
                if not self._model_available:
                    logger.error("No embedding model could be loaded. Semantic matching will not be available.")
                    return
            finally:
                # Restore original logging level
                st_logger.setLevel(original_level)
        
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
            try:
                embedding = self.embedding_model.encode(processed_text)
                self.concept_embeddings[uri] = embedding
            except Exception as e:
                logger.warning(f"Failed to generate embedding for concept {uri}: {e}")
                continue
        
        logger.info(f"Built embeddings for {len(self.concept_embeddings)} concepts")
    
    def match_concepts(
        self,
        description: str,
        context: Optional[str] = None,
        max_results: int = 5,
        min_confidence: float = 0.5
    ) -> List[ConceptMatch]:
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
        
        # If model is not available, return empty list
        if not self._model_available or not self.concept_embeddings:
            logger.warning("Embedding model not available, cannot perform semantic matching")
            return []
        
        # Preprocess input text
        processed_description = preprocess_text(description)
        
        # Add context if provided
        if context:
            processed_description += " " + preprocess_text(context)
        
        # Generate embedding for the description
        try:
            description_embedding = self.embedding_model.encode(processed_description)
        except Exception as e:
            logger.error(f"Failed to encode description: {e}")
            return []
        
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
                        synonyms=concept["synonyms"]
                    )
                    matches.append(match)
        
        # Sort by confidence and limit results
        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches[:max_results]
    
    def _calculate_similarities(
        self, 
        description_embedding: np.ndarray
    ) -> List[Tuple[str, float]]:
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
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_exact_matches(self, description: str) -> List[ConceptMatch]:
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
                    synonyms=concept["synonyms"]
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
                        synonyms=concept["synonyms"]
                    )
                    matches.append(match)
                    break
        
        return matches
    
    def get_concept_neighbors(
        self, 
        concept_uri: str, 
        max_distance: int = 2
    ) -> List[ConceptMatch]:
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
                    synonyms=concept["synonyms"]
                )
                neighbors.append(match)
            
            # Add parent and child concepts
            if distance < max_distance:
                for parent_uri in concept["parents"]:
                    queue.append((parent_uri, distance + 1))
                
                for child_uri in concept["children"]:
                    queue.append((child_uri, distance + 1))
        
        return neighbors