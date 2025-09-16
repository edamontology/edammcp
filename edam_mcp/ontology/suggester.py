"""Concept suggestion functionality for proposing new EDAM concepts."""

import logging
import re
from typing import Dict, List, Optional

from ..models.responses import SuggestedConcept
from ..utils.text_processing import preprocess_text
from .loader import OntologyLoader
from .matcher import ConceptMatcher

logger = logging.getLogger(__name__)


class ConceptSuggester:
    """Handles generation of new EDAM concept suggestions."""
    
    def __init__(self, ontology_loader: OntologyLoader, concept_matcher: ConceptMatcher):
        """Initialize the concept suggester.
        
        Args:
            ontology_loader: Loaded ontology instance.
            concept_matcher: Concept matcher instance.
        """
        self.ontology_loader = ontology_loader
        self.concept_matcher = concept_matcher
    
    def suggest_concepts(
        self,
        description: str,
        concept_type: Optional[str] = None,
        parent_concept: Optional[str] = None,
        rationale: Optional[str] = None,
        max_suggestions: int = 5
    ) -> List[SuggestedConcept]:
        """Generate suggestions for new EDAM concepts.
        
        Args:
            description: Description of the concept to suggest.
            concept_type: Type of concept (Operation, Data, Format, Topic, Identifier).
            parent_concept: Suggested parent concept.
            rationale: Rationale for the suggestion.
            max_suggestions: Maximum number of suggestions to generate.
            
        Returns:
            List of suggested concepts.
        """
        suggestions = []
        
        # Determine concept type if not provided
        if not concept_type:
            concept_type = self._infer_concept_type(description)
        
        # Generate multiple suggestions with different approaches
        suggestions.extend(
            self._generate_label_based_suggestions(description, concept_type, max_suggestions)
        )
        
        suggestions.extend(
            self._generate_hierarchical_suggestions(description, concept_type, parent_concept, max_suggestions)
        )
        
        # Remove duplicates and sort by confidence
        unique_suggestions = self._deduplicate_suggestions(suggestions)
        unique_suggestions.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_suggestions[:max_suggestions]
    
    def _infer_concept_type(self, description: str) -> str:
        """Infer the concept type from the description.
        
        Args:
            description: Description text.
            
        Returns:
            Inferred concept type.
        """
        description_lower = description.lower()
        
        # Keywords for different concept types
        operation_keywords = [
            "analyze", "process", "filter", "transform", "convert", "calculate",
            "compute", "generate", "create", "extract", "merge", "split",
            "align", "assemble", "annotate", "predict", "classify", "cluster"
        ]
        
        data_keywords = [
            "sequence", "alignment", "matrix", "table", "list", "tree",
            "graph", "network", "profile", "signature", "pattern", "motif",
            "dataset", "collection", "set", "file", "record"
        ]
        
        format_keywords = [
            "format", "file", "extension", "encoding", "structure",
            "fasta", "fastq", "sam", "bam", "vcf", "bed", "gff", "gtf",
            "csv", "tsv", "json", "xml", "yaml"
        ]
        
        topic_keywords = [
            "biology", "genomics", "proteomics", "metabolomics", "transcriptomics",
            "phylogenetics", "evolution", "disease", "cancer", "drug", "protein",
            "gene", "dna", "rna", "sequence", "alignment", "annotation"
        ]
        
        # Count keyword matches
        operation_score = sum(1 for keyword in operation_keywords if keyword in description_lower)
        data_score = sum(1 for keyword in data_keywords if keyword in description_lower)
        format_score = sum(1 for keyword in format_keywords if keyword in description_lower)
        topic_score = sum(1 for keyword in topic_keywords if keyword in description_lower)
        
        # Return the type with highest score
        scores = {
            "Operation": operation_score,
            "Data": data_score,
            "Format": format_score,
            "Topic": topic_score
        }
        
        return max(scores, key=scores.get)
    
    def _generate_label_based_suggestions(
        self,
        description: str,
        concept_type: str,
        max_suggestions: int
    ) -> List[SuggestedConcept]:
        """Generate suggestions based on the description text.
        
        Args:
            description: Description text.
            concept_type: Type of concept.
            max_suggestions: Maximum number of suggestions.
            
        Returns:
            List of suggested concepts.
        """
        suggestions = []
        
        # Clean and process the description
        processed_description = preprocess_text(description)
        
        # Generate different label variations
        label_variations = self._generate_label_variations(processed_description)
        
        for i, label in enumerate(label_variations[:max_suggestions]):
            # Generate URI
            uri = self._generate_uri(label, concept_type)
            
            # Generate definition
            definition = self._generate_definition(description, label, concept_type)
            
            # Calculate confidence based on label quality
            confidence = self._calculate_label_confidence(label, description)
            
            suggestion = SuggestedConcept(
                suggested_label=label,
                suggested_uri=uri,
                concept_type=concept_type,
                definition=definition,
                parent_concept=None,  # Will be set by hierarchical suggestions
                rationale=f"Generated from description: '{description}'",
                confidence=confidence
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_hierarchical_suggestions(
        self,
        description: str,
        concept_type: str,
        parent_concept: Optional[str],
        max_suggestions: int
    ) -> List[SuggestedConcept]:
        """Generate suggestions with hierarchical placement.
        
        Args:
            description: Description text.
            concept_type: Type of concept.
            parent_concept: Suggested parent concept.
            max_suggestions: Maximum number of suggestions.
            
        Returns:
            List of suggested concepts with hierarchical placement.
        """
        suggestions = []
        
        # Find potential parent concepts
        potential_parents = self._find_potential_parents(description, concept_type)
        
        if parent_concept:
            potential_parents.insert(0, parent_concept)
        
        for parent in potential_parents[:max_suggestions]:
            # Generate label based on parent context
            label = self._generate_contextual_label(description, parent)
            uri = self._generate_uri(label, concept_type)
            definition = self._generate_definition(description, label, concept_type)
            
            # Calculate confidence based on parent relationship
            confidence = self._calculate_hierarchical_confidence(description, parent)
            
            suggestion = SuggestedConcept(
                suggested_label=label,
                suggested_uri=uri,
                concept_type=concept_type,
                definition=definition,
                parent_concept=parent,
                rationale=f"Suggested as child of '{parent}' based on description",
                confidence=confidence
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_label_variations(self, text: str) -> List[str]:
        """Generate different label variations from text.
        
        Args:
            text: Input text.
            
        Returns:
            List of label variations.
        """
        variations = []
        
        # Convert to title case
        title_case = text.title()
        variations.append(title_case)
        
        # Extract key terms
        words = text.split()
        if len(words) > 1:
            # Use first few words
            key_terms = " ".join(words[:3]).title()
            variations.append(key_terms)
        
        # Remove common words and create label
        common_words = {"the", "a", "an", "and", "or", "for", "with", "in", "on", "at", "to", "of"}
        filtered_words = [word for word in words if word.lower() not in common_words]
        if filtered_words:
            filtered_label = " ".join(filtered_words[:4]).title()
            variations.append(filtered_label)
        
        return list(set(variations))  # Remove duplicates
    
    def _generate_uri(self, label: str, concept_type: str) -> str:
        """Generate a URI for a concept.
        
        Args:
            label: Concept label.
            concept_type: Type of concept.
            
        Returns:
            Generated URI.
        """
        # Convert label to URI format
        uri_part = re.sub(r'[^a-zA-Z0-9\s]', '', label)
        uri_part = re.sub(r'\s+', '_', uri_part).lower()
        
        # Add concept type prefix
        type_prefix = concept_type.lower()
        
        return f"http://edamontology.org/{type_prefix}_{uri_part}"
    
    def _generate_definition(self, description: str, label: str, concept_type: str) -> str:
        """Generate a definition for a concept.
        
        Args:
            description: Original description.
            label: Concept label.
            concept_type: Type of concept.
            
        Returns:
            Generated definition.
        """
        # Use the original description as base
        definition = description.strip()
        
        # Ensure it starts with a capital letter and ends with a period
        if definition:
            definition = definition[0].upper() + definition[1:]
            if not definition.endswith('.'):
                definition += '.'
        
        return definition
    
    def _find_potential_parents(self, description: str, concept_type: str) -> List[str]:
        """Find potential parent concepts for the description.
        
        Args:
            description: Description text.
            concept_type: Type of concept.
            
        Returns:
            List of potential parent concept URIs.
        """
        # Get concepts of the same type
        same_type_concepts = self.ontology_loader.get_concepts_by_type(concept_type)
        
        # Find similar concepts
        matches = self.concept_matcher.match_concepts(
            description,
            max_results=10,
            min_confidence=0.3
        )
        
        # Extract parent concepts from matches
        potential_parents = []
        for match in matches:
            concept = self.ontology_loader.get_concept(match.concept_uri)
            if concept and concept["parents"]:
                potential_parents.extend(concept["parents"])
        
        return list(set(potential_parents))  # Remove duplicates
    
    def _generate_contextual_label(self, description: str, parent_uri: str) -> str:
        """Generate a label considering the parent context.
        
        Args:
            description: Description text.
            parent_uri: Parent concept URI.
            
        Returns:
            Contextual label.
        """
        parent_concept = self.ontology_loader.get_concept(parent_uri)
        if parent_concept:
            # Use parent label as prefix
            parent_label = parent_concept["label"]
            # Extract key terms from description
            words = description.split()[:2]
            key_terms = " ".join(words).title()
            return f"{key_terms} {parent_label}"
        
        return description.title()
    
    def _calculate_label_confidence(self, label: str, description: str) -> float:
        """Calculate confidence based on label quality.
        
        Args:
            label: Generated label.
            description: Original description.
            
        Returns:
            Confidence score.
        """
        # Base confidence
        confidence = 0.5
        
        # Length factor
        if 3 <= len(label.split()) <= 6:
            confidence += 0.2
        
        # Specificity factor
        if len(label) > 10:
            confidence += 0.1
        
        # Clarity factor (no special characters)
        if re.match(r'^[a-zA-Z0-9\s]+$', label):
            confidence += 0.1
        
        # Relevance factor (label contains words from description)
        description_words = set(description.lower().split())
        label_words = set(label.lower().split())
        overlap = len(description_words.intersection(label_words))
        if overlap > 0:
            confidence += min(0.1 * overlap, 0.2)
        
        return min(confidence, 1.0)
    
    def _calculate_hierarchical_confidence(self, description: str, parent_uri: str) -> float:
        """Calculate confidence based on hierarchical relationship.
        
        Args:
            description: Description text.
            parent_uri: Parent concept URI.
            
        Returns:
            Confidence score.
        """
        # Base confidence
        confidence = 0.6
        
        # Check if parent concept is relevant
        parent_concept = self.ontology_loader.get_concept(parent_uri)
        if parent_concept:
            # Check if parent label appears in description
            if parent_concept["label"].lower() in description.lower():
                confidence += 0.2
            
            # Check if parent definition is relevant
            if parent_concept["definition"]:
                parent_words = set(parent_concept["definition"].lower().split())
                desc_words = set(description.lower().split())
                overlap = len(parent_words.intersection(desc_words))
                if overlap > 0:
                    confidence += min(0.1 * overlap, 0.2)
        
        return min(confidence, 1.0)
    
    def _deduplicate_suggestions(self, suggestions: List[SuggestedConcept]) -> List[SuggestedConcept]:
        """Remove duplicate suggestions.
        
        Args:
            suggestions: List of suggestions.
            
        Returns:
            Deduplicated list.
        """
        seen_labels = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            if suggestion.suggested_label not in seen_labels:
                seen_labels.add(suggestion.suggested_label)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions 