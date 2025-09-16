"""Similarity calculation utilities for ontology matching."""

from typing import List, Set
import numpy as np


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two text strings.
    
    Args:
        text1: First text string.
        text2: Second text string.
        
    Returns:
        Similarity score between 0 and 1.
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_jaccard_similarity(set1: Set, set2: Set) -> float:
    """Calculate Jaccard similarity between two sets.
    
    Args:
        set1: First set.
        set2: Second set.
        
    Returns:
        Jaccard similarity score between 0 and 1.
    """
    if not set1 and not set2:
        return 1.0  # Both empty sets are considered identical
    
    if not set1 or not set2:
        return 0.0  # One empty set means no similarity
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector.
        vec2: Second vector.
        
    Returns:
        Cosine similarity score between -1 and 1.
    """
    if vec1.size == 0 or vec2.size == 0:
        return 0.0
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Avoid division by zero
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def calculate_levenshtein_distance(str1: str, str2: str) -> int:
    """Calculate Levenshtein distance between two strings.
    
    Args:
        str1: First string.
        str2: Second string.
        
    Returns:
        Levenshtein distance.
    """
    if not str1:
        return len(str2)
    
    if not str2:
        return len(str1)
    
    # Create matrix
    matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
    
    # Initialize first row and column
    for i in range(len(str1) + 1):
        matrix[i][0] = i
    
    for j in range(len(str2) + 1):
        matrix[0][j] = j
    
    # Fill matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = min(
                    matrix[i - 1][j] + 1,      # deletion
                    matrix[i][j - 1] + 1,      # insertion
                    matrix[i - 1][j - 1] + 1   # substitution
                )
    
    return matrix[len(str1)][len(str2)]


def calculate_string_similarity(str1: str, str2: str) -> float:
    """Calculate similarity between two strings using Levenshtein distance.
    
    Args:
        str1: First string.
        str2: Second string.
        
    Returns:
        Similarity score between 0 and 1.
    """
    if not str1 and not str2:
        return 1.0
    
    if not str1 or not str2:
        return 0.0
    
    distance = calculate_levenshtein_distance(str1, str2)
    max_length = max(len(str1), len(str2))
    
    return 1.0 - (distance / max_length)


def calculate_overlap_similarity(list1: List, list2: List) -> float:
    """Calculate overlap similarity between two lists.
    
    Args:
        list1: First list.
        list2: Second list.
        
    Returns:
        Overlap similarity score between 0 and 1.
    """
    if not list1 and not list2:
        return 1.0
    
    if not list1 or not list2:
        return 0.0
    
    set1 = set(list1)
    set2 = set(list2)
    
    intersection = len(set1.intersection(set2))
    min_length = min(len(set1), len(set2))
    
    if min_length == 0:
        return 0.0
    
    return intersection / min_length


def calculate_weighted_similarity(
    similarities: List[float],
    weights: List[float]
) -> float:
    """Calculate weighted average of multiple similarity scores.
    
    Args:
        similarities: List of similarity scores.
        weights: List of weights for each similarity score.
        
    Returns:
        Weighted average similarity score.
    """
    if not similarities or not weights:
        return 0.0
    
    if len(similarities) != len(weights):
        raise ValueError("Number of similarities must match number of weights")
    
    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        return 0.0
    
    normalized_weights = [w / total_weight for w in weights]
    
    # Calculate weighted average
    weighted_sum = sum(s * w for s, w in zip(similarities, normalized_weights))
    
    return weighted_sum 