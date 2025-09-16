"""Text processing utilities for ontology matching."""

import re
import string
from typing import List


def preprocess_text(text: str) -> str:
    """Preprocess text for ontology matching.
    
    Args:
        text: Input text to preprocess.
        
    Returns:
        Preprocessed text.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation except hyphens and underscores
    text = re.sub(r'[^\w\s\-_]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def normalize_text(text: str) -> str:
    """Normalize text for comparison.
    
    Args:
        text: Input text to normalize.
        
    Returns:
        Normalized text.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract key terms from text.
    
    Args:
        text: Input text.
        max_keywords: Maximum number of keywords to extract.
        
    Returns:
        List of extracted keywords.
    """
    # Common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
    }
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Split into words
    words = processed_text.split()
    
    # Filter out stop words and short words
    keywords = [
        word for word in words
        if word not in stop_words and len(word) > 2
    ]
    
    # Count word frequencies
    word_counts = {}
    for word in keywords:
        word_counts[word] = word_counts.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, count in sorted_keywords[:max_keywords]]


def tokenize_text(text: str) -> List[str]:
    """Tokenize text into words.
    
    Args:
        text: Input text.
        
    Returns:
        List of tokens.
    """
    if not text:
        return []
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Split into tokens
    tokens = processed_text.split()
    
    return tokens


def calculate_text_length(text: str) -> int:
    """Calculate the length of text in characters.
    
    Args:
        text: Input text.
        
    Returns:
        Length of text.
    """
    return len(text) if text else 0


def calculate_word_count(text: str) -> int:
    """Calculate the number of words in text.
    
    Args:
        text: Input text.
        
    Returns:
        Number of words.
    """
    if not text:
        return 0
    
    tokens = tokenize_text(text)
    return len(tokens) 