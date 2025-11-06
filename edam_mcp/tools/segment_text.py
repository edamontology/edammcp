"""MCP tool for segmenting input text"""

from collections import Counter
from heapq import nlargest
from string import punctuation

import spacy
from fastmcp.server import Context
from spacy.lang.en.stop_words import STOP_WORDS

from ..models.segment import ReadyForMapping


def is_not_all_stopwords(phrase: str, nlp: spacy.language.Language) -> bool:
    """
    Returns True if the phrase contains at least one token that is not a stop word or punctuation.
    """
    doc = nlp(phrase)
    return any(not token.is_stop and not token.is_punct for token in doc)


def extract_concepts(text: str) -> ReadyForMapping:
    #
    # use NLP
    #
    """
    Extracts non-redundant noun chunks from input text, excluding those comprised only of stop words or punctuation.
    Returns a list of concept-phrases sorted (longest first, then lexically).
    """
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as e:
        print("spaCy model 'en_core_web_sm' not found or not installed.")
        print("If pip is not available in your environment, you may need to add it first:")
        print("    uv add pip")
        raise e  # Optionally re-raise to halt execution
    noun_chunks = set(chunk.text.strip() for chunk in nlp(text).noun_chunks)
    filtered = [phrase for phrase in noun_chunks if is_not_all_stopwords(phrase, nlp)]
    concepts = sorted(filtered, key=lambda x: (-len(x), x.lower()))
    return concepts


async def segment_text(request: str, context: Context) -> ReadyForMapping:
    """Convert free text to a JSON document with a list of concepts enumerated using spacy nlp

    This tool takes a free text string and produces a 'segmentation'
    based on key concepts, eliminating stopwords.

    Args:
        request: free text string
        context: MCP context for logging and progress reporting.

    Returns:
        JSON document
    """
    try:
        # Log the request
        context.info(f"Mapping description: {request[:100]}...")

        concepts = extract_concepts(request)
        top_concept = spacy_summary_phrase(request)

        context.info(f"Found {len(concepts)} conceptual segments")

        print(concepts)
        print(top_concept)
        return ReadyForMapping(top_concept=top_concept, chunks=concepts)

    except Exception as e:
        context.error(f"Error in segment_text: {e}")
        raise


# by perplexity
def spacy_text_summary(text: str, num_sentences: int = 2) -> str:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    # Calculate word frequencies
    word_freq = {}
    for token in doc:
        if token.text.lower() not in STOP_WORDS and token.text not in punctuation:
            word = token.text.lower()
            word_freq[word] = word_freq.get(word, 0) + 1

    max_freq = max(word_freq.values(), default=1)
    for word in word_freq:
        word_freq[word] /= max_freq

    # Rank sentences by word frequency
    sent_strength = {}
    for sent in doc.sents:
        for word in sent:
            if word.text.lower() in word_freq:
                sent_strength[sent] = sent_strength.get(sent, 0) + word_freq[word.text.lower()]

    # Select top n sentences for summary
    selected = nlargest(num_sentences, sent_strength, key=sent_strength.get)
    summary = " ".join(sent.text for sent in selected)
    return summary


# print(spacy_text_summary(text, num_sentences=1))


def spacy_keywords(text: str, max_keywords: int = 3) -> list[str]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    words = [
        token.lemma_.lower()
        for token in doc
        if (
            token.pos_ in {"NOUN", "PROPN", "ADJ"}
            and token.lemma_.lower() not in STOP_WORDS
            and not token.is_punct
            and len(token.lemma_) > 2
        )
    ]
    counts = Counter(words)
    keywords = [kw for kw, _ in counts.most_common(max_keywords)]
    return keywords


def spacy_summary_phrase(text: str) -> str:
    keywords = spacy_keywords(text, max_keywords=3)
    # If fewer than 3 keywords, fill with blanks or join as is
    phrase = " ".join(keywords)
    return phrase


# Example usage:
# text = "spaCy is a popular library for efficient Natural Language Processing in Python."
# print(spacy_summary_phrase(text))  # Possible output: 'library spacy language'
