"""MCP tool for segmenting input text"""

from collections import Counter
from heapq import nlargest
from string import punctuation

import spacy
from fastmcp.server import Context
from spacy import cli
from spacy.lang.en.stop_words import STOP_WORDS

from ..models.segmentation import SegmentationRequest, SegmentationResponse


def load_spacy_model(model_name: str = "en_core_web_sm") -> spacy.language.Language:
    """
    Load a spaCy model, attempting to download it if not found.

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        Loaded spaCy language model

    Raises:
        OSError: If the model cannot be loaded even after attempting to download
    """
    try:
        return spacy.load(model_name)
    except OSError:
        # Try to download the model once
        try:
            cli.download(model_name)
            # Try loading again after download
            return spacy.load(model_name)
        except Exception as download_error:
            raise OSError(
                f"spaCy model '{model_name}' not found and could not be downloaded. Download error: {download_error}"
            ) from download_error


def is_not_all_stopwords(phrase: str, nlp: spacy.language.Language) -> bool:
    """
    Returns True if the phrase contains at least one token that is not a stop word or punctuation.
    """
    doc = nlp(phrase)
    return any(not token.is_stop and not token.is_punct for token in doc)


def extract_concepts(text: str) -> SegmentationResponse:
    #
    # use NLP
    #
    """
    Extracts non-redundant noun chunks from input text, excluding those comprised only of stop words or punctuation.
    Returns a list of concept-phrases sorted (longest first, then lexically).
    """
    nlp = load_spacy_model()
    noun_chunks = set(chunk.text.strip() for chunk in nlp(text).noun_chunks)
    filtered = [phrase for phrase in noun_chunks if is_not_all_stopwords(phrase, nlp)]
    concepts = sorted(filtered, key=lambda x: (-len(x), x.lower()))
    return concepts


async def segment_text(request: SegmentationRequest, context: Context) -> SegmentationResponse:
    """Convert free text to a JSON document with a list of concepts enumerated using spacy nlp

    This tool takes a free text string and produces a 'segmentation'
    based on key concepts, eliminating stopwords.

    Args:
        request: Segmentation request containing text to segment
        context: MCP context for logging and progress reporting.

    Returns:
        Segmentation response with topic and keywords
    """
    try:
        # Log the request
        context.info(f"Segmenting text: {request.text[:100]}...")

        concepts = extract_concepts(request.text)
        topic = spacy_summary_phrase(request.text)

        context.info(f"Found {len(concepts)} conceptual segments")

        print(concepts)
        print(topic)
        return SegmentationResponse(topic=topic, keywords=concepts)

    except Exception as e:
        context.error(f"Error in segment_text: {e}")
        raise


# by perplexity
def spacy_text_summary(text: str, num_sentences: int = 2) -> str:
    nlp = load_spacy_model()
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
    nlp = load_spacy_model()
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
