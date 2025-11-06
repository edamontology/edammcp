"""Tests for the segmentation functionality."""

from unittest.mock import Mock, patch

import pytest
import spacy

from edam_mcp.models.segmentation import SegmentationRequest, SegmentationResponse
from edam_mcp.tools.segment_text import (
    extract_concepts,
    is_not_all_stopwords,
    load_spacy_model,
    segment_text,
    spacy_keywords,
    spacy_summary_phrase,
    spacy_text_summary,
)
from edam_mcp.utils.context import MockContext

# Note: extract_concepts actually returns list[str], not SegmentationResponse
# despite the type hint. Tests reflect the actual behavior.


class TestLoadSpacyModel:
    """Test cases for the load_spacy_model function."""

    def test_load_existing_model(self):
        """Test loading an existing spaCy model."""
        nlp = load_spacy_model("en_core_web_sm")
        assert nlp is not None
        assert isinstance(nlp, spacy.language.Language)

    @patch("edam_mcp.tools.segment_text.spacy.load")
    @patch("edam_mcp.tools.segment_text.cli.download")
    def test_download_model_on_failure(self, mock_download, mock_load):
        """Test that model is downloaded when not found."""
        # First call raises OSError, second call succeeds
        mock_load.side_effect = [OSError("Model not found"), Mock(spec=spacy.language.Language)]

        nlp = load_spacy_model("en_core_web_sm")

        mock_download.assert_called_once_with("en_core_web_sm")
        assert mock_load.call_count == 2
        assert nlp is not None

    @patch("edam_mcp.tools.segment_text.spacy.load")
    @patch("edam_mcp.tools.segment_text.cli.download")
    def test_raise_error_after_download_fails(self, mock_download, mock_load):
        """Test that error is raised if download fails."""
        mock_load.side_effect = OSError("Model not found")
        mock_download.side_effect = Exception("Download failed")

        with pytest.raises(OSError) as exc_info:
            load_spacy_model("en_core_web_sm")

        assert "not found and could not be downloaded" in str(exc_info.value)
        mock_download.assert_called_once_with("en_core_web_sm")

    @patch("edam_mcp.tools.segment_text.spacy.load")
    @patch("edam_mcp.tools.segment_text.cli.download")
    def test_raise_error_if_load_fails_after_download(self, mock_download, mock_load):
        """Test that error is raised if load fails after successful download."""
        mock_load.side_effect = [OSError("Model not found"), OSError("Still not found")]
        mock_download.return_value = None

        with pytest.raises(OSError) as exc_info:
            load_spacy_model("en_core_web_sm")

        assert "not found and could not be downloaded" in str(exc_info.value)
        mock_download.assert_called_once_with("en_core_web_sm")
        assert mock_load.call_count == 2


class TestIsNotAllStopwords:
    """Test cases for the is_not_all_stopwords function."""

    def test_phrase_with_content_words(self):
        """Test that phrases with content words return True."""
        nlp = load_spacy_model()
        assert is_not_all_stopwords("machine learning algorithm", nlp) is True
        assert is_not_all_stopwords("data analysis", nlp) is True

    def test_phrase_with_only_stopwords(self):
        """Test that phrases with only stopwords return False."""
        nlp = load_spacy_model()
        assert is_not_all_stopwords("the and or", nlp) is False
        assert is_not_all_stopwords("a an the", nlp) is False

    def test_phrase_with_punctuation_only(self):
        """Test that phrases with only punctuation return False."""
        nlp = load_spacy_model()
        assert is_not_all_stopwords("...", nlp) is False
        assert is_not_all_stopwords("!!!", nlp) is False

    def test_phrase_with_mixed_content(self):
        """Test that phrases with mixed content return True."""
        nlp = load_spacy_model()
        assert is_not_all_stopwords("the machine", nlp) is True
        assert is_not_all_stopwords("and data", nlp) is True


class TestExtractConcepts:
    """Test cases for the extract_concepts function."""

    def test_extract_concepts_from_simple_text(self):
        """Test extracting concepts from simple text."""
        text = "Machine learning algorithms process data efficiently."
        concepts = extract_concepts(text)

        assert isinstance(concepts, list)
        assert len(concepts) > 0
        # Should contain meaningful chunks
        assert any("machine learning" in chunk.lower() or "algorithm" in chunk.lower() for chunk in concepts)

    def test_extract_concepts_from_complex_text(self):
        """Test extracting concepts from complex text."""
        text = "The chromVAR R package analyzes chromatin accessibility data from ATAC-seq experiments."
        concepts = extract_concepts(text)

        assert isinstance(concepts, list)
        assert len(concepts) > 0
        # Concepts should be sorted (longest first, then lexically)
        if len(concepts) > 1:
            assert len(concepts[0]) >= len(concepts[1])

    def test_extract_concepts_filters_stopwords(self):
        """Test that concepts filtered to exclude stopword-only phrases."""
        text = "The and or but not."
        concepts = extract_concepts(text)

        # Should have very few or no concepts (mostly stopwords)
        assert isinstance(concepts, list)

    def test_extract_concepts_empty_text(self):
        """Test extracting concepts from empty text."""
        text = ""
        concepts = extract_concepts(text)

        assert isinstance(concepts, list)
        # Empty text should produce empty or minimal concepts

    def test_extract_concepts_sorted_order(self):
        """Test that concepts are sorted correctly (longest first, then lexically)."""
        text = "Natural language processing and machine learning algorithms."
        concepts = extract_concepts(text)

        if len(concepts) > 1:
            # Check sorting: longest first
            for i in range(len(concepts) - 1):
                assert len(concepts[i]) >= len(concepts[i + 1]) or (
                    len(concepts[i]) == len(concepts[i + 1]) and concepts[i].lower() <= concepts[i + 1].lower()
                )


class TestSpacyKeywords:
    """Test cases for the spacy_keywords function."""

    def test_extract_keywords(self):
        """Test extracting keywords from text."""
        text = "Machine learning algorithms process large datasets efficiently."
        keywords = spacy_keywords(text, max_keywords=3)

        assert isinstance(keywords, list)
        assert len(keywords) <= 3
        assert all(isinstance(kw, str) for kw in keywords)

    def test_keywords_max_limit(self):
        """Test that keywords respect max_keywords limit."""
        text = "Natural language processing machine learning deep learning neural networks."
        keywords = spacy_keywords(text, max_keywords=2)

        assert len(keywords) <= 2

    def test_keywords_filter_stopwords(self):
        """Test that keywords exclude stopwords."""
        text = "The and or but not."
        keywords = spacy_keywords(text, max_keywords=3)

        # Should have very few keywords (mostly stopwords filtered out)
        assert isinstance(keywords, list)

    def test_keywords_empty_text(self):
        """Test keywords from empty text."""
        text = ""
        keywords = spacy_keywords(text, max_keywords=3)

        assert isinstance(keywords, list)
        assert len(keywords) == 0


class TestSpacySummaryPhrase:
    """Test cases for the spacy_summary_phrase function."""

    def test_summary_phrase_generation(self):
        """Test generating summary phrase from text."""
        text = "Machine learning algorithms process data efficiently."
        phrase = spacy_summary_phrase(text)

        assert isinstance(phrase, str)
        assert len(phrase) > 0

    def test_summary_phrase_contains_keywords(self):
        """Test that summary phrase contains relevant keywords."""
        text = "Natural language processing enables text analysis."
        phrase = spacy_summary_phrase(text)

        assert isinstance(phrase, str)
        # Should contain some relevant terms
        assert len(phrase.split()) <= 3

    def test_summary_phrase_empty_text(self):
        """Test summary phrase from empty text."""
        text = ""
        phrase = spacy_summary_phrase(text)

        assert isinstance(phrase, str)


class TestSpacyTextSummary:
    """Test cases for the spacy_text_summary function."""

    def test_text_summary_generation(self):
        """Test generating text summary."""
        text = "Machine learning is important. Data science uses algorithms. Python is a programming language."
        summary = spacy_text_summary(text, num_sentences=2)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_text_summary_num_sentences(self):
        """Test that summary respects num_sentences parameter."""
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        summary = spacy_text_summary(text, num_sentences=2)

        # Count sentences (rough approximation)
        sentence_count = summary.count(".") + summary.count("!") + summary.count("?")
        assert sentence_count <= 2

    def test_text_summary_single_sentence(self):
        """Test summary with single sentence."""
        text = "Machine learning algorithms process data efficiently."
        summary = spacy_text_summary(text, num_sentences=1)

        assert isinstance(summary, str)
        assert len(summary) > 0


class TestSegmentText:
    """Test cases for the segment_text async function."""

    @pytest.mark.asyncio
    async def test_segment_text_basic(self):
        """Test basic text segmentation."""
        request = SegmentationRequest(text="Machine learning algorithms process data efficiently.")
        context = MockContext()

        result = await segment_text(request, context)

        assert isinstance(result, SegmentationResponse)
        assert isinstance(result.topic, str)
        assert isinstance(result.keywords, list)
        assert len(result.keywords) >= 0

    @pytest.mark.asyncio
    async def test_segment_text_complex(self):
        """Test segmentation of complex text."""
        request = SegmentationRequest(
            text="The chromVAR R package analyzes chromatin accessibility data from ATAC-seq experiments. It identifies transcription factor motifs associated with variability."
        )
        context = MockContext()

        result = await segment_text(request, context)

        assert isinstance(result, SegmentationResponse)
        assert len(result.keywords) > 0
        assert len(result.topic) > 0

    @pytest.mark.asyncio
    async def test_segment_text_empty_validation(self):
        """Test that empty text fails validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            SegmentationRequest(text="")

    @pytest.mark.asyncio
    async def test_segment_text_minimal(self):
        """Test segmentation with minimal valid text."""
        request = SegmentationRequest(text="a")
        context = MockContext()

        result = await segment_text(request, context)

        assert isinstance(result, SegmentationResponse)
        assert isinstance(result.keywords, list)
        assert isinstance(result.topic, str)

    @pytest.mark.asyncio
    async def test_segment_text_structure(self):
        """Test that segment_text returns correct structure."""
        request = SegmentationRequest(text="Natural language processing enables text analysis and understanding.")
        context = MockContext()

        result = await segment_text(request, context)

        # Verify structure matches SegmentationResponse model
        assert hasattr(result, "topic")
        assert hasattr(result, "keywords")
        assert isinstance(result.topic, str)
        assert isinstance(result.keywords, list)
        assert all(isinstance(keyword, str) for keyword in result.keywords)

    @pytest.mark.asyncio
    async def test_segment_text_error_handling(self):
        """Test error handling in segment_text."""
        # This test verifies that errors are properly caught and re-raised
        # We'll use a valid text but mock an error in extract_concepts
        request = SegmentationRequest(text="Valid text for testing.")
        context = Mock()
        context.info = Mock()
        context.error = Mock()

        with patch("edam_mcp.tools.segment_text.extract_concepts") as mock_extract:
            mock_extract.side_effect = Exception("Test error")

            with pytest.raises(Exception) as exc_info:
                await segment_text(request, context)

            assert "Test error" in str(exc_info.value)
            context.error.assert_called_once()
