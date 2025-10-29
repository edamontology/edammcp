"""Tests for the KeywordMatcher functionality."""

import pytest

from edam_mcp.models.responses import KeywordMatch
from edam_mcp.ontology.keywords_matcher import KeywordMatcher
from edam_mcp.ontology.loader import OntologyLoader


class TestKeywordMatcher:
    """Test cases for the KeywordMatcher class."""

    @pytest.fixture
    def ontology_loader(self):
        """Create a real ontology loader instance."""
        loader = OntologyLoader()
        loader.load_ontology()
        return loader

    @pytest.fixture
    def keyword_matcher(self, ontology_loader):
        """Create a KeywordMatcher instance with loaded ontology."""
        return KeywordMatcher(ontology_loader)

    def test_initialization(self, keyword_matcher):
        """Test that KeywordMatcher initializes correctly."""
        assert keyword_matcher.ontology_loader is not None
        assert keyword_matcher.term_descriptions is not None
        assert len(keyword_matcher.term_descriptions) > 0

    def test_match_by_substring_single_keyword(self, keyword_matcher):
        """Test substring matching with a single keyword."""
        keywords = ["alignment"]
        matches = keyword_matcher.match_by_substring(keywords)

        assert isinstance(matches, list)
        assert len(matches) > 0

        # Check that all matches have the correct structure
        for match in matches:
            assert isinstance(match, KeywordMatch)
            assert match.concept_uri
            assert match.concept_label
            assert match.confidence > 0
            assert match.match_type == "substring"
            assert "alignment" in match.matched_keywords

    def test_match_by_substring_multiple_keywords(self, keyword_matcher):
        """Test substring matching with multiple keywords."""
        keywords = ["sequence", "alignment"]
        matches = keyword_matcher.match_by_substring(keywords)

        assert isinstance(matches, list)
        assert len(matches) > 0

        # Matches should be sorted by confidence (match count)
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].confidence >= matches[i + 1].confidence

    def test_match_by_substring_case_insensitive(self, keyword_matcher):
        """Test that substring matching is case-insensitive."""
        keywords_lower = ["alignment"]
        keywords_upper = ["ALIGNMENT"]
        keywords_mixed = ["Alignment"]

        matches_lower = keyword_matcher.match_by_substring(keywords_lower)
        matches_upper = keyword_matcher.match_by_substring(keywords_upper)
        matches_mixed = keyword_matcher.match_by_substring(keywords_mixed)

        # Should return same number of results regardless of case
        assert len(matches_lower) == len(matches_upper) == len(matches_mixed)

    def test_match_by_substring_empty_keywords(self, keyword_matcher):
        """Test substring matching with empty keyword list."""
        keywords = []
        matches = keyword_matcher.match_by_substring(keywords)

        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_match_by_substring_no_matches(self, keyword_matcher):
        """Test substring matching with keywords that don't match anything."""
        keywords = ["xyzabc123nonexistent"]
        matches = keyword_matcher.match_by_substring(keywords)

        assert isinstance(matches, list)
        # Should return empty list or very few matches
        assert len(matches) == 0

    def test_match_by_substring_ranking(self, keyword_matcher):
        """Test that results are ranked by match count."""
        keywords = ["sequence", "alignment", "protein"]
        matches = keyword_matcher.match_by_substring(keywords)

        assert len(matches) > 0

        # First result should have highest confidence
        if len(matches) > 1:
            assert matches[0].confidence >= matches[-1].confidence

        # Check that matched_keywords contains the correct keywords
        for match in matches:
            assert all(kw in keywords for kw in match.matched_keywords)

    def test_match_by_substring_special_characters(self, keyword_matcher):
        """Test substring matching with special characters in keywords."""
        keywords = ["sequence-alignment", "rna/dna"]
        matches = keyword_matcher.match_by_substring(keywords)

        # Should handle gracefully (may or may not match depending on preprocessing)
        assert isinstance(matches, list)

    # Tests for Mode 2: List Embeddings

    def test_match_by_list_embeddings_initialization(self, keyword_matcher):
        """Test that embeddings are initialized when needed."""
        keywords = ["alignment"]
        matches = keyword_matcher.match_by_list_embeddings(keywords)

        # Embeddings should be created during first call
        assert keyword_matcher.embedding_model is not None
        assert len(keyword_matcher.term_embeddings) > 0
        assert isinstance(matches, list)

    def test_match_by_list_embeddings_single_keyword(self, keyword_matcher):
        """Test list embeddings matching with a single keyword."""
        keywords = ["alignment"]
        matches = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.3)

        assert isinstance(matches, list)
        assert len(matches) > 0

        # Check match structure
        for match in matches:
            assert isinstance(match, KeywordMatch)
            assert match.concept_uri
            assert match.concept_label
            assert match.confidence >= 0.3
            assert match.match_type == "list_embeddings"
            assert len(match.matched_keywords) == 1
            assert match.matched_keywords[0] == "alignment"

    def test_match_by_list_embeddings_multiple_keywords(self, keyword_matcher):
        """Test list embeddings matching with multiple keywords."""
        keywords = ["sequence", "alignment", "protein"]
        matches = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.3)

        assert isinstance(matches, list)
        assert len(matches) > 0

        # Check that each match has one of the keywords
        for match in matches:
            assert len(match.matched_keywords) == 1
            assert match.matched_keywords[0] in keywords

    def test_match_by_list_embeddings_threshold_filtering(self, keyword_matcher):
        """Test that threshold properly filters results."""
        keywords = ["alignment"]

        # Lower threshold should return more results
        matches_low = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.2)
        matches_high = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.5)

        assert len(matches_low) >= len(matches_high)

        # All confidence scores should be above threshold
        for match in matches_high:
            assert match.confidence >= 0.5

    def test_match_by_list_embeddings_max_aggregation(self, keyword_matcher):
        """Test that max aggregation works correctly."""
        keywords = ["sequence", "alignment"]
        matches = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.3)

        assert isinstance(matches, list)

        # Each match should have the keyword that gave the best score
        for match in matches:
            assert len(match.matched_keywords) == 1
            assert match.matched_keywords[0] in keywords

    def test_match_by_list_embeddings_sorting(self, keyword_matcher):
        """Test that results are sorted by confidence in descending order."""
        keywords = ["alignment", "protein"]
        matches = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.3)

        assert len(matches) > 0

        # Check sorting
        if len(matches) > 1:
            for i in range(len(matches) - 1):
                assert matches[i].confidence >= matches[i + 1].confidence

    def test_match_by_list_embeddings_empty_keywords(self, keyword_matcher):
        """Test list embeddings matching with empty keyword list."""
        keywords = []
        matches = keyword_matcher.match_by_list_embeddings(keywords)

        assert isinstance(matches, list)
        assert len(matches) == 0

    def test_match_by_list_embeddings_high_threshold(self, keyword_matcher):
        """Test that very high threshold returns no or few results."""
        keywords = ["alignment"]
        matches = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.99)

        assert isinstance(matches, list)
        # May return empty list if no very close matches

    def test_match_by_list_embeddings_semantic_similarity(self, keyword_matcher):
        """Test that semantically similar keywords find related terms."""
        # These keywords are semantically related
        keywords = ["align", "alignment"]
        matches = keyword_matcher.match_by_list_embeddings(keywords, threshold=0.3)

        assert len(matches) > 0

        # Should find terms related to alignment
        alignment_terms = [m for m in matches if "align" in m.concept_label.lower()]
        assert len(alignment_terms) > 0
