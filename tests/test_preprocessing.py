"""Tests for preprocessing module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from address_parser.preprocessing import AddressNormalizer, HindiTransliterator


class TestAddressNormalizer:
    """Test cases for AddressNormalizer."""

    @pytest.fixture
    def normalizer(self):
        return AddressNormalizer(uppercase=True, expand_abbrev=True)

    def test_basic_normalization(self, normalizer):
        """Test basic text normalization."""
        result = normalizer.normalize("  plot no  123  ")
        assert result == "PLOT NO 123"

    def test_abbreviation_expansion(self, normalizer):
        """Test abbreviation expansion."""
        cases = [
            ("H.NO. 123", "HOUSE NO"),
            ("KH NO 45/2", "KHASRA NO 45/2"),
            ("GF", "GROUND FLOOR"),
            ("BLK A", "BLOCK A"),
        ]
        for input_text, expected_partial in cases:
            result = normalizer.normalize(input_text)
            assert expected_partial in result, f"Expected '{expected_partial}' in '{result}'"

    def test_punctuation_standardization(self, normalizer):
        """Test punctuation handling."""
        result = normalizer.normalize("123,,,  456---789")
        assert ",," not in result
        assert "--" not in result

    def test_pincode_extraction(self, normalizer):
        """Test pincode extraction."""
        address = "NEW DELHI, 110041"
        pincode = normalizer.extract_pincode(address)
        assert pincode == "110041"

    def test_invalid_pincode(self, normalizer):
        """Test that invalid pincodes are not extracted."""
        # Too short
        assert normalizer.extract_pincode("12345") is None
        # Starts with 0
        assert normalizer.extract_pincode("012345") is None

    def test_empty_input(self, normalizer):
        """Test empty input handling."""
        assert normalizer.normalize("") == ""
        assert normalizer.normalize(None) == ""

    def test_tokenization(self, normalizer):
        """Test address tokenization."""
        text = "PLOT NO752 BLOCK H-3"
        tokens = normalizer.tokenize(text)
        assert "PLOT" in tokens
        assert "NO752" in tokens or "752" in tokens
        assert "BLOCK" in tokens


class TestHindiTransliterator:
    """Test cases for HindiTransliterator."""

    @pytest.fixture
    def transliterator(self):
        return HindiTransliterator(use_known_terms=True)

    def test_devanagari_detection(self, transliterator):
        """Test Devanagari script detection."""
        assert transliterator.contains_devanagari("गली नं 5")
        assert not transliterator.contains_devanagari("GALI NO 5")

    def test_known_term_transliteration(self, transliterator):
        """Test transliteration of known terms."""
        cases = [
            ("गली", "GALI"),
            ("नगर", "NAGAR"),
            ("कॉलोनी", "COLONY"),
            ("दिल्ली", "DELHI"),
        ]
        for hindi, expected in cases:
            result = transliterator.transliterate(hindi)
            assert expected in result.upper(), f"Expected '{expected}' in '{result}'"

    def test_digit_conversion(self, transliterator):
        """Test Devanagari digit conversion."""
        result = transliterator.transliterate("१२३४५६")
        assert "123456" in result

    def test_mixed_script(self, transliterator):
        """Test mixed Hindi-English text."""
        text = "PLOT नं 123 गली NO 5"
        result = transliterator.normalize_mixed_script(text)
        assert "PLOT" in result.upper()
        assert "GALI" in result.upper()

    def test_latin_passthrough(self, transliterator):
        """Test that Latin text passes through unchanged."""
        text = "BLOCK A-5 NEW DELHI"
        result = transliterator.transliterate(text)
        assert result.upper() == text

    def test_script_ratio(self, transliterator):
        """Test script ratio calculation."""
        ratios = transliterator.get_script_ratio("Hello दिल्ली")
        assert ratios["latin"] > 0
        assert ratios["devanagari"] > 0


class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing flow."""
        normalizer = AddressNormalizer()
        transliterator = HindiTransliterator()

        # Mixed script input
        input_text = "H.NO. 123, गली नं 5, दिल्ली"

        # Transliterate first
        if transliterator.contains_devanagari(input_text):
            text = transliterator.normalize_mixed_script(input_text)
        else:
            text = input_text

        # Then normalize
        result = normalizer.normalize(text)

        assert "HOUSE NO" in result
        assert "GALI" in result
        assert "DELHI" in result
