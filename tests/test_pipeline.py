"""Tests for main pipeline."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from address_parser import AddressParser, ParsedAddress


class TestAddressParser:
    """Test cases for AddressParser."""

    @pytest.fixture
    def parser(self):
        """Create rules-only parser for testing."""
        return AddressParser.rules_only(use_gazetteer=True)

    def test_basic_parsing(self, parser):
        """Test basic address parsing."""
        result = parser.parse("PLOT NO752, NEW DELHI, 110041")

        assert isinstance(result, ParsedAddress)
        assert result.raw_address == "PLOT NO752, NEW DELHI, 110041"
        assert result.normalized_address is not None

    def test_pincode_extraction(self, parser):
        """Test pincode is extracted."""
        result = parser.parse("NEW DELHI 110041")
        assert result.pincode == "110041"

    def test_empty_input(self, parser):
        """Test empty input handling."""
        result = parser.parse("")
        assert result.raw_address == ""
        assert len(result.entities) == 0

    def test_whitespace_input(self, parser):
        """Test whitespace-only input."""
        result = parser.parse("   ")
        assert len(result.entities) == 0

    def test_parse_with_timing(self, parser):
        """Test parsing with timing info."""
        response = parser.parse_with_timing("NEW DELHI 110041")

        assert response.success
        assert response.result is not None
        assert response.inference_time_ms >= 0

    def test_batch_parsing(self, parser):
        """Test batch parsing."""
        addresses = [
            "NEW DELHI 110041",
            "BLOCK A, DWARKA, 110078",
            "LAJPAT NAGAR, 110024",
        ]

        response = parser.parse_batch(addresses)

        assert response.success
        assert len(response.results) == 3
        assert response.total_inference_time_ms >= 0
        assert response.avg_inference_time_ms >= 0

    def test_entity_fields(self, parser):
        """Test convenience fields are populated."""
        result = parser.parse("PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041")

        # At least some fields should be populated (with rules-only parser)
        populated = [
            result.house_number,
            result.floor,
            result.block,
            result.pincode,
            result.city,
        ]
        # Check that at least some are populated
        assert any(f is not None for f in populated)


class TestRulesOnlyParser:
    """Test rules-only parsing mode."""

    def test_rules_only_creation(self):
        """Test rules-only parser creation."""
        parser = AddressParser.rules_only()
        assert parser.model is None
        assert parser.tokenizer is None
        assert parser.refiner is not None

    def test_rules_extract_patterns(self):
        """Test pattern extraction without model."""
        parser = AddressParser.rules_only()

        result = parser.parse("KH NO 45/2, BLOCK A-5, NEW DELHI, 110001")

        # Should extract some entities via rules
        labels = [e.label for e in result.entities]
        assert "PINCODE" in labels


class TestParserEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def parser(self):
        return AddressParser.rules_only()

    def test_special_characters(self, parser):
        """Test handling of special characters."""
        result = parser.parse("H.NO. 123/A-5, BLOCK-B, DELHI")
        assert result is not None
        assert isinstance(result, ParsedAddress)

    def test_very_long_address(self, parser):
        """Test handling of long addresses."""
        long_address = "PLOT NO 123, " * 50 + "NEW DELHI, 110041"
        result = parser.parse(long_address)
        assert result is not None

    def test_unicode_characters(self, parser):
        """Test handling of Unicode."""
        result = parser.parse("H.NO. 123, गली नं 5, DELHI")
        assert result is not None

    def test_numbers_only(self, parser):
        """Test address with mostly numbers."""
        result = parser.parse("123/456/789, 110041")
        assert result is not None
        # Should still extract pincode
        assert result.pincode == "110041"


class TestParserOutput:
    """Test output format and structure."""

    @pytest.fixture
    def parser(self):
        return AddressParser.rules_only()

    def test_entity_structure(self, parser):
        """Test entity objects have correct structure."""
        result = parser.parse("NEW DELHI, 110041")

        for entity in result.entities:
            assert hasattr(entity, "label")
            assert hasattr(entity, "value")
            assert hasattr(entity, "start")
            assert hasattr(entity, "end")
            assert hasattr(entity, "confidence")
            assert 0 <= entity.confidence <= 1

    def test_entity_offsets(self, parser):
        """Test entity offsets are valid."""
        result = parser.parse("NEW DELHI, 110041")

        for entity in result.entities:
            assert entity.start >= 0
            assert entity.end > entity.start
            assert entity.end <= len(result.normalized_address)

    def test_json_serialization(self, parser):
        """Test results can be JSON serialized."""
        import json

        result = parser.parse("NEW DELHI, 110041")
        json_str = result.model_dump_json()

        # Should not raise
        parsed = json.loads(json_str)
        assert "raw_address" in parsed
        assert "entities" in parsed
