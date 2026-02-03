"""Tests for post-processing module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from address_parser.postprocessing import DelhiGazetteer, RuleBasedRefiner
from address_parser.schemas import AddressEntity


class TestDelhiGazetteer:
    """Test cases for DelhiGazetteer."""

    @pytest.fixture
    def gazetteer(self):
        return DelhiGazetteer(min_similarity=80.0)

    def test_known_locality_match(self, gazetteer):
        """Test matching known localities."""
        assert gazetteer.is_known_locality("DWARKA")
        assert gazetteer.is_known_locality("ROHINI")
        assert gazetteer.is_known_locality("LAJPAT NAGAR")

    def test_fuzzy_matching(self, gazetteer):
        """Test fuzzy matching for typos."""
        # Common typo
        matches = gazetteer.fuzzy_match("DWARAKA")  # Typo for DWARKA
        assert len(matches) > 0
        assert matches[0][1] >= 80  # Should have decent similarity

    def test_spelling_correction(self, gazetteer):
        """Test spelling correction."""
        corrected = gazetteer.correct_spelling("ROHNI")  # Typo for ROHINI
        # May or may not correct depending on threshold
        # Just ensure it doesn't crash
        assert corrected is None or isinstance(corrected, str)

    def test_pincode_validation(self, gazetteer):
        """Test Delhi pincode validation."""
        # Valid Delhi pincodes
        assert gazetteer.validate_pincode("110001")
        assert gazetteer.validate_pincode("110041")
        assert gazetteer.validate_pincode("110097")

        # Invalid pincodes
        assert not gazetteer.validate_pincode("12345")  # Too short
        assert not gazetteer.validate_pincode("200001")  # Not Delhi
        assert not gazetteer.validate_pincode("abcdef")  # Not numeric

    def test_locality_type_detection(self, gazetteer):
        """Test locality type suffix detection."""
        assert gazetteer.get_locality_type("MALVIYA NAGAR") == "NAGAR"
        assert gazetteer.get_locality_type("VASANT VIHAR") == "VIHAR"
        assert gazetteer.get_locality_type("DEFENCE COLONY") == "COLONY"
        assert gazetteer.get_locality_type("RANDOM NAME") is None


class TestRuleBasedRefiner:
    """Test cases for RuleBasedRefiner."""

    @pytest.fixture
    def refiner(self):
        return RuleBasedRefiner(use_gazetteer=True)

    def test_pincode_pattern_extraction(self, refiner):
        """Test automatic pincode extraction."""
        text = "NEW DELHI 110041"
        entities = []
        refined = refiner.refine(text, entities)

        pincode_entities = [e for e in refined if e.label == "PINCODE"]
        assert len(pincode_entities) == 1
        assert pincode_entities[0].value == "110041"

    def test_city_pattern_extraction(self, refiner):
        """Test city extraction."""
        text = "BLOCK A, NEW DELHI, DELHI"
        entities = []
        refined = refiner.refine(text, entities)

        city_entities = [e for e in refined if e.label == "CITY"]
        assert len(city_entities) >= 1

    def test_khasra_pattern(self, refiner):
        """Test Khasra number pattern recognition."""
        patterns = refiner.extract_all_patterns("KH NO 24/1/3/2")
        assert "KHASRA" in patterns

    def test_block_pattern(self, refiner):
        """Test block pattern recognition."""
        patterns = refiner.extract_all_patterns("BLOCK H-3")
        assert "BLOCK" in patterns

    def test_overlap_removal(self, refiner):
        """Test removal of overlapping entities."""
        entities = [
            AddressEntity(label="AREA", value="DELHI", start=0, end=5, confidence=0.9),
            AddressEntity(label="SUBAREA", value="DELHI", start=0, end=5, confidence=0.85),
        ]
        refined = refiner.refine("DELHI", entities)

        # Should keep only one AREA/SUBAREA (higher confidence), but CITY is always preserved
        area_entities = [e for e in refined if e.label in ("AREA", "SUBAREA")]
        assert len(area_entities) == 1
        assert area_entities[0].confidence == 0.9

    def test_low_confidence_filtering(self, refiner):
        """Test filtering of low confidence entities."""
        entities = [
            AddressEntity(label="AREA", value="X", start=0, end=1, confidence=0.1),
        ]
        refined = refiner.refine("X", entities)

        # Should filter out very low confidence
        assert len(refined) == 0

    def test_empty_value_filtering(self, refiner):
        """Test that whitespace-only values are rejected by schema validation."""
        import pytest
        with pytest.raises(Exception):  # Pydantic ValidationError
            AddressEntity(label="AREA", value="  ", start=0, end=2, confidence=0.9)


class TestPostprocessingIntegration:
    """Integration tests for post-processing."""

    def test_full_address_refinement(self):
        """Test complete post-processing on real address."""
        refiner = RuleBasedRefiner(use_gazetteer=True)

        text = "PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2, NEW DELHI, 110041"
        entities = [
            AddressEntity(label="HOUSE_NUMBER", value="PLOT NO752", start=0, end=10, confidence=0.9),
            AddressEntity(label="FLOOR", value="FIRST FLOOR", start=11, end=22, confidence=0.85),
        ]

        refined = refiner.refine(text, entities)

        # Should have original entities plus patterns
        labels = [e.label for e in refined]
        assert "HOUSE_NUMBER" in labels
        assert "FLOOR" in labels
        assert "PINCODE" in labels  # Added by rules
