"""Rule-based post-processing for entity refinement."""

import re

from address_parser.postprocessing.gazetteer import DelhiGazetteer
from address_parser.schemas import AddressEntity


class RuleBasedRefiner:
    """
    Post-processing rules for refining NER predictions.

    Handles:
    - Pattern-based entity detection (pincodes, khasra numbers)
    - Entity boundary correction
    - Confidence adjustment
    - Validation and filtering
    """

    # Regex patterns for deterministic entities
    PATTERNS = {
        "PINCODE": re.compile(r'\b[1-9]\d{5}\b'),
        "KHASRA": re.compile(
            r'\b(?:KH\.?\s*(?:NO\.?)?\s*|KHASRA\s*(?:NO\.?)?\s*)[\d/]+(?:[/-]\d+)*\b',
            re.IGNORECASE
        ),
        "HOUSE_NUMBER": re.compile(
            r'\b(?:H\.?\s*(?:NO\.?)?\s*|HOUSE\s*(?:NO\.?)?\s*|PLOT\s*(?:NO\.?)?\s*)?[A-Z]?\d+[A-Z]?(?:[-/]\d+)*\b',
            re.IGNORECASE
        ),
        "FLOOR": re.compile(
            r'\b(?:GROUND|FIRST|SECOND|THIRD|FOURTH|FIFTH|1ST|2ND|3RD|4TH|5TH|GF|FF|SF|TF)?\s*(?:FLOOR|FLR)\b',
            re.IGNORECASE
        ),
        "BLOCK": re.compile(
            r'\b(?:BLOCK|BLK|BL)\s*[A-Z]?[-]?[A-Z0-9]+\b',
            re.IGNORECASE
        ),
        "SECTOR": re.compile(
            r'\b(?:SECTOR|SEC)\s*\d+[A-Z]?\b',
            re.IGNORECASE
        ),
        "GALI": re.compile(
            r'\b(?:GALI|GALLI|LANE)\s*(?:NO\.?)?\s*\d+\b',
            re.IGNORECASE
        ),
    }

    # City patterns
    CITY_PATTERNS = [
        (re.compile(r'\bNEW\s+DELHI\b', re.IGNORECASE), "NEW DELHI"),
        (re.compile(r'\bDELHI\b', re.IGNORECASE), "DELHI"),
        (re.compile(r'\bNOIDA\b', re.IGNORECASE), "NOIDA"),
        (re.compile(r'\bGURUGRAM\b', re.IGNORECASE), "GURUGRAM"),
        (re.compile(r'\bGURGAON\b', re.IGNORECASE), "GURGAON"),
        (re.compile(r'\bFARIDABAD\b', re.IGNORECASE), "FARIDABAD"),
        (re.compile(r'\bGHAZIABAD\b', re.IGNORECASE), "GHAZIABAD"),
    ]

    # State patterns
    STATE_PATTERNS = [
        (re.compile(r'\bDELHI\b', re.IGNORECASE), "DELHI"),
        (re.compile(r'\bHARYANA\b', re.IGNORECASE), "HARYANA"),
        (re.compile(r'\bUTTAR\s+PRADESH\b', re.IGNORECASE), "UTTAR PRADESH"),
        (re.compile(r'\bU\.?\s*P\.?\b'), "UTTAR PRADESH"),
    ]

    # Colony/Nagar indicators
    COLONY_SUFFIXES = [
        "NAGAR", "VIHAR", "COLONY", "ENCLAVE", "PARK", "GARDEN",
        "PURI", "BAGH", "KUNJ", "EXTENSION", "EXTN", "PHASE",
    ]

    def __init__(self, use_gazetteer: bool = True):
        """
        Initialize refiner.

        Args:
            use_gazetteer: Use gazetteer for validation/correction
        """
        self.gazetteer = DelhiGazetteer() if use_gazetteer else None

    def refine(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """
        Refine entity predictions.

        Args:
            text: Original address text
            entities: Predicted entities from NER model

        Returns:
            Refined list of entities
        """
        refined = list(entities)

        # Add rule-based entities that may have been missed
        refined = self._add_pattern_entities(text, refined)

        # Correct entity boundaries
        refined = self._correct_boundaries(text, refined)

        # Adjust confidence scores
        refined = self._adjust_confidence(text, refined)

        # Remove duplicates and overlapping entities
        refined = self._remove_overlaps(refined)

        # Validate entities
        refined = self._validate_entities(refined)

        return refined

    def _add_pattern_entities(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Add entities detected by regex patterns."""
        result = list(entities)
        existing_spans = {(e.start, e.end) for e in entities}

        # Check for pincode
        if not any(e.label == "PINCODE" for e in entities):
            match = self.PATTERNS["PINCODE"].search(text)
            if match and (match.start(), match.end()) not in existing_spans:
                result.append(AddressEntity(
                    label="PINCODE",
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=1.0  # Rule-based, high confidence
                ))

        # Check for city
        if not any(e.label == "CITY" for e in entities):
            for pattern, city_name in self.CITY_PATTERNS:
                match = pattern.search(text)
                if match and (match.start(), match.end()) not in existing_spans:
                    result.append(AddressEntity(
                        label="CITY",
                        value=city_name,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95
                    ))
                    break

        # Check for state
        if not any(e.label == "STATE" for e in entities):
            for pattern, state_name in self.STATE_PATTERNS:
                match = pattern.search(text)
                if match and (match.start(), match.end()) not in existing_spans:
                    # Avoid tagging "DELHI" as state if it's already a city
                    if state_name == "DELHI" and any(e.label == "CITY" and "DELHI" in e.value.upper() for e in result):
                        continue
                    result.append(AddressEntity(
                        label="STATE",
                        value=state_name,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.90
                    ))
                    break

        return result

    def _correct_boundaries(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Correct entity boundaries based on patterns."""
        result = []

        for entity in entities:
            corrected = entity.model_copy()

            # Expand KHASRA to include full pattern
            if entity.label == "KHASRA":
                match = self.PATTERNS["KHASRA"].search(text)
                if match:
                    corrected.value = match.group(0)
                    corrected.start = match.start()
                    corrected.end = match.end()

            # Expand BLOCK to include identifier
            elif entity.label == "BLOCK":
                match = self.PATTERNS["BLOCK"].search(text)
                if match:
                    corrected.value = match.group(0)
                    corrected.start = match.start()
                    corrected.end = match.end()

            # Expand FLOOR to include floor number
            elif entity.label == "FLOOR":
                match = self.PATTERNS["FLOOR"].search(text)
                if match:
                    corrected.value = match.group(0)
                    corrected.start = match.start()
                    corrected.end = match.end()

            # Clean up leading/trailing whitespace from value
            corrected.value = corrected.value.strip()

            result.append(corrected)

        return result

    def _adjust_confidence(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Adjust confidence scores based on patterns and gazetteer."""
        result = []

        for entity in entities:
            adjusted = entity.model_copy()

            # Boost confidence for pattern matches
            if entity.label in self.PATTERNS:
                pattern = self.PATTERNS[entity.label]
                if pattern.fullmatch(entity.value):
                    adjusted.confidence = min(1.0, entity.confidence + 0.1)

            # Boost confidence for gazetteer matches
            if self.gazetteer and entity.label in ("AREA", "SUBAREA", "COLONY"):
                if self.gazetteer.is_known_locality(entity.value):
                    adjusted.confidence = min(1.0, entity.confidence + 0.15)

            # Reduce confidence for very short entities
            if len(entity.value) < 3:
                adjusted.confidence = max(0.0, entity.confidence - 0.2)

            result.append(adjusted)

        return result

    def _remove_overlaps(
        self,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Remove overlapping entities, keeping higher confidence ones."""
        if not entities:
            return entities

        # Sort by confidence (descending) then by start position
        sorted_entities = sorted(entities, key=lambda e: (-e.confidence, e.start))

        result: list[AddressEntity] = []
        used_ranges: list[tuple[int, int]] = []

        for entity in sorted_entities:
            # Check for overlap with existing entities
            overlaps = False
            for start, end in used_ranges:
                if not (entity.end <= start or entity.start >= end):
                    overlaps = True
                    break

            if not overlaps:
                result.append(entity)
                used_ranges.append((entity.start, entity.end))

        # Sort by position for output
        return sorted(result, key=lambda e: e.start)

    def _validate_entities(
        self,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Validate and filter entities."""
        result = []

        for entity in entities:
            # Skip empty values
            if not entity.value.strip():
                continue

            # Skip very low confidence
            if entity.confidence < 0.3:
                continue

            # Validate pincode format
            if entity.label == "PINCODE":
                if not re.fullmatch(r'[1-9]\d{5}', entity.value):
                    continue
                if self.gazetteer and not self.gazetteer.validate_pincode(entity.value):
                    # Pincode outside Delhi range - reduce confidence but keep
                    entity = entity.model_copy()
                    entity.confidence *= 0.7

            result.append(entity)

        return result

    def extract_all_patterns(self, text: str) -> dict[str, list[str]]:
        """
        Extract all pattern-based entities from text.

        Returns dict of label -> list of matched values.
        """
        results = {}

        for label, pattern in self.PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                results[label] = matches

        return results
