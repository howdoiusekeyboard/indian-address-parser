"""Rule-based post-processing for entity refinement."""

import re

from address_parser.postprocessing.gazetteer import DelhiGazetteer
from address_parser.schemas import AddressEntity


class RuleBasedRefiner:
    """
    Post-processing rules for refining NER predictions.

    Handles:
    - Pattern-based entity detection (pincodes, khasra numbers)
    - Entity boundary correction using gazetteer
    - Entity merging for fragmented predictions
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
            r'\b(?:GALI|GALLI|LANE)\s*(?:NO\.?)?\s*\d+[A-Z]?\b',
            re.IGNORECASE
        ),
    }

    # Area patterns - directional areas
    AREA_PATTERNS = [
        (re.compile(r'\bSOUTH\s+DELHI\b', re.IGNORECASE), "SOUTH DELHI"),
        (re.compile(r'\bNORTH\s+DELHI\b', re.IGNORECASE), "NORTH DELHI"),
        (re.compile(r'\bEAST\s+DELHI\b', re.IGNORECASE), "EAST DELHI"),
        (re.compile(r'\bWEST\s+DELHI\b', re.IGNORECASE), "WEST DELHI"),
        (re.compile(r'\bCENTRAL\s+DELHI\b', re.IGNORECASE), "CENTRAL DELHI"),
        (re.compile(r'\bSOUTH\s+WEST\s+DELHI\b', re.IGNORECASE), "SOUTH WEST DELHI"),
        (re.compile(r'\bNORTH\s+WEST\s+DELHI\b', re.IGNORECASE), "NORTH WEST DELHI"),
        (re.compile(r'\bNORTH\s+EAST\s+DELHI\b', re.IGNORECASE), "NORTH EAST DELHI"),
        (re.compile(r'\bSOUTH\s+EAST\s+DELHI\b', re.IGNORECASE), "SOUTH EAST DELHI"),
        (re.compile(r'\bOUTER\s+DELHI\b', re.IGNORECASE), "OUTER DELHI"),
    ]

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

    # Known multi-word localities that get fragmented
    KNOWN_LOCALITIES = [
        "LAJPAT NAGAR", "MALVIYA NAGAR", "KAROL BAGH", "HAUZ KHAS",
        "GREEN PARK", "GREATER KAILASH", "DEFENCE COLONY", "SOUTH EXTENSION",
        "CHITTARANJAN PARK", "NEHRU PLACE", "SARITA VIHAR", "VASANT KUNJ",
        "CIVIL LINES", "MODEL TOWN", "MUKHERJEE NAGAR", "KAMLA NAGAR",
        "ASHOK VIHAR", "SHALIMAR BAGH", "PREET VIHAR", "MAYUR VIHAR",
        "LAKSHMI NAGAR", "GANDHI NAGAR", "DILSHAD GARDEN", "ANAND VIHAR",
        "UTTAM NAGAR", "TILAK NAGAR", "RAJOURI GARDEN", "PUNJABI BAGH",
        "PASCHIM VIHAR", "CONNAUGHT PLACE", "RAJENDER NAGAR", "PATEL NAGAR",
        "KIRTI NAGAR", "LODHI ROAD", "GOLF LINKS", "SANGAM VIHAR",
        "GOVINDPURI", "AMBEDKAR NAGAR", "LADO SARAI", "KAUNWAR SINGH NAGAR",
        "BABA HARI DAS COLONY", "SWARN PARK", "CHANCHAL PARK", "DURGA PARK",
        "RAJ NAGAR", "SADH NAGAR", "VIJAY ENCLAVE", "PALAM COLONY",
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

        # First: detect and fix known localities from gazetteer
        refined = self._fix_known_localities(text, refined)

        # Add rule-based entities that may have been missed
        refined = self._add_pattern_entities(text, refined)

        # Detect area patterns (SOUTH DELHI, etc.)
        refined = self._add_area_patterns(text, refined)

        # Correct entity boundaries
        refined = self._correct_boundaries(text, refined)

        # Merge fragmented entities
        refined = self._merge_fragmented_entities(text, refined)

        # Adjust confidence scores
        refined = self._adjust_confidence(text, refined)

        # Remove duplicates and overlapping entities
        refined = self._remove_overlaps(refined)

        # Validate entities
        refined = self._validate_entities(refined)

        return refined

    def _fix_known_localities(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Fix fragmented known localities using gazetteer lookup."""
        text_upper = text.upper()
        result = []
        used_ranges: list[tuple[int, int]] = []

        # First pass: find all known localities in text
        locality_entities = []
        for locality in self.KNOWN_LOCALITIES:
            idx = 0
            while True:
                pos = text_upper.find(locality, idx)
                if pos == -1:
                    break
                end = pos + len(locality)
                locality_entities.append(AddressEntity(
                    label="SUBAREA",
                    value=text[pos:end],
                    start=pos,
                    end=end,
                    confidence=0.95
                ))
                used_ranges.append((pos, end))
                idx = end

        # Also check area patterns
        for pattern, area_name in self.AREA_PATTERNS:
            match = pattern.search(text)
            if match:
                start, end = match.start(), match.end()
                # Check for overlap with existing ranges
                overlaps = any(
                    not (end <= s or start >= e)
                    for s, e in used_ranges
                )
                if not overlaps:
                    locality_entities.append(AddressEntity(
                        label="AREA",
                        value=area_name,
                        start=start,
                        end=end,
                        confidence=0.95
                    ))
                    used_ranges.append((start, end))

        # Filter out original entities that overlap with found localities
        for entity in entities:
            # Check if entity overlaps with any locality range
            overlaps_locality = any(
                not (entity.end <= start or entity.start >= end)
                for start, end in used_ranges
            )

            if overlaps_locality and entity.label in ("AREA", "SUBAREA", "COLONY", "CITY"):
                # Skip this fragmented entity
                continue

            result.append(entity)

        # Add the locality entities
        result.extend(locality_entities)

        return result

    def _add_area_patterns(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Add area patterns like SOUTH DELHI, NORTH DELHI (already handled in _fix_known_localities)."""
        # This is now handled in _fix_known_localities to avoid duplicates
        return entities

    def _merge_fragmented_entities(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Merge adjacent entities of same type that should be together."""
        if len(entities) < 2:
            return entities

        # Sort by position
        sorted_entities = sorted(entities, key=lambda e: e.start)
        result = []
        i = 0

        while i < len(sorted_entities):
            current = sorted_entities[i]

            # Look for adjacent entities to merge
            if current.label in ("AREA", "SUBAREA", "COLONY", "CITY"):
                merged_end = current.end
                merged_confidence = current.confidence
                j = i + 1

                # Check subsequent entities
                while j < len(sorted_entities):
                    next_ent = sorted_entities[j]

                    # Check if adjacent (within 2 chars - allows for space)
                    gap = next_ent.start - merged_end
                    if gap <= 2 and next_ent.label in ("AREA", "SUBAREA", "COLONY", "CITY"):
                        # Check if the merged text forms a known locality
                        merged_text = text[current.start:next_ent.end].strip()
                        if self._is_valid_merge(merged_text):
                            merged_end = next_ent.end
                            merged_confidence = max(merged_confidence, next_ent.confidence)
                            j += 1
                        else:
                            break
                    else:
                        break

                # Create merged entity if we merged anything
                if j > i + 1:
                    merged_value = text[current.start:merged_end].strip()
                    result.append(AddressEntity(
                        label=current.label,
                        value=merged_value,
                        start=current.start,
                        end=merged_end,
                        confidence=merged_confidence
                    ))
                    i = j
                    continue

            result.append(current)
            i += 1

        return result

    def _is_valid_merge(self, text: str) -> bool:
        """Check if merged text forms a valid locality name."""
        text_upper = text.upper().strip()

        # Check against known localities
        if text_upper in self.KNOWN_LOCALITIES:
            return True

        # Check gazetteer
        if self.gazetteer and self.gazetteer.is_known_locality(text_upper, threshold=80):
            return True

        # Check if ends with common suffix
        for suffix in self.COLONY_SUFFIXES:
            if text_upper.endswith(suffix):
                return True

        return False

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

        # Check for city - DELHI addresses always have DELHI as city
        has_city = any(e.label == "CITY" for e in result)
        if not has_city:
            # If text contains DELHI anywhere, set city to DELHI
            if "DELHI" in text.upper():
                # Find the last occurrence of DELHI (usually the city mention)
                delhi_positions = [m.start() for m in re.finditer(r'\bDELHI\b', text.upper())]
                if delhi_positions:
                    pos = delhi_positions[-1]  # Use last occurrence
                    result.append(AddressEntity(
                        label="CITY",
                        value="DELHI",
                        start=pos,
                        end=pos + 5,
                        confidence=0.90
                    ))
            else:
                # Check other city patterns
                for pattern, city_name in self.CITY_PATTERNS:
                    if city_name == "DELHI":
                        continue  # Already handled above
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
            updates: dict[str, object] = {}

            # Expand KHASRA to include full pattern
            if entity.label == "KHASRA":
                match = self.PATTERNS["KHASRA"].search(text)
                if match:
                    updates = {"value": match.group(0), "start": match.start(), "end": match.end()}

            # Expand BLOCK to include identifier
            elif entity.label == "BLOCK":
                match = self.PATTERNS["BLOCK"].search(text)
                if match:
                    updates = {"value": match.group(0), "start": match.start(), "end": match.end()}

            # Expand FLOOR to include floor number
            elif entity.label == "FLOOR":
                match = self.PATTERNS["FLOOR"].search(text)
                if match:
                    updates = {"value": match.group(0), "start": match.start(), "end": match.end()}

            # Clean up leading/trailing whitespace from value
            final_value = (updates.get("value") or entity.value).strip()
            if final_value != entity.value or updates:
                updates["value"] = final_value

            result.append(entity.model_copy(update=updates) if updates else entity)

        return result

    def _adjust_confidence(
        self,
        text: str,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Adjust confidence scores based on patterns and gazetteer."""
        result = []

        for entity in entities:
            new_confidence = entity.confidence

            # Boost confidence for pattern matches
            if entity.label in self.PATTERNS:
                pattern = self.PATTERNS[entity.label]
                if pattern.fullmatch(entity.value):
                    new_confidence = min(1.0, new_confidence + 0.1)

            # Boost confidence for gazetteer matches
            if self.gazetteer and entity.label in ("AREA", "SUBAREA", "COLONY"):
                if self.gazetteer.is_known_locality(entity.value):
                    new_confidence = min(1.0, new_confidence + 0.15)

            # Reduce confidence for very short entities
            if len(entity.value) < 3:
                new_confidence = max(0.0, new_confidence - 0.2)

            if new_confidence != entity.confidence:
                result.append(entity.model_copy(update={"confidence": new_confidence}))
            else:
                result.append(entity)

        return result

    def _remove_overlaps(
        self,
        entities: list[AddressEntity]
    ) -> list[AddressEntity]:
        """Remove overlapping entities, keeping higher confidence ones."""
        if not entities:
            return entities

        # Separate CITY and PINCODE entities - these should always be kept
        # as they represent different semantic levels than AREA/SUBAREA
        preserved_labels = {"CITY", "PINCODE", "STATE"}
        preserved_entities = [e for e in entities if e.label in preserved_labels]
        other_entities = [e for e in entities if e.label not in preserved_labels]

        # Sort non-preserved by confidence (descending) then by start position
        sorted_entities = sorted(other_entities, key=lambda e: (-e.confidence, e.start))

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

        # Add back preserved entities (CITY, PINCODE, STATE)
        result.extend(preserved_entities)

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
                    entity = entity.model_copy(update={"confidence": entity.confidence * 0.7})

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
