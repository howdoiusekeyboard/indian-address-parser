"""
Main address parsing pipeline.

Orchestrates preprocessing, model inference, and post-processing
to extract structured entities from Indian addresses.
"""

import time
import warnings
from pathlib import Path

from transformers import AutoTokenizer, logging as hf_logging

# Suppress false positive tokenizer warnings in transformers 4.57+
# The Mistral regex warning is incorrectly triggered for BERT tokenizers
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

from address_parser.models.config import ID2LABEL, ModelConfig
from address_parser.postprocessing import DelhiGazetteer, RuleBasedRefiner
from address_parser.preprocessing import AddressNormalizer, HindiTransliterator
from address_parser.schemas import (
    AddressEntity,
    BatchParseResponse,
    ParsedAddress,
    ParseResponse,
)


class AddressParser:
    """
    Main address parsing pipeline.

    Combines:
    - Text normalization and Hindi transliteration
    - mBERT-CRF model for NER
    - Rule-based post-processing with gazetteer

    Example:
        >>> parser = AddressParser.from_pretrained("./models/address_ner_v3")
        >>> result = parser.parse("PLOT NO752 FIRST FLOOR, NEW DELHI, 110041")
        >>> print(result.house_number)  # "PLOT NO752"
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        config: ModelConfig | None = None,
        device: str = "cpu",
        use_rules: bool = True,
        use_gazetteer: bool = True,
    ):
        """
        Initialize parser.

        Args:
            model: Trained NER model (BertCRFForTokenClassification)
            tokenizer: HuggingFace tokenizer
            config: Model configuration
            device: Device to run on ('cpu', 'cuda', 'mps')
            use_rules: Enable rule-based post-processing
            use_gazetteer: Enable gazetteer for validation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ModelConfig()
        self.device = device

        # Initialize preprocessing
        self.normalizer = AddressNormalizer(uppercase=True, expand_abbrev=True)
        self.transliterator = HindiTransliterator(use_known_terms=True)

        # Initialize post-processing
        self.refiner = RuleBasedRefiner(use_gazetteer=use_gazetteer) if use_rules else None
        self.gazetteer = DelhiGazetteer() if use_gazetteer else None

        # Move model to device
        if self.model is not None:
            self.model.to(device)
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        device: str = "cpu",
        use_rules: bool = True,
        use_gazetteer: bool = True,
    ) -> AddressParser:
        """
        Load parser from pretrained model directory.

        Args:
            model_path: Path to saved model directory
            device: Device to run on
            use_rules: Enable rule-based post-processing
            use_gazetteer: Enable gazetteer for validation

        Returns:
            Initialized AddressParser
        """
        from address_parser.models import BertCRFForTokenClassification

        model_path = Path(model_path)

        # Load model
        model = BertCRFForTokenClassification.from_pretrained(str(model_path), device=device)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=device,
            use_rules=use_rules,
            use_gazetteer=use_gazetteer,
        )

    @classmethod
    def rules_only(cls, use_gazetteer: bool = True) -> AddressParser:
        """
        Create a rules-only parser (no ML model).

        Useful for testing or when model is not available.
        """
        return cls(
            model=None,
            tokenizer=None,
            use_rules=True,
            use_gazetteer=use_gazetteer,
        )

    def parse(self, address: str) -> ParsedAddress:
        """
        Parse a single address.

        Args:
            address: Raw address string

        Returns:
            ParsedAddress with extracted entities
        """
        if not address or not address.strip():
            return ParsedAddress(
                raw_address=address,
                normalized_address="",
                entities=[]
            )

        # Preprocessing
        normalized = self._preprocess(address)

        # Model inference
        entities = self._extract_entities(normalized)

        # Post-processing
        if self.refiner:
            entities = self.refiner.refine(normalized, entities)

        return ParsedAddress(
            raw_address=address,
            normalized_address=normalized,
            entities=entities
        )

    def parse_with_timing(self, address: str) -> ParseResponse:
        """
        Parse address and return response with timing info.

        Args:
            address: Raw address string

        Returns:
            ParseResponse with result and timing
        """
        start = time.perf_counter()

        try:
            result = self.parse(address)
            elapsed = (time.perf_counter() - start) * 1000

            return ParseResponse(
                success=True,
                result=result,
                inference_time_ms=elapsed
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ParseResponse(
                success=False,
                error=str(e),
                inference_time_ms=elapsed
            )

    def parse_batch(self, addresses: list[str]) -> BatchParseResponse:
        """
        Parse multiple addresses.

        Args:
            addresses: List of raw address strings

        Returns:
            BatchParseResponse with all results
        """
        start = time.perf_counter()

        results = []
        for address in addresses:
            result = self.parse(address)
            results.append(result)

        total_time = (time.perf_counter() - start) * 1000
        avg_time = total_time / len(addresses) if addresses else 0

        return BatchParseResponse(
            success=True,
            results=results,
            total_inference_time_ms=total_time,
            avg_inference_time_ms=avg_time
        )

    def _preprocess(self, text: str) -> str:
        """Preprocess address text."""
        # Handle Hindi text
        if self.transliterator.contains_devanagari(text):
            text = self.transliterator.normalize_mixed_script(text)

        # Normalize
        return self.normalizer.normalize(text)

    def _extract_entities(self, text: str) -> list[AddressEntity]:
        """Extract entities using NER model."""
        if self.model is None or self.tokenizer is None:
            # Rules-only mode
            return self._extract_entities_rules_only(text)

        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
            return_offsets_mapping=True,
            padding=True,
        )

        # Get offset mapping for alignment
        offset_mapping = encoding.pop("offset_mapping")[0].tolist()

        # Move to device
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # Inference
        predictions = self.model.decode(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]  # First (and only) sample

        # Convert to entities
        entities = self._predictions_to_entities(
            text=text,
            predictions=predictions,
            offset_mapping=offset_mapping,
            attention_mask=encoding["attention_mask"][0].tolist(),
        )

        return entities

    def _extract_entities_rules_only(self, text: str) -> list[AddressEntity]:
        """Extract entities using comprehensive rules (no ML)."""
        import re
        entities = []
        text_upper = text.upper()

        # Known localities (multi-word)
        known_localities = [
            "LAJPAT NAGAR", "MALVIYA NAGAR", "HAUZ KHAS", "GREEN PARK",
            "GREATER KAILASH", "DEFENCE COLONY", "SOUTH EXTENSION", "KALKAJI",
            "CIVIL LINES", "MODEL TOWN", "MUKHERJEE NAGAR", "KAMLA NAGAR",
            "PREET VIHAR", "MAYUR VIHAR", "LAKSHMI NAGAR", "GANDHI NAGAR",
            "JANAKPURI", "DWARKA", "UTTAM NAGAR", "TILAK NAGAR", "RAJOURI GARDEN",
            "PUNJABI BAGH", "PASCHIM VIHAR", "KAROL BAGH", "CONNAUGHT PLACE",
            "KAUNWAR SINGH NAGAR", "PALAM COLONY", "RAJ NAGAR", "SADH NAGAR",
            "VIJAY ENCLAVE", "DURGA PARK", "SWARN PARK", "CHANCHAL PARK",
        ]

        for locality in known_localities:
            pos = text_upper.find(locality)
            if pos >= 0:
                entities.append(AddressEntity(
                    label="SUBAREA",
                    value=text[pos:pos + len(locality)],
                    start=pos,
                    end=pos + len(locality),
                    confidence=0.95
                ))

        # Area patterns (directional)
        area_patterns = [
            (r'\bSOUTH\s+DELHI\b', "SOUTH DELHI"),
            (r'\bNORTH\s+DELHI\b', "NORTH DELHI"),
            (r'\bEAST\s+DELHI\b', "EAST DELHI"),
            (r'\bWEST\s+DELHI\b', "WEST DELHI"),
            (r'\bCENTRAL\s+DELHI\b', "CENTRAL DELHI"),
            (r'\bOUTER\s+DELHI\b', "OUTER DELHI"),
        ]

        for pattern, area_name in area_patterns:
            match = re.search(pattern, text_upper)
            if match:
                entities.append(AddressEntity(
                    label="AREA",
                    value=area_name,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.95
                ))

        # House number patterns (order matters - more specific first)
        house_patterns = [
            r'\b(?:FLAT\s*NO\.?\s*)[A-Z]?[-]?\d+[A-Z]?(?:[-/]\d+)*\b',
            r'\b(?:PLOT\s*NO\.?)\s*[A-Z]?\d+[A-Z]?(?:[-/]\d+)*\b',
            r'\b(?:H\.?\s*NO\.?|HOUSE\s*NO\.?|HNO)\s*[A-Z]?\d+[A-Z]?(?:[-/]\d+)*\b',
            r'\b[RW]Z[-\s]?[A-Z]?[-/]?\d+[A-Z]?(?:[-/]\d+)*\b',
        ]

        for pattern in house_patterns:
            match = re.search(pattern, text_upper)
            if match:
                entities.append(AddressEntity(
                    label="HOUSE_NUMBER",
                    value=text[match.start():match.end()],
                    start=match.start(),
                    end=match.end(),
                    confidence=0.90
                ))
                break  # Only first match

        # Floor patterns
        floor_match = re.search(
            r'\b(?:GROUND|FIRST|SECOND|THIRD|FOURTH|1ST|2ND|3RD|4TH|GF|FF|SF|TF)\s*(?:FLOOR|FLR)?\b',
            text_upper
        )
        if floor_match:
            entities.append(AddressEntity(
                label="FLOOR",
                value=text[floor_match.start():floor_match.end()],
                start=floor_match.start(),
                end=floor_match.end(),
                confidence=0.90
            ))

        # Gali patterns
        gali_match = re.search(r'\b(?:GALI|GALLI|LANE)\s*(?:NO\.?)?\s*\d+[A-Z]?\b', text_upper)
        if gali_match:
            entities.append(AddressEntity(
                label="GALI",
                value=text[gali_match.start():gali_match.end()],
                start=gali_match.start(),
                end=gali_match.end(),
                confidence=0.90
            ))

        # Block patterns
        block_match = re.search(r'\b(?:BLOCK|BLK|BL)\s*[A-Z]?[-]?[A-Z0-9]+\b', text_upper)
        if block_match:
            entities.append(AddressEntity(
                label="BLOCK",
                value=text[block_match.start():block_match.end()],
                start=block_match.start(),
                end=block_match.end(),
                confidence=0.90
            ))

        # Sector patterns
        sector_match = re.search(r'\b(?:SECTOR|SEC)\s*\d+[A-Z]?\b', text_upper)
        if sector_match:
            entities.append(AddressEntity(
                label="SECTOR",
                value=text[sector_match.start():sector_match.end()],
                start=sector_match.start(),
                end=sector_match.end(),
                confidence=0.90
            ))

        # Khasra patterns
        khasra_match = re.search(
            r'\b(?:KH\.?\s*(?:NO\.?)?\s*|KHASRA\s*(?:NO\.?)?\s*)[\d/]+(?:[/-]\d+)*\b',
            text_upper
        )
        if khasra_match:
            entities.append(AddressEntity(
                label="KHASRA",
                value=text[khasra_match.start():khasra_match.end()],
                start=khasra_match.start(),
                end=khasra_match.end(),
                confidence=0.90
            ))

        # Pincode (6-digit Delhi codes)
        pincode_match = re.search(r'\b1[1][0]\d{3}\b', text)
        if pincode_match:
            entities.append(AddressEntity(
                label="PINCODE",
                value=pincode_match.group(0),
                start=pincode_match.start(),
                end=pincode_match.end(),
                confidence=1.0
            ))

        # City - always DELHI for Delhi addresses
        if "DELHI" in text_upper:
            # Find standalone DELHI or NEW DELHI
            delhi_match = re.search(r'\bNEW\s+DELHI\b', text_upper)
            if delhi_match:
                entities.append(AddressEntity(
                    label="CITY",
                    value="NEW DELHI",
                    start=delhi_match.start(),
                    end=delhi_match.end(),
                    confidence=0.95
                ))
            else:
                # Find last DELHI
                delhi_positions = [m.start() for m in re.finditer(r'\bDELHI\b', text_upper)]
                if delhi_positions:
                    pos = delhi_positions[-1]
                    entities.append(AddressEntity(
                        label="CITY",
                        value="DELHI",
                        start=pos,
                        end=pos + 5,
                        confidence=0.90
                    ))

        return entities

    def _predictions_to_entities(
        self,
        text: str,
        predictions: list[int],
        offset_mapping: list[tuple[int, int]],
        attention_mask: list[int],
    ) -> list[AddressEntity]:
        """Convert model predictions to entity objects."""
        entities = []
        current_entity = None

        for idx, (pred, offset, mask) in enumerate(zip(predictions, offset_mapping, attention_mask)):
            if mask == 0 or offset == (0, 0):  # Skip padding and special tokens
                continue

            label = ID2LABEL.get(pred, "O")
            start, end = offset

            if label == "O":
                # End current entity if any
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, text))
                    current_entity = None
            elif label.startswith("B-"):
                # Start new entity
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, text))

                entity_type = label[2:]  # Remove "B-" prefix
                current_entity = {
                    "label": entity_type,
                    "start": start,
                    "end": end,
                    "confidence": 0.9,  # Base confidence
                }
            elif label.startswith("I-"):
                # Continue entity
                entity_type = label[2:]
                if current_entity and current_entity["label"] == entity_type:
                    current_entity["end"] = end
                else:
                    # I- without matching B- - treat as new B-
                    if current_entity:
                        entities.append(self._finalize_entity(current_entity, text))
                    current_entity = {
                        "label": entity_type,
                        "start": start,
                        "end": end,
                        "confidence": 0.85,
                    }

        # Don't forget last entity
        if current_entity:
            entities.append(self._finalize_entity(current_entity, text))

        return entities

    def _finalize_entity(self, entity_dict: dict, text: str) -> AddressEntity:
        """Finalize entity with extracted value."""
        value = text[entity_dict["start"]:entity_dict["end"]].strip()

        return AddressEntity(
            label=entity_dict["label"],
            value=value,
            start=entity_dict["start"],
            end=entity_dict["end"],
            confidence=entity_dict["confidence"]
        )


# Convenience function for quick parsing
def parse_address(address: str, model_path: str | None = None) -> ParsedAddress:
    """
    Quick address parsing function.

    Args:
        address: Address to parse
        model_path: Optional path to model (uses rules-only if None)

    Returns:
        ParsedAddress
    """
    if model_path:
        parser = AddressParser.from_pretrained(model_path)
    else:
        parser = AddressParser.rules_only()

    return parser.parse(address)
