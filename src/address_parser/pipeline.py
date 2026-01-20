"""
Main address parsing pipeline.

Orchestrates preprocessing, model inference, and post-processing
to extract structured entities from Indian addresses.
"""

import time
from pathlib import Path

from transformers import AutoTokenizer

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
        >>> parser = AddressParser.from_pretrained("./models/address_ner")
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
        """Extract entities using rules only (no ML)."""
        entities = []

        if self.refiner:
            patterns = self.refiner.extract_all_patterns(text)
            for label, values in patterns.items():
                for value in values:
                    # Find position in text
                    start = text.upper().find(value.upper())
                    if start >= 0:
                        entities.append(AddressEntity(
                            label=label,
                            value=value,
                            start=start,
                            end=start + len(value),
                            confidence=0.85  # Rule-based confidence
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
