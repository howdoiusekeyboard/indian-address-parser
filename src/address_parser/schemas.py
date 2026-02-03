"""Pydantic v2 schemas for address parsing I/O."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Entity label definitions
ENTITY_LABELS = [
    "AREA",
    "SUBAREA",
    "HOUSE_NUMBER",
    "SECTOR",
    "GALI",
    "COLONY",
    "BLOCK",
    "CAMP",
    "POLE",
    "KHASRA",
    "FLOOR",
    "PLOT",
    "PINCODE",
    "CITY",
    "STATE",
]

# Type-safe entity label literal
EntityLabel = Literal[
    "AREA", "SUBAREA", "HOUSE_NUMBER", "SECTOR", "GALI",
    "COLONY", "BLOCK", "CAMP", "POLE", "KHASRA",
    "FLOOR", "PLOT", "PINCODE", "CITY", "STATE",
]

# BIO tag generation
BIO_LABELS = ["O"] + [f"B-{label}" for label in ENTITY_LABELS] + [f"I-{label}" for label in ENTITY_LABELS]
LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(BIO_LABELS)}


class AddressEntity(BaseModel):
    """A single extracted entity from an address. Immutable after creation."""

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "label": "HOUSE_NUMBER",
                "value": "PLOT NO752",
                "start": 0,
                "end": 10,
                "confidence": 0.95,
            }
        },
    )

    label: EntityLabel = Field(..., description="Entity type (e.g., HOUSE_NUMBER, AREA)")
    value: str = Field(..., min_length=1, description="Extracted text value")
    start: int = Field(..., ge=0, description="Start character offset in original text")
    end: int = Field(..., ge=0, description="End character offset in original text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")


class ParsedAddress(BaseModel):
    """Complete parsed address with all entities and computed convenience accessors."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "raw_address": "PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041",
                "normalized_address": "PLOT NO752 FIRST FLOOR BLOCK H-3 NEW DELHI 110041",
                "entities": [
                    {"label": "HOUSE_NUMBER", "value": "PLOT NO752", "start": 0, "end": 10, "confidence": 0.95},
                    {"label": "FLOOR", "value": "FIRST FLOOR", "start": 11, "end": 22, "confidence": 0.98},
                ],
                "house_number": "PLOT NO752",
                "floor": "FIRST FLOOR",
            }
        },
    )

    raw_address: str = Field(..., description="Original input address")
    normalized_address: str = Field(..., description="Normalized/cleaned address")
    entities: list[AddressEntity] = Field(default_factory=list, description="Extracted entities")

    def _get_entity(self, *labels: str) -> str | None:
        """Look up first matching entity value by label(s)."""
        for entity in self.entities:
            if entity.label in labels:
                return entity.value
        return None

    @computed_field(description="Extracted house/plot number")
    @property
    def house_number(self) -> str | None:
        return self._get_entity("HOUSE_NUMBER", "PLOT")

    @computed_field(description="Extracted floor")
    @property
    def floor(self) -> str | None:
        return self._get_entity("FLOOR")

    @computed_field(description="Extracted block")
    @property
    def block(self) -> str | None:
        return self._get_entity("BLOCK")

    @computed_field(description="Extracted gali/lane")
    @property
    def gali(self) -> str | None:
        return self._get_entity("GALI")

    @computed_field(description="Extracted colony name")
    @property
    def colony(self) -> str | None:
        return self._get_entity("COLONY")

    @computed_field(description="Extracted area/locality")
    @property
    def area(self) -> str | None:
        return self._get_entity("AREA")

    @computed_field(description="Extracted sub-area")
    @property
    def subarea(self) -> str | None:
        return self._get_entity("SUBAREA")

    @computed_field(description="Extracted sector")
    @property
    def sector(self) -> str | None:
        return self._get_entity("SECTOR")

    @computed_field(description="Extracted khasra number")
    @property
    def khasra(self) -> str | None:
        return self._get_entity("KHASRA")

    @computed_field(description="Extracted PIN code")
    @property
    def pincode(self) -> str | None:
        return self._get_entity("PINCODE")

    @computed_field(description="Extracted city")
    @property
    def city(self) -> str | None:
        return self._get_entity("CITY")

    @computed_field(description="Extracted state")
    @property
    def state(self) -> str | None:
        return self._get_entity("STATE")


class ParseRequest(BaseModel):
    """Request schema for parsing addresses."""

    model_config = ConfigDict(
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "address": "PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041",
                "return_confidence": True,
            }
        },
    )

    address: str = Field(..., min_length=5, max_length=500, description="Address to parse")
    return_confidence: bool = Field(default=True, description="Include confidence scores")


class BatchParseRequest(BaseModel):
    """Request schema for batch parsing."""

    addresses: list[str] = Field(..., min_length=1, max_length=100, description="List of addresses")
    return_confidence: bool = Field(default=True, description="Include confidence scores")


class ParseResponse(BaseModel):
    """Response schema for single address parsing."""

    success: bool = Field(default=True, description="Whether parsing succeeded")
    result: ParsedAddress | None = Field(None, description="Parsed address result")
    error: str | None = Field(None, description="Error message if failed")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")


class BatchParseResponse(BaseModel):
    """Response schema for batch parsing."""

    success: bool = Field(default=True)
    results: list[ParsedAddress] = Field(default_factory=list)
    total_inference_time_ms: float = Field(..., description="Total inference time")
    avg_inference_time_ms: float = Field(..., description="Average per-address time")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    model_loaded: bool = Field(default=False)
    version: str = Field(default="2.1.0")
