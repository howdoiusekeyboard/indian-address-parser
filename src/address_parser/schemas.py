"""Pydantic schemas for address parsing I/O."""

from pydantic import BaseModel, ConfigDict, Field

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

# BIO tag generation
BIO_LABELS = ["O"] + [f"B-{label}" for label in ENTITY_LABELS] + [f"I-{label}" for label in ENTITY_LABELS]
LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(BIO_LABELS)}


class AddressEntity(BaseModel):
    """A single extracted entity from an address."""

    label: str = Field(..., description="Entity type (e.g., HOUSE_NUMBER, AREA)")
    value: str = Field(..., description="Extracted text value")
    start: int = Field(..., description="Start character offset in original text")
    end: int = Field(..., description="End character offset in original text")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "label": "HOUSE_NUMBER",
                "value": "PLOT NO752",
                "start": 0,
                "end": 10,
                "confidence": 0.95,
            }
        }
    )


class ParsedAddress(BaseModel):
    """Complete parsed address with all entities."""

    raw_address: str = Field(..., description="Original input address")
    normalized_address: str = Field(..., description="Normalized/cleaned address")
    entities: list[AddressEntity] = Field(default_factory=list, description="Extracted entities")

    # Convenience accessors for common fields
    house_number: str | None = Field(None, description="Extracted house/plot number")
    floor: str | None = Field(None, description="Extracted floor")
    block: str | None = Field(None, description="Extracted block")
    gali: str | None = Field(None, description="Extracted gali/lane")
    colony: str | None = Field(None, description="Extracted colony name")
    area: str | None = Field(None, description="Extracted area/locality")
    subarea: str | None = Field(None, description="Extracted sub-area")
    sector: str | None = Field(None, description="Extracted sector")
    khasra: str | None = Field(None, description="Extracted khasra number")
    pincode: str | None = Field(None, description="Extracted PIN code")
    city: str | None = Field(None, description="Extracted city")
    state: str | None = Field(None, description="Extracted state")

    def model_post_init(self, __context) -> None:
        """Populate convenience fields from entities."""
        entity_map = {e.label.upper(): e.value for e in self.entities}

        self.house_number = entity_map.get("HOUSE_NUMBER") or entity_map.get("PLOT")
        self.floor = entity_map.get("FLOOR")
        self.block = entity_map.get("BLOCK")
        self.gali = entity_map.get("GALI")
        self.colony = entity_map.get("COLONY")
        self.area = entity_map.get("AREA")
        self.subarea = entity_map.get("SUBAREA")
        self.sector = entity_map.get("SECTOR")
        self.khasra = entity_map.get("KHASRA")
        self.pincode = entity_map.get("PINCODE")
        self.city = entity_map.get("CITY")
        self.state = entity_map.get("STATE")

    model_config = ConfigDict(
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
        }
    )


class ParseRequest(BaseModel):
    """Request schema for parsing addresses."""

    address: str = Field(..., min_length=5, max_length=500, description="Address to parse")
    return_confidence: bool = Field(default=True, description="Include confidence scores")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "address": "PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041",
                "return_confidence": True,
            }
        }
    )


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
    version: str = Field(default="2.0.0")
