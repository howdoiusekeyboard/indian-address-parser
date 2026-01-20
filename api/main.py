"""
FastAPI service for Indian Address Parser.

Production-ready REST API for address parsing with:
- Single and batch parsing endpoints
- Swagger documentation
- Health checks
- CORS support
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from address_parser import (
    AddressParser,
    ParsedAddress,
    ParseRequest,
    ParseResponse,
)
from address_parser.schemas import (
    BatchParseRequest,
    BatchParseResponse,
    HealthResponse,
)

# Global parser instance
parser: AddressParser | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize parser on startup."""
    global parser

    model_path = os.getenv("MODEL_PATH", "./models/address_ner")

    # Check if model exists
    if Path(model_path).exists() and (Path(model_path) / "pytorch_model.bin").exists():
        print(f"Loading model from {model_path}")
        parser = AddressParser.from_pretrained(
            model_path,
            device=os.getenv("DEVICE", "cpu"),
            use_rules=True,
            use_gazetteer=True,
        )
        print("Model loaded successfully!")
    else:
        print(f"Model not found at {model_path}, using rules-only mode")
        parser = AddressParser.rules_only(use_gazetteer=True)

    yield

    # Cleanup
    parser = None


# Create FastAPI app
app = FastAPI(
    title="Indian Address Parser API",
    description="""
    Production-grade API for parsing unstructured Indian addresses into structured components.

    ## Features
    - **mBERT-CRF Model**: Multilingual BERT with CRF layer for high-accuracy NER
    - **Hindi + English**: Supports both Devanagari and Latin scripts
    - **Rule-based Enhancement**: Post-processing with Delhi locality gazetteer
    - **Fast Inference**: < 30ms per address with ONNX optimization

    ## Entity Types
    - HOUSE_NUMBER, PLOT, FLOOR, BLOCK, SECTOR
    - GALI, COLONY, AREA, SUBAREA
    - KHASRA, PINCODE, CITY, STATE

    ## Example
    ```json
    POST /parse
    {"address": "PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041"}
    ```
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


# Health check endpoint
@app.get("/", response_model=HealthResponse, tags=["Health"])
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns service status and model availability.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=parser is not None and parser.model is not None,
        version="2.0.0"
    )


# Parse single address
@app.post("/parse", response_model=ParseResponse, tags=["Parsing"])
async def parse_address(request: ParseRequest):
    """
    Parse a single address.

    Extracts structured entities from an unstructured Indian address string.

    **Example Request:**
    ```json
    {
        "address": "PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041",
        "return_confidence": true
    }
    ```

    **Example Response:**
    ```json
    {
        "success": true,
        "result": {
            "raw_address": "PLOT NO752 FIRST FLOOR...",
            "normalized_address": "PLOT NO752 FIRST FLOOR...",
            "entities": [
                {"label": "HOUSE_NUMBER", "value": "PLOT NO752", "start": 0, "end": 10, "confidence": 0.95},
                {"label": "FLOOR", "value": "FIRST FLOOR", "start": 11, "end": 22, "confidence": 0.98}
            ],
            "house_number": "PLOT NO752",
            "floor": "FIRST FLOOR"
        },
        "inference_time_ms": 25.5
    }
    ```
    """
    if parser is None:
        raise HTTPException(status_code=503, detail="Parser not initialized")

    try:
        response = parser.parse_with_timing(request.address)

        # Optionally remove confidence scores
        if not request.return_confidence and response.result:
            for entity in response.result.entities:
                entity.confidence = 1.0

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Batch parse endpoint
@app.post("/parse/batch", response_model=BatchParseResponse, tags=["Parsing"])
async def parse_batch(request: BatchParseRequest):
    """
    Parse multiple addresses in a single request.

    **Limits:**
    - Maximum 100 addresses per request
    - Recommended batch size: 10-50 for optimal performance

    **Example Request:**
    ```json
    {
        "addresses": [
            "PLOT NO752 FIRST FLOOR, NEW DELHI, 110041",
            "H.NO. 123, GALI NO. 5, LAJPAT NAGAR, DELHI"
        ],
        "return_confidence": true
    }
    ```
    """
    if parser is None:
        raise HTTPException(status_code=503, detail="Parser not initialized")

    if len(request.addresses) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 addresses per batch")

    try:
        response = parser.parse_batch(request.addresses)

        if not request.return_confidence:
            for result in response.results:
                for entity in result.entities:
                    entity.confidence = 1.0

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Simple GET endpoint for testing
@app.get("/parse/{address:path}", response_model=ParsedAddress, tags=["Parsing"])
async def parse_address_get(address: str):
    """
    Parse address via GET request (for testing).

    Note: Use POST /parse for production - this endpoint is for quick testing only.
    """
    if parser is None:
        raise HTTPException(status_code=503, detail="Parser not initialized")

    try:
        return parser.parse(address)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
