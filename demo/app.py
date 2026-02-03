"""Indian Address Parser - Gradio 6.5.1 Demo with FastAPI integration."""

import os
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Path setup: add src to path for imports
_app_dir = Path(__file__).parent
sys.path.insert(0, str(_app_dir / "src" if (_app_dir / "src").exists() else _app_dir.parent / "src"))

from address_parser import AddressParser, ParsedAddress

VERSION = "2.1.0"

ENTITY_COLORS = {
    "HOUSE_NUMBER": "#e74c3c",  # Red
    "PLOT": "#e74c3c",
    "FLOOR": "#1abc9c",         # Teal
    "BLOCK": "#3498db",         # Blue
    "SECTOR": "#27ae60",        # Green
    "GALI": "#f39c12",          # Orange
    "COLONY": "#9b59b6",        # Purple
    "AREA": "#16a085",          # Dark teal
    "SUBAREA": "#f1c40f",       # Yellow
    "KHASRA": "#8e44ad",        # Dark purple
    "PINCODE": "#2980b9",       # Dark blue
    "CITY": "#e67e22",          # Dark orange
    "STATE": "#2ecc71",
}

EXAMPLES = [
    ["PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041"],
    ["H.NO. 123, GALI NO. 5, LAJPAT NAGAR, SOUTH DELHI, 110024"],
    ["FLAT NO A-501, SECTOR 15, DWARKA, NEW DELHI, 110078"],
    ["KHASRA NO 45/2, VILLAGE MUNDKA, OUTER DELHI, 110041"],
    ["S-3/166, GROUND FLOOR, KH NO 98/4, GALI NO-6, SWARN PARK MUNDKA, Delhi, 110041"],
    ["PLOT NO A5 GROUND FLOOR, KHASRA NO 15/20/2 BABA HARI DAS COLONY, TIKARI KALA, DELHI, 110041"],
]


def load_parser() -> AddressParser:
    """Load parser from local path, HuggingFace Hub, or fall back to rules-only."""
    from huggingface_hub import snapshot_download

    local_path = Path(os.getenv("MODEL_PATH", "./models/address_ner_v3"))
    hf_repo = os.getenv("HF_MODEL_REPO", "")

    if local_path.exists() and (local_path / "pytorch_model.bin").exists():
        print(f"[INFO] Loading model from: {local_path}")
        return AddressParser.from_pretrained(str(local_path), device="cpu")

    if hf_repo:
        try:
            print(f"[INFO] Downloading from HuggingFace Hub: {hf_repo}")
            model_path = snapshot_download(repo_id=hf_repo, repo_type="model")
            return AddressParser.from_pretrained(model_path, device="cpu")
        except Exception as e:
            print(f"[WARN] HF Hub download failed: {e}")

    print("[INFO] Using rules-only mode (no ML model)")
    return AddressParser.rules_only()


parser = load_parser()

class ParseRequest(BaseModel):
    address: str
    return_confidence: bool = True


class BatchParseRequest(BaseModel):
    addresses: list[str]
    return_confidence: bool = True


app = FastAPI(
    title="Indian Address Parser API",
    description="Parse unstructured Indian addresses into structured components using mBERT-CRF NER",
    version=VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": parser is not None and hasattr(parser, "model") and parser.model is not None,
        "version": VERSION,
        "gradio_version": gr.__version__,
    }


@app.post("/parse", tags=["Parsing"])
async def api_parse(request: ParseRequest) -> dict[str, Any]:
    """Parse a single address."""
    if parser is None:
        raise HTTPException(status_code=503, detail="Parser not initialized")

    start = time.perf_counter()
    result = parser.parse(request.address)
    inference_ms = (time.perf_counter() - start) * 1000

    return {
        "success": True,
        "result": format_api_result(result, request.return_confidence),
        "inference_time_ms": round(inference_ms, 2),
    }


@app.post("/parse/batch", tags=["Parsing"])
async def api_parse_batch(request: BatchParseRequest) -> dict[str, Any]:
    """Parse multiple addresses (max 100)."""
    if parser is None:
        raise HTTPException(status_code=503, detail="Parser not initialized")
    if len(request.addresses) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 addresses per batch")

    start = time.perf_counter()
    results = [format_api_result(parser.parse(addr), request.return_confidence) for addr in request.addresses]
    total_ms = (time.perf_counter() - start) * 1000

    return {
        "success": True,
        "results": results,
        "total_addresses": len(results),
        "total_time_ms": round(total_ms, 2),
        "avg_time_ms": round(total_ms / len(results), 2) if results else 0,
    }


def format_api_result(result: ParsedAddress, include_confidence: bool) -> dict[str, Any]:
    """Format ParsedAddress for API response."""
    data = result.model_dump()
    if not include_confidence:
        for entity in data["entities"]:
            entity["confidence"] = 1.0
    return data


def format_highlighted(result: ParsedAddress) -> dict:
    """Format for gr.HighlightedText component."""
    entities = [
        {"entity": e.label, "start": e.start, "end": e.end}
        for e in sorted(result.entities, key=lambda x: x.start)
    ]
    return {"text": result.normalized_address, "entities": entities}


def format_structured(result: ParsedAddress) -> dict:
    """Format non-None fields for gr.JSON component."""
    fields = ["house_number", "floor", "block", "sector", "gali", "colony",
              "area", "subarea", "khasra", "pincode", "city", "state"]
    return {f: getattr(result, f) for f in fields if getattr(result, f) is not None}


def format_table(result: ParsedAddress) -> list[list[str]]:
    """Format entities as table rows for gr.Dataframe."""
    return [
        [e.label, e.value, f"{e.confidence:.1%}"]
        for e in sorted(result.entities, key=lambda x: x.start)
    ]


def parse_address_ui(address: str) -> tuple[dict, list[list[str]], dict, str]:
    """Parse address and return formatted outputs for Gradio UI."""
    if not address or not address.strip():
        return {"text": "Please enter an address", "entities": []}, [], {}, "-"

    start = time.perf_counter()
    result = parser.parse(address)
    inference_ms = (time.perf_counter() - start) * 1000

    return (
        format_highlighted(result),
        format_table(result),
        format_structured(result),
        f"{inference_ms:.1f} ms",
    )


def gradio_api_parse(address: str) -> dict:
    """Gradio native API endpoint for single address parsing."""
    return format_api_result(parser.parse(address), include_confidence=True)


def gradio_api_batch(addresses: list[str]) -> list[dict]:
    """Gradio native API endpoint for batch parsing (max 100)."""
    return [format_api_result(parser.parse(addr), include_confidence=True) for addr in addresses[:100]]


CUSTOM_CSS = """
.inference-time { font-family: monospace; font-size: 1.1em; }
"""

with gr.Blocks(title="Indian Address Parser", css=CUSTOM_CSS) as demo:
    gr.Markdown("""
# Indian Address Parser

Parse unstructured Indian addresses into structured components using **mBERT-CRF**.

**Features**: Hindi + English | 13 entity types | Delhi gazetteer | REST API
""")

    with gr.Tabs():
        with gr.Tab("Parse Address", id="parse"):
            with gr.Row():
                with gr.Column(scale=2):
                    address_input = gr.Textbox(
                        label="Enter Address",
                        placeholder="e.g., PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041",
                        lines=3,
                    )
                    with gr.Row():
                        parse_btn = gr.Button("Parse Address", variant="primary", scale=2)
                        clear_btn = gr.ClearButton(scale=1)
                with gr.Column(scale=1):
                    inference_time = gr.Textbox(
                        label="Inference Time",
                        interactive=False,
                        elem_classes=["inference-time"],
                    )

            gr.Markdown("## Results")

            with gr.Row():
                with gr.Column():
                    highlighted_output = gr.HighlightedText(
                        label="Highlighted Entities",
                        color_map=ENTITY_COLORS,
                        show_legend=True,
                        combine_adjacent=True,
                    )
                with gr.Column():
                    entity_table = gr.Dataframe(
                        label="Extracted Entities",
                        headers=["Entity", "Value", "Confidence"],
                        datatype=["str", "str", "str"],
                        row_count=8,
                        interactive=False,
                    )

            with gr.Accordion("Structured Output (JSON)", open=False):
                structured_output = gr.JSON()

            gr.Examples(
                examples=EXAMPLES,
                inputs=[address_input],
                outputs=[highlighted_output, entity_table, structured_output, inference_time],
                fn=parse_address_ui,
                cache_examples=True,
                label="Example Addresses",
            )

            parse_btn.click(
                fn=parse_address_ui,
                inputs=[address_input],
                outputs=[highlighted_output, entity_table, structured_output, inference_time],
            )

            address_input.submit(
                fn=parse_address_ui,
                inputs=[address_input],
                outputs=[highlighted_output, entity_table, structured_output, inference_time],
            )

            clear_btn.add([address_input, highlighted_output, entity_table, structured_output, inference_time])

        with gr.Tab("API", id="api"):
            gr.Markdown(f"""
## REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/parse` | Parse single address |
| `POST` | `/parse/batch` | Parse up to 100 addresses |
| `GET` | `/docs` | Swagger UI |

### Example

```bash
curl -X POST "https://your-domain/parse" \\
  -H "Content-Type: application/json" \\
  -d '{{"address": "B-42, Sector 15, Gurgaon"}}'
```
""")
            gr.api(gradio_api_parse, api_name="parse")
            gr.api(gradio_api_batch, api_name="parse_batch")

        with gr.Tab("About", id="about"):
            gr.Markdown(f"""
## Indian Address Parser

Production-grade NLP for parsing Indian addresses using **IndicBERTv2-SS + CRF**.

### Entity Types

`HOUSE_NUMBER` | `FLOOR` | `BLOCK` | `SECTOR` | `GALI` | `COLONY` | `AREA` | `SUBAREA` | `KHASRA` | `PINCODE` | `CITY` | `STATE`

### Links

- [GitHub](https://github.com/howdoiusekeyboard/indian-address-parser)
- [Model on HuggingFace](https://huggingface.co/x2aqq/indian-address-parser-model)
- [API Docs](/docs)

**Version**: {VERSION} | **Gradio**: {gr.__version__}
""")

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"[INFO] Starting on port {port} | Docs: http://localhost:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
