"""
Gradio demo for Indian Address Parser.

Interactive web interface for HuggingFace Spaces deployment.
Features:
- Real-time address parsing
- Entity highlighting
- Example addresses
- Confidence scores
"""

import os
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from address_parser import AddressParser, ParsedAddress

# Entity colors for visualization
ENTITY_COLORS = {
    "HOUSE_NUMBER": "#FF6B6B",  # Red
    "PLOT": "#FF6B6B",
    "FLOOR": "#4ECDC4",  # Teal
    "BLOCK": "#45B7D1",  # Blue
    "SECTOR": "#96CEB4",  # Green
    "GALI": "#FFEAA7",  # Yellow
    "COLONY": "#DDA0DD",  # Plum
    "AREA": "#98D8C8",  # Mint
    "SUBAREA": "#F7DC6F",  # Light yellow
    "KHASRA": "#BB8FCE",  # Purple
    "PINCODE": "#85C1E9",  # Light blue
    "CITY": "#F8B500",  # Orange
    "STATE": "#58D68D",  # Light green
}

# Example addresses
EXAMPLES = [
    "PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041",
    "H.NO. 123, GALI NO. 5, LAJPAT NAGAR, SOUTH DELHI, 110024",
    "FLAT NO A-501, SECTOR 15, DWARKA, NEW DELHI, 110078",
    "KHASRA NO 45/2, VILLAGE MUNDKA, OUTER DELHI, 110041",
    "S-3/166, GROUND FLOOR, KH NO 98/4, GALI NO-6, SWARN PARK MUNDKA, Delhi, 110041",
    "PLOT NO A5 GROUND FLOOR, KHASRA NO 15/20/2 BABA HARI DAS COLONY, TIKARI KALA, DELHI, 110041",
]


def load_parser():
    """Load the address parser."""
    model_path = os.getenv("MODEL_PATH", "./models/address_ner")

    if Path(model_path).exists() and (Path(model_path) / "pytorch_model.bin").exists():
        print(f"Loading model from {model_path}")
        return AddressParser.from_pretrained(model_path, device="cpu")
    else:
        print("Model not found, using rules-only mode")
        return AddressParser.rules_only()


# Initialize parser
parser = load_parser()


def create_highlighted_html(result: ParsedAddress) -> str:
    """Create HTML with highlighted entities."""
    if not result.entities:
        return f"<p>{result.normalized_address}</p>"

    # Sort entities by position
    sorted_entities = sorted(result.entities, key=lambda e: e.start)

    html_parts = []
    last_end = 0
    text = result.normalized_address

    for entity in sorted_entities:
        # Add text before entity
        if entity.start > last_end:
            html_parts.append(text[last_end:entity.start])

        # Add highlighted entity
        color = ENTITY_COLORS.get(entity.label, "#CCCCCC")
        html_parts.append(
            f'<span style="background-color: {color}; padding: 2px 6px; '
            f'border-radius: 4px; margin: 0 2px; font-weight: bold;" '
            f'title="{entity.label} ({entity.confidence:.0%})">'
            f'{entity.value}</span>'
        )

        last_end = entity.end

    # Add remaining text
    if last_end < len(text):
        html_parts.append(text[last_end:])

    return "".join(html_parts)


def create_entity_table(result: ParsedAddress) -> list[list[str]]:
    """Create table of extracted entities."""
    if not result.entities:
        return []

    return [
        [entity.label, entity.value, f"{entity.confidence:.0%}"]
        for entity in sorted(result.entities, key=lambda e: e.start)
    ]


def parse_address(address: str) -> tuple[str, list[list[str]], str]:
    """
    Parse address and return results for Gradio interface.

    Returns:
        - Highlighted HTML
        - Entity table
        - Structured output JSON
    """
    if not address or not address.strip():
        return "<p>Please enter an address</p>", [], "{}"

    # Parse
    result = parser.parse(address)

    # Create outputs
    highlighted = create_highlighted_html(result)
    table = create_entity_table(result)

    # Structured output
    structured = {
        "house_number": result.house_number,
        "floor": result.floor,
        "block": result.block,
        "gali": result.gali,
        "colony": result.colony,
        "area": result.area,
        "subarea": result.subarea,
        "sector": result.sector,
        "khasra": result.khasra,
        "pincode": result.pincode,
        "city": result.city,
        "state": result.state,
    }
    # Remove None values
    structured = {k: v for k, v in structured.items() if v}

    import json
    structured_json = json.dumps(structured, indent=2, ensure_ascii=False)

    return highlighted, table, structured_json


# Custom CSS for the demo
CUSTOM_CSS = """
.highlighted-text {
    font-size: 1.1em;
    line-height: 1.8;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 8px;
}
"""

# Create Gradio interface
with gr.Blocks(title="Indian Address Parser") as demo:
    gr.Markdown(
        """
        # Indian Address Parser

        Parse unstructured Indian addresses into structured components using
        **mBERT-CRF** (Multilingual BERT with Conditional Random Field).

        ## Features
        - Supports Hindi + English (Devanagari and Latin scripts)
        - 15 entity types: House Number, Floor, Block, Gali, Colony, Area, Khasra, Pincode, etc.
        - Delhi-specific locality gazetteer for improved accuracy
        - < 30ms inference time

        ---
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            address_input = gr.Textbox(
                label="Enter Address",
                placeholder="e.g., PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041",
                lines=3,
            )
            parse_btn = gr.Button("Parse Address", variant="primary")

            gr.Examples(
                examples=[[ex] for ex in EXAMPLES],
                inputs=[address_input],
                label="Example Addresses",
            )

    gr.Markdown("## Results")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Highlighted Entities")
            highlighted_output = gr.HTML(
                elem_classes=["highlighted-text"]
            )

        with gr.Column(scale=1):
            gr.Markdown("### Extracted Entities")
            entity_table = gr.Dataframe(
                headers=["Entity Type", "Value", "Confidence"],
                datatype=["str", "str", "str"],
                row_count=10,
            )

    with gr.Row():
        gr.Markdown("### Structured Output")
        structured_output = gr.Code(
            language="json",
            label="Structured JSON",
        )

    # Legend
    gr.Markdown("### Entity Legend")
    legend_html = " ".join([
        f'<span style="background-color: {color}; padding: 2px 8px; '
        f'border-radius: 4px; margin: 2px; display: inline-block;">{label}</span>'
        for label, color in ENTITY_COLORS.items()
    ])
    gr.HTML(f"<div style='line-height: 2.5;'>{legend_html}</div>")

    # Footer
    gr.Markdown(
        """
        ---
        **Model**: IndicBERTv2-SS + CRF (ai4bharat/IndicBERTv2-SS + CRF layer)
        | **Training Data**: 600+ annotated Delhi addresses
        | **GitHub**: [indian-address-parser](https://github.com/howdoiusekeyboard/indian-address-parser)
        """
    )

    # Event handlers
    parse_btn.click(
        fn=parse_address,
        inputs=[address_input],
        outputs=[highlighted_output, entity_table, structured_output],
    )

    address_input.submit(
        fn=parse_address,
        inputs=[address_input],
        outputs=[highlighted_output, entity_table, structured_output],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )
