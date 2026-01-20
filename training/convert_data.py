"""
Convert Label Studio JSON annotations to BIO format for NER training.

This script converts the existing labels.json from the 2024 project
into a format suitable for training transformer-based NER models.
"""

import json
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
import random

# Label normalization mapping
LABEL_NORMALIZE = {
    "House Number": "HOUSE_NUMBER",
    "house number": "HOUSE_NUMBER",
    "HOUSE_NUMBER": "HOUSE_NUMBER",
    "Floor": "FLOOR",
    "floor": "FLOOR",
    "FLOOR": "FLOOR",
    "Khasra": "KHASRA",
    "khasra": "KHASRA",
    "KHASRA": "KHASRA",
    "Area": "AREA",
    "area": "AREA",
    "AREA": "AREA",
    "Subarea": "SUBAREA",
    "subarea": "SUBAREA",
    "SUBAREA": "SUBAREA",
    "Colony": "COLONY",
    "colony": "COLONY",
    "COLONY": "COLONY",
    "Block": "BLOCK",
    "block": "BLOCK",
    "BLOCK": "BLOCK",
    "Gali": "GALI",
    "gali": "GALI",
    "GALI": "GALI",
    "Sector": "SECTOR",
    "sector": "SECTOR",
    "SECTOR": "SECTOR",
    "Plot": "PLOT",
    "plot": "PLOT",
    "PLOT": "PLOT",
    "Camp": "CAMP",
    "camp": "CAMP",
    "CAMP": "CAMP",
    "Pole": "POLE",
    "pole": "POLE",
    "POLE": "POLE",
    "Pincode": "PINCODE",
    "pincode": "PINCODE",
    "PINCODE": "PINCODE",
    "City": "CITY",
    "city": "CITY",
    "CITY": "CITY",
    "State": "STATE",
    "state": "STATE",
    "STATE": "STATE",
}


@dataclass
class Token:
    """A single token with its label."""
    text: str
    label: str = "O"
    start: int = 0
    end: int = 0


@dataclass
class AnnotatedSample:
    """A single annotated address sample."""
    id: int
    text: str
    tokens: list[Token] = field(default_factory=list)

    def to_bio(self) -> tuple[list[str], list[str]]:
        """Convert to BIO format (tokens, labels)."""
        return [t.text for t in self.tokens], [t.label for t in self.tokens]


def simple_tokenize(text: str) -> list[tuple[str, int, int]]:
    """
    Simple whitespace + punctuation aware tokenizer.
    Returns list of (token, start_offset, end_offset).
    """
    tokens = []
    # Pattern to split on whitespace and keep punctuation separate
    pattern = r'(\s+|[,./\-()])'

    pos = 0
    parts = re.split(pattern, text)

    for part in parts:
        if not part:
            continue
        if part.isspace():
            pos += len(part)
            continue

        # Handle punctuation attached to words
        start = pos
        end = pos + len(part)

        if part.strip():
            tokens.append((part, start, end))

        pos = end

    return tokens


def assign_bio_labels(
    text: str,
    annotations: list[dict],
    tokens: list[tuple[str, int, int]]
) -> list[Token]:
    """
    Assign BIO labels to tokens based on character-level annotations.
    """
    labeled_tokens = []

    # Sort annotations by start position
    sorted_anns = sorted(annotations, key=lambda x: x["start"])

    for token_text, token_start, token_end in tokens:
        token = Token(text=token_text, start=token_start, end=token_end)

        # Find matching annotation
        for ann in sorted_anns:
            ann_start = ann["start"]
            ann_end = ann["end"]
            raw_label = ann["labels"][0] if ann["labels"] else "O"
            label = LABEL_NORMALIZE.get(raw_label, "O")

            # Check if token overlaps with annotation
            if token_start >= ann_start and token_end <= ann_end:
                # Token is fully within annotation
                if token_start == ann_start or labeled_tokens and labeled_tokens[-1].label == "O":
                    # Beginning of entity
                    token.label = f"B-{label}"
                else:
                    # Inside entity
                    prev_label = labeled_tokens[-1].label if labeled_tokens else "O"
                    if prev_label.endswith(label):
                        token.label = f"I-{label}"
                    else:
                        token.label = f"B-{label}"
                break
            elif token_start < ann_end and token_end > ann_start:
                # Partial overlap - assign based on majority
                overlap = min(token_end, ann_end) - max(token_start, ann_start)
                if overlap > (token_end - token_start) / 2:
                    if token_start <= ann_start:
                        token.label = f"B-{label}"
                    else:
                        prev_label = labeled_tokens[-1].label if labeled_tokens else "O"
                        if prev_label.endswith(label):
                            token.label = f"I-{label}"
                        else:
                            token.label = f"B-{label}"
                    break

        labeled_tokens.append(token)

    return labeled_tokens


def convert_label_studio_to_bio(data: list[dict]) -> list[AnnotatedSample]:
    """
    Convert Label Studio format to BIO format.
    """
    samples = []

    for item in data:
        sample_id = item.get("id", 0)
        text = item.get("ADDRESS", "")
        annotations = item.get("label", [])

        if not text or not annotations:
            continue

        # Tokenize
        token_spans = simple_tokenize(text)

        if not token_spans:
            continue

        # Assign labels
        labeled_tokens = assign_bio_labels(text, annotations, token_spans)

        sample = AnnotatedSample(id=sample_id, text=text, tokens=labeled_tokens)
        samples.append(sample)

    return samples


def add_pincode_labels(samples: list[AnnotatedSample]) -> list[AnnotatedSample]:
    """
    Post-process to add PINCODE labels for 6-digit patterns.
    """
    pincode_pattern = re.compile(r'\b[1-9]\d{5}\b')

    for sample in samples:
        for token in sample.tokens:
            if token.label == "O" and pincode_pattern.match(token.text):
                token.label = "B-PINCODE"

    return samples


def add_city_state_labels(samples: list[AnnotatedSample]) -> list[AnnotatedSample]:
    """
    Post-process to add CITY/STATE labels for common patterns.
    """
    cities = {"DELHI", "NEW DELHI", "NOIDA", "GURGAON", "GURUGRAM", "FARIDABAD", "GHAZIABAD"}
    states = {"DELHI", "HARYANA", "UTTAR PRADESH", "UP", "RAJASTHAN"}

    for sample in samples:
        text_upper = sample.text.upper()
        for token in sample.tokens:
            if token.label == "O":
                token_upper = token.text.upper()
                if token_upper in cities or token_upper == "NEW" and "NEW DELHI" in text_upper:
                    token.label = "B-CITY"
                elif token_upper in states:
                    token.label = "B-STATE"

    return samples


def split_data(
    samples: list[AnnotatedSample],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> tuple[list[AnnotatedSample], list[AnnotatedSample], list[AnnotatedSample]]:
    """
    Split data into train/val/test sets.
    """
    random.seed(seed)
    shuffled = samples.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_bio_format(samples: list[AnnotatedSample], output_path: Path) -> None:
    """
    Save samples in CoNLL-style BIO format.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            tokens, labels = sample.to_bio()
            for token, label in zip(tokens, labels):
                f.write(f"{token}\t{label}\n")
            f.write("\n")  # Blank line between samples


def save_jsonl_format(samples: list[AnnotatedSample], output_path: Path) -> None:
    """
    Save samples in JSONL format for HuggingFace datasets.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            tokens, labels = sample.to_bio()
            record = {
                "id": sample.id,
                "text": sample.text,
                "tokens": tokens,
                "ner_tags": labels
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    """Main conversion pipeline."""
    # Paths
    project_root = Path(__file__).parent.parent
    input_path = project_root.parent / "2 broken versions of the same project" / "v2" / "labels.json"
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}")

    # Load Label Studio data
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} annotated samples")

    # Convert to BIO format
    samples = convert_label_studio_to_bio(data)
    print(f"Converted {len(samples)} samples to BIO format")

    # Add automatic labels for pincodes and cities
    samples = add_pincode_labels(samples)
    samples = add_city_state_labels(samples)

    # Split data
    train, val, test = split_data(samples)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")

    # Save in both formats
    for name, split_data_list in [("train", train), ("val", val), ("test", test)]:
        # CoNLL format
        save_bio_format(split_data_list, output_dir / f"{name}.bio")
        # JSONL format
        save_jsonl_format(split_data_list, output_dir / f"{name}.jsonl")
        print(f"Saved {name} split")

    # Save label info
    from address_parser.schemas import BIO_LABELS, LABEL2ID
    label_info = {
        "labels": BIO_LABELS,
        "label2id": LABEL2ID,
        "id2label": {v: k for k, v in LABEL2ID.items()}
    }
    with open(output_dir / "label_info.json", "w") as f:
        json.dump(label_info, f, indent=2)

    print(f"\nConversion complete! Files saved to {output_dir}")

    # Print sample
    print("\n--- Sample Output ---")
    if train:
        tokens, labels = train[0].to_bio()
        print(f"Text: {train[0].text}")
        print("Tokens and labels:")
        for t, l in zip(tokens[:10], labels[:10]):
            print(f"  {t:20} {l}")


if __name__ == "__main__":
    main()
