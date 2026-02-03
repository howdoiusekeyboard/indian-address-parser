"""
Generate synthetic training data for address NER.

Creates properly labeled samples using:
- Known localities from gazetteer
- Address patterns
- Realistic combinations
"""

import json
import random
from pathlib import Path


# Known localities for training
LOCALITIES = [
    # South Delhi
    "LAJPAT NAGAR", "MALVIYA NAGAR", "HAUZ KHAS", "GREEN PARK",
    "GREATER KAILASH", "DEFENCE COLONY", "SOUTH EXTENSION", "KALKAJI",
    "NEHRU PLACE", "OKHLA", "JASOLA", "SARITA VIHAR", "VASANT KUNJ",

    # North Delhi
    "CIVIL LINES", "MODEL TOWN", "MUKHERJEE NAGAR", "KAMLA NAGAR",
    "ASHOK VIHAR", "SHALIMAR BAGH", "PITAMPURA", "ROHINI",

    # East Delhi
    "PREET VIHAR", "MAYUR VIHAR", "PATPARGANJ", "LAKSHMI NAGAR",
    "GANDHI NAGAR", "DILSHAD GARDEN", "ANAND VIHAR", "SHAHDARA",

    # West Delhi
    "JANAKPURI", "DWARKA", "PALAM", "UTTAM NAGAR", "VIKASPURI",
    "TILAK NAGAR", "RAJOURI GARDEN", "PUNJABI BAGH", "PASCHIM VIHAR",
    "MUNDKA", "NANGLOI", "NAJAFGARH",

    # Central Delhi
    "CONNAUGHT PLACE", "KAROL BAGH", "PAHARGANJ", "DARYAGANJ",
    "RAJENDER NAGAR", "PATEL NAGAR", "KIRTI NAGAR",
]

AREAS = [
    "SOUTH DELHI", "NORTH DELHI", "EAST DELHI", "WEST DELHI",
    "CENTRAL DELHI", "SOUTH WEST DELHI", "NORTH WEST DELHI",
    "OUTER DELHI",
]

COLONIES = [
    "PALAM COLONY", "RAJ NAGAR", "VIJAY ENCLAVE", "SADH NAGAR",
    "DURGA PARK", "SWARN PARK", "CHANCHAL PARK", "KAUNWAR SINGH NAGAR",
    "BABA HARI DAS COLONY", "AMBICA VIHAR", "SHIV PURI", "BUDH VIHAR",
]

HOUSE_PATTERNS = [
    "H.NO. {num}", "HOUSE NO. {num}", "HNO {num}", "H NO {num}",
    "PLOT NO {num}", "PLOT NO. {num}", "{num}",
    "FLAT NO {num}", "FLAT NO. {num}", "FLAT {num}",
    "{letter}-{num}", "{letter}/{num}",
    "RZ-{num}", "WZ-{num}", "RZ {num}", "WZ {num}",
]

FLOOR_OPTIONS = [
    "GROUND FLOOR", "FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR",
    "GF", "FF", "SF", "TF", "1ST FLOOR", "2ND FLOOR", "3RD FLOOR",
]

GALI_PATTERNS = [
    "GALI NO. {num}", "GALI NO {num}", "GALI {num}",
    "LANE NO. {num}", "LANE {num}",
]

BLOCK_PATTERNS = [
    "BLOCK {letter}", "BLOCK {letter}-{num}", "BLK {letter}",
]

SECTOR_PATTERNS = [
    "SECTOR {num}", "SEC {num}", "SECTOR-{num}",
]

PINCODES = [
    "110001", "110002", "110003", "110005", "110006",
    "110007", "110008", "110009", "110010", "110011",
    "110015", "110016", "110017", "110019", "110020",
    "110021", "110022", "110024", "110025", "110026",
    "110027", "110028", "110029", "110030", "110031",
    "110041", "110042", "110043", "110044", "110045",
    "110046", "110047", "110048", "110049", "110051",
    "110052", "110053", "110054", "110055", "110056",
    "110057", "110058", "110059", "110060", "110061",
    "110062", "110063", "110064", "110065", "110066",
    "110067", "110068", "110070", "110071", "110072",
    "110073", "110074", "110075", "110076", "110077",
    "110078", "110080", "110081", "110082", "110083",
    "110084", "110085", "110086", "110087", "110088",
    "110091", "110092", "110093", "110094", "110095",
    "110096",
]


def generate_house_number():
    pattern = random.choice(HOUSE_PATTERNS)
    num = random.randint(1, 999)
    letter = random.choice("ABCDEFGH")
    return pattern.format(num=num, letter=letter)


def generate_gali():
    pattern = random.choice(GALI_PATTERNS)
    num = random.randint(1, 20)
    return pattern.format(num=num)


def generate_block():
    pattern = random.choice(BLOCK_PATTERNS)
    letter = random.choice("ABCDEFGH")
    num = random.randint(1, 10)
    return pattern.format(letter=letter, num=num)


def generate_sector():
    pattern = random.choice(SECTOR_PATTERNS)
    num = random.randint(1, 30)
    return pattern.format(num=num)


def tokenize_and_label(text, label):
    """Tokenize text and create BIO labels."""
    tokens = text.split()
    if not tokens:
        return [], []
    labels = ["B-" + label] + ["I-" + label] * (len(tokens) - 1)
    return tokens, labels


def generate_sample(sample_id):
    """Generate a single synthetic training sample."""
    # Collect (component_text, label) pairs first
    component_parts = []

    # House number (80% chance)
    if random.random() < 0.8:
        component_parts.append((generate_house_number(), "HOUSE_NUMBER"))

    # Floor (50% chance)
    if random.random() < 0.5:
        component_parts.append((random.choice(FLOOR_OPTIONS), "FLOOR"))

    # Block (30% chance)
    if random.random() < 0.3:
        component_parts.append((generate_block(), "BLOCK"))

    # Sector (20% chance)
    if random.random() < 0.2:
        component_parts.append((generate_sector(), "SECTOR"))

    # Gali (40% chance)
    if random.random() < 0.4:
        component_parts.append((generate_gali(), "GALI"))

    # Colony (50% chance)
    if random.random() < 0.5:
        component_parts.append((random.choice(COLONIES), "COLONY"))

    # Subarea/Locality (70% chance)
    if random.random() < 0.7:
        component_parts.append((random.choice(LOCALITIES), "SUBAREA"))

    # Area (40% chance)
    if random.random() < 0.4:
        component_parts.append((random.choice(AREAS), "AREA"))

    # City (always DELHI for Delhi addresses)
    component_parts.append((random.choice(["DELHI", "NEW DELHI", "Delhi"]), "CITY"))

    # Pincode (90% chance)
    if random.random() < 0.9:
        component_parts.append((random.choice(PINCODES), "PINCODE"))

    # Build tokens and labels with comma separators between components
    all_tokens = []
    all_labels = []

    for i, (comp_text, label) in enumerate(component_parts):
        if i > 0:
            # Add comma separator token with O label
            all_tokens.append(",")
            all_labels.append("O")

        tokens, labels = tokenize_and_label(comp_text, label)
        all_tokens.extend(tokens)
        all_labels.extend(labels)

    # Build text to match tokens
    text = " ".join(all_tokens)

    return {
        "id": sample_id,
        "text": text,
        "tokens": all_tokens,
        "ner_tags": all_labels,
    }


def generate_dataset(n_samples=500, seed=42):
    """Generate a synthetic dataset."""
    random.seed(seed)
    samples = []

    for i in range(n_samples):
        sample = generate_sample(10000 + i)
        if sample["tokens"]:  # Only add non-empty samples
            samples.append(sample)

    return samples


def save_jsonl(samples, path):
    """Save samples to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def load_jsonl(path):
    """Load samples from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    return samples


def validate_sample(sample):
    """Check that tokens and ner_tags are aligned."""
    if len(sample["tokens"]) != len(sample["ner_tags"]):
        return False
    # Check text roughly matches tokens
    expected_text = " ".join(sample["tokens"])
    return sample["text"] == expected_text


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data",
        color=True,
        suggest_on_error=True,
    )
    parser.add_argument("--n-train", type=int, default=1000, help="Number of synthetic training samples")
    parser.add_argument("--n-val", type=int, default=50, help="Number of synthetic validation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Generate synthetic training samples
    print(f"Generating {args.n_train} synthetic training samples...")
    synthetic_train = generate_dataset(n_samples=args.n_train, seed=args.seed)
    print(f"Generated {len(synthetic_train)} synthetic training samples")

    # Validate
    valid = sum(1 for s in synthetic_train if validate_sample(s))
    print(f"Valid samples: {valid}/{len(synthetic_train)}")

    # Save synthetic training data
    synth_train_path = data_dir / "synthetic_train.jsonl"
    save_jsonl(synthetic_train, synth_train_path)
    print(f"Saved synthetic training data to {synth_train_path}")

    # Generate synthetic validation samples (different seed)
    print(f"\nGenerating {args.n_val} synthetic validation samples...")
    synthetic_val = generate_dataset(n_samples=args.n_val, seed=args.seed + 1000)
    print(f"Generated {len(synthetic_val)} synthetic validation samples")

    synth_val_path = data_dir / "synthetic_val.jsonl"
    save_jsonl(synthetic_val, synth_val_path)
    print(f"Saved synthetic validation data to {synth_val_path}")

    # Load existing data and create combined datasets
    train_path = data_dir / "train.jsonl"
    train_aug_path = data_dir / "train_augmented.jsonl"
    val_path = data_dir / "val.jsonl"

    # Combined training: original + augmented + synthetic
    combined_train = []
    if train_aug_path.exists():
        augmented = load_jsonl(train_aug_path)
        combined_train.extend(augmented)
        print(f"\nLoaded {len(augmented)} augmented samples (includes original)")
    elif train_path.exists():
        original = load_jsonl(train_path)
        combined_train.extend(original)
        print(f"\nLoaded {len(original)} original samples")

    combined_train.extend(synthetic_train)
    print(f"Combined training set: {len(combined_train)} samples")

    combined_train_path = data_dir / "train_combined.jsonl"
    save_jsonl(combined_train, combined_train_path)
    print(f"Saved combined training data to {combined_train_path}")

    # Expanded validation: original val + synthetic val
    combined_val = []
    if val_path.exists():
        original_val = load_jsonl(val_path)
        combined_val.extend(original_val)
        print(f"\nLoaded {len(original_val)} original validation samples")

    combined_val.extend(synthetic_val)
    print(f"Combined validation set: {len(combined_val)} samples")

    val_expanded_path = data_dir / "val_expanded.jsonl"
    save_jsonl(combined_val, val_expanded_path)
    print(f"Saved expanded validation data to {val_expanded_path}")

    # Show examples
    print("\nExample synthetic samples:")
    for sample in synthetic_train[:3]:
        print(f"  Text:   {sample['text']}")
        print(f"  Tokens: {sample['tokens'][:10]}...")
        print(f"  Labels: {sample['ner_tags'][:10]}...")
        aligned = len(sample['tokens']) == len(sample['ner_tags'])
        print(f"  Aligned: {aligned} ({len(sample['tokens'])} tokens, {len(sample['ner_tags'])} labels)")
        print()


if __name__ == "__main__":
    main()
