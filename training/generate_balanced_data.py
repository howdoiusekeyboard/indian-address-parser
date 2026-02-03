"""
Generate balanced training and test data for address NER.

Focuses on underperforming entities:
- COLONY (0% F1)
- PLOT (0% F1)
- SUBAREA (30% F1)
- GALI (71% F1)

Also ensures test set has statistically significant samples for ALL entity types.
"""

import json
import random
from pathlib import Path
from collections import Counter


# ============================================================================
# EXPANDED DATA SOURCES
# ============================================================================

# Colonies - expanded list with various naming patterns
COLONIES = [
    # -NAGAR suffix
    "RAJ NAGAR", "PREM NAGAR", "SHIV NAGAR", "HARI NAGAR", "KRISHNA NAGAR",
    "GANESH NAGAR", "RAM NAGAR", "LAKSHMI NAGAR", "VIJAY NAGAR", "JAI NAGAR",
    "KAUNWAR SINGH NAGAR", "SADH NAGAR", "BALJIT NAGAR", "TILAK NAGAR",
    "PANDAV NAGAR", "SUNDER NAGAR", "SANT NAGAR", "DEV NAGAR", "GURU NAGAR",
    "MOHAN NAGAR", "RATAN NAGAR", "KISHAN NAGAR", "AMAR NAGAR", "INDRA NAGAR",

    # -VIHAR suffix
    "PREET VIHAR", "ASHOK VIHAR", "BUDH VIHAR", "AMBICA VIHAR", "NIRMAN VIHAR",
    "ANAND VIHAR", "MAYUR VIHAR", "SARITA VIHAR", "EAST OF KAILASH VIHAR",
    "LOK VIHAR", "JANATA VIHAR", "PUSHP VIHAR", "DEEP VIHAR", "RAJ VIHAR",

    # -COLONY suffix
    "DEFENCE COLONY", "PALAM COLONY", "FRIENDS COLONY", "NEW FRIENDS COLONY",
    "BABA HARI DAS COLONY", "TAGORE GARDEN COLONY", "MOTI BAGH COLONY",
    "GULABI BAGH COLONY", "SHADIPUR COLONY", "PANCHSHEEL COLONY",
    "GOLF LINKS COLONY", "JANGPURA EXTENSION COLONY", "LODHI COLONY",

    # -ENCLAVE suffix
    "VIJAY ENCLAVE", "PANCHSHEEL ENCLAVE", "SAINIK ENCLAVE", "SHALIMAR ENCLAVE",
    "MALVIYA ENCLAVE", "GREATER KAILASH ENCLAVE", "NEHRU ENCLAVE",
    "CHITTARANJAN ENCLAVE", "SAKET ENCLAVE", "VASANT ENCLAVE",

    # -PARK suffix
    "DURGA PARK", "SWARN PARK", "CHANCHAL PARK", "GREEN PARK", "DEER PARK",
    "KRISHNA PARK", "SHANTI PARK", "RAJOURI PARK", "TILAK PARK",
    "CHITTARANJAN PARK", "SUBHASH PARK", "NEHRU PARK", "INDIRA PARK",

    # -PURI suffix
    "GOVINDPURI", "KHIRKI PURI", "MADANGIR PURI", "SANGAM PURI",
    "SHIV PURI", "RAM PURI", "HARI PURI", "GANESH PURI",

    # -BAGH suffix
    "KAROL BAGH", "PUNJABI BAGH", "GULABI BAGH", "KIRTI BAGH",
    "SHALIMAR BAGH", "ASHOK BAGH", "PREM BAGH",

    # -EXTENSION suffix
    "SOUTH EXTENSION", "JANGPURA EXTENSION", "LAJPAT NAGAR EXTENSION",
    "SAFDARJUNG EXTENSION", "GREATER KAILASH EXTENSION", "KALKAJI EXTENSION",
]

# Subareas - distinct localities (NOT colonies, NOT areas)
SUBAREAS = [
    # Major localities
    "LAJPAT NAGAR", "MALVIYA NAGAR", "HAUZ KHAS", "SAKET", "MEHRAULI",
    "VASANT KUNJ", "DWARKA", "JANAKPURI", "ROHINI", "PITAMPURA",
    "MODEL TOWN", "CIVIL LINES", "KAMLA NAGAR", "MUKHERJEE NAGAR",
    "CONNAUGHT PLACE", "KAROL BAGH", "PAHARGANJ", "DARYAGANJ",
    "CHANDNI CHOWK", "SADAR BAZAAR", "KASHMERE GATE",

    # Villages/Urban villages
    "MUNDKA", "TIKRI KALAN", "NILOTHI", "NANGLOI", "SULTANPURI",
    "MANGOLPURI", "BAWANA", "NARELA", "ALIPUR", "BAKHTAWARPUR",
    "KHERA KALAN", "KHERA KHURD", "BADLI", "SAMAYPUR BADLI",

    # Commercial/Industrial areas
    "OKHLA", "NEHRU PLACE", "JASOLA", "SARITA VIHAR", "PATPARGANJ",
    "MAYAPURI", "KIRTI NAGAR", "WAZIRPUR", "LAWRENCE ROAD",

    # Residential areas
    "GREATER KAILASH", "KALKAJI", "GOVINDPURI", "SANGAM VIHAR",
    "LADO SARAI", "DERA MANDI", "SATBARI", "CHATTARPUR",
]

# Areas - directional regions
AREAS = [
    "SOUTH DELHI", "NORTH DELHI", "EAST DELHI", "WEST DELHI",
    "CENTRAL DELHI", "SOUTH WEST DELHI", "NORTH WEST DELHI",
    "NORTH EAST DELHI", "SOUTH EAST DELHI", "OUTER DELHI",
    "NEW DELHI", "SHAHDARA",
]

# House number patterns
HOUSE_PATTERNS = [
    "H.NO. {num}", "HOUSE NO. {num}", "HNO {num}", "H NO {num}",
    "H.NO {num}", "HOUSE NO {num}", "H. NO. {num}",
    "{num}", "NO. {num}", "NO {num}",
    "FLAT NO {num}", "FLAT NO. {num}", "FLAT {num}",
    "{letter}-{num}", "{letter}/{num}", "{letter} {num}",
    "RZ-{num}", "WZ-{num}", "RZ {num}", "WZ {num}",
    "RZ-{num}/{num2}", "WZ-{num}/{num2}",
]

# Plot patterns - SEPARATE from house numbers
PLOT_PATTERNS = [
    "PLOT NO {num}", "PLOT NO. {num}", "PLT NO {num}", "PLT NO. {num}",
    "PLOT {num}", "PLT {num}", "P.NO. {num}", "P.NO {num}",
    "PLOT NO {letter}{num}", "PLOT NO. {letter}-{num}",
    "PLOT NO {letter}/{num}", "PLT {letter}{num}",
]

# Khasra patterns - land revenue records
KHASRA_PATTERNS = [
    "KH NO {num}", "KH. NO. {num}", "KHASRA NO {num}", "KHASRA NO. {num}",
    "KH NO {num}/{num2}", "KH. NO. {num}/{num2}",
    "KHASRA NO {num}/{num2}/{num3}", "KH NO {num}/{num2}/{num3}",
    "KH {num}", "KHASRA {num}",
]

# Floor options
FLOOR_OPTIONS = [
    "GROUND FLOOR", "FIRST FLOOR", "SECOND FLOOR", "THIRD FLOOR", "FOURTH FLOOR",
    "GF", "FF", "SF", "TF", "1ST FLOOR", "2ND FLOOR", "3RD FLOOR", "4TH FLOOR",
    "G/F", "F/F", "S/F", "T/F", "BASEMENT", "LOWER GROUND", "UPPER GROUND",
]

# Gali patterns - expanded
GALI_PATTERNS = [
    "GALI NO. {num}", "GALI NO {num}", "GALI {num}",
    "G. NO. {num}", "G.NO. {num}", "G NO {num}",
    "GALLI NO. {num}", "GALLI NO {num}", "GALLI {num}",
    "LANE NO. {num}", "LANE NO {num}", "LANE {num}",
    "STREET NO. {num}", "STREET NO {num}", "ST. NO. {num}",
]

# Block patterns
BLOCK_PATTERNS = [
    "BLOCK {letter}", "BLOCK {letter}-{num}", "BLOCK {letter}{num}",
    "BLK {letter}", "BLK {letter}-{num}", "BL {letter}",
    "BLOCK-{letter}", "B-{letter}", "B {letter}",
]

# Sector patterns
SECTOR_PATTERNS = [
    "SECTOR {num}", "SECTOR-{num}", "SECTOR {num}{letter}",
    "SEC {num}", "SEC-{num}", "SEC. {num}",
]

# Delhi pincodes
PINCODES = [
    "110001", "110002", "110003", "110005", "110006", "110007", "110008",
    "110009", "110010", "110011", "110015", "110016", "110017", "110019",
    "110020", "110021", "110022", "110024", "110025", "110026", "110027",
    "110028", "110029", "110030", "110031", "110032", "110033", "110034",
    "110035", "110036", "110037", "110038", "110039", "110040", "110041",
    "110042", "110043", "110044", "110045", "110046", "110047", "110048",
    "110049", "110051", "110052", "110053", "110054", "110055", "110056",
    "110057", "110058", "110059", "110060", "110061", "110062", "110063",
    "110064", "110065", "110066", "110067", "110068", "110070", "110071",
    "110072", "110073", "110074", "110075", "110076", "110077", "110078",
    "110080", "110081", "110082", "110083", "110084", "110085", "110086",
    "110087", "110088", "110091", "110092", "110093", "110094", "110095",
    "110096",
]

# Cities
CITIES = ["DELHI", "NEW DELHI", "Delhi", "New Delhi"]

# States
STATES = ["DELHI", "Delhi"]


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_house_number():
    pattern = random.choice(HOUSE_PATTERNS)
    num = random.randint(1, 999)
    num2 = random.randint(1, 99)
    letter = random.choice("ABCDEFGHS")
    return pattern.format(num=num, num2=num2, letter=letter)


def generate_plot():
    pattern = random.choice(PLOT_PATTERNS)
    num = random.randint(1, 500)
    letter = random.choice("ABCDEFGH")
    return pattern.format(num=num, letter=letter)


def generate_khasra():
    pattern = random.choice(KHASRA_PATTERNS)
    num = random.randint(1, 999)
    num2 = random.randint(1, 99)
    num3 = random.randint(1, 9)
    return pattern.format(num=num, num2=num2, num3=num3)


def generate_floor():
    return random.choice(FLOOR_OPTIONS)


def generate_gali():
    pattern = random.choice(GALI_PATTERNS)
    num = random.randint(1, 25)
    return pattern.format(num=num)


def generate_block():
    pattern = random.choice(BLOCK_PATTERNS)
    letter = random.choice("ABCDEFGHJK")
    num = random.randint(1, 15)
    return pattern.format(letter=letter, num=num)


def generate_sector():
    pattern = random.choice(SECTOR_PATTERNS)
    num = random.randint(1, 40)
    letter = random.choice("ABCD")
    return pattern.format(num=num, letter=letter)


def tokenize_and_label(text: str, label: str) -> tuple[list[str], list[str]]:
    """Tokenize text and create BIO labels."""
    tokens = text.split()
    if not tokens:
        return [], []
    labels = ["B-" + label] + ["I-" + label] * (len(tokens) - 1)
    return tokens, labels


def generate_sample(
    sample_id: int,
    required_entities: set[str] | None = None,
    entity_weights: dict[str, float] | None = None,
) -> dict:
    """
    Generate a single synthetic training sample.

    Args:
        sample_id: Unique identifier
        required_entities: Set of entities that MUST appear in this sample
        entity_weights: Custom probability weights for each entity type
    """
    required = required_entities or set()
    weights = entity_weights or {}

    component_parts = []

    # HOUSE_NUMBER or PLOT (mutually exclusive usually)
    use_plot = "PLOT" in required or (random.random() < weights.get("PLOT", 0.15))
    use_house = "HOUSE_NUMBER" in required or (not use_plot and random.random() < weights.get("HOUSE_NUMBER", 0.75))

    if use_plot:
        component_parts.append((generate_plot(), "PLOT"))
    elif use_house:
        component_parts.append((generate_house_number(), "HOUSE_NUMBER"))

    # KHASRA (rural/urban village addresses)
    if "KHASRA" in required or random.random() < weights.get("KHASRA", 0.25):
        component_parts.append((generate_khasra(), "KHASRA"))

    # FLOOR
    if "FLOOR" in required or random.random() < weights.get("FLOOR", 0.45):
        component_parts.append((generate_floor(), "FLOOR"))

    # BLOCK
    if "BLOCK" in required or random.random() < weights.get("BLOCK", 0.25):
        component_parts.append((generate_block(), "BLOCK"))

    # SECTOR
    if "SECTOR" in required or random.random() < weights.get("SECTOR", 0.15):
        component_parts.append((generate_sector(), "SECTOR"))

    # GALI
    if "GALI" in required or random.random() < weights.get("GALI", 0.40):
        component_parts.append((generate_gali(), "GALI"))

    # COLONY
    if "COLONY" in required or random.random() < weights.get("COLONY", 0.50):
        component_parts.append((random.choice(COLONIES), "COLONY"))

    # SUBAREA
    if "SUBAREA" in required or random.random() < weights.get("SUBAREA", 0.60):
        component_parts.append((random.choice(SUBAREAS), "SUBAREA"))

    # AREA
    if "AREA" in required or random.random() < weights.get("AREA", 0.35):
        component_parts.append((random.choice(AREAS), "AREA"))

    # CITY (almost always)
    if "CITY" in required or random.random() < weights.get("CITY", 0.95):
        component_parts.append((random.choice(CITIES), "CITY"))

    # STATE (sometimes)
    if "STATE" in required or random.random() < weights.get("STATE", 0.20):
        component_parts.append((random.choice(STATES), "STATE"))

    # PINCODE (almost always)
    if "PINCODE" in required or random.random() < weights.get("PINCODE", 0.90):
        component_parts.append((random.choice(PINCODES), "PINCODE"))

    # Shuffle middle components (keep house/plot first, pincode/city/state last)
    # This creates more natural variation
    if len(component_parts) > 3:
        first_parts = component_parts[:1]  # House/Plot
        last_parts = [p for p in component_parts if p[1] in ("CITY", "STATE", "PINCODE")]
        middle_parts = [p for p in component_parts[1:] if p[1] not in ("CITY", "STATE", "PINCODE")]
        random.shuffle(middle_parts)
        component_parts = first_parts + middle_parts + last_parts

    # Build tokens and labels
    all_tokens = []
    all_labels = []

    for i, (comp_text, label) in enumerate(component_parts):
        if i > 0:
            # Add comma separator
            all_tokens.append(",")
            all_labels.append("O")

        tokens, labels = tokenize_and_label(comp_text, label)
        all_tokens.extend(tokens)
        all_labels.extend(labels)

    text = " ".join(all_tokens)

    return {
        "id": sample_id,
        "text": text,
        "tokens": all_tokens,
        "ner_tags": all_labels,
    }


def generate_balanced_test_set(n_samples: int = 150, min_per_entity: int = 20, seed: int = 42) -> list[dict]:
    """
    Generate a balanced test set with minimum representation for ALL entity types.
    """
    random.seed(seed)

    all_entities = [
        "HOUSE_NUMBER", "PLOT", "FLOOR", "BLOCK", "SECTOR",
        "GALI", "COLONY", "SUBAREA", "AREA", "KHASRA",
        "CITY", "PINCODE", "STATE",
    ]

    samples = []
    entity_counts = Counter()
    sample_id = 50000  # Test set IDs start at 50000

    # Phase 1: Generate samples ensuring minimum coverage for each entity
    for entity in all_entities:
        while entity_counts[entity] < min_per_entity:
            sample = generate_sample(sample_id, required_entities={entity})
            samples.append(sample)
            sample_id += 1

            # Update counts
            for tag in sample["ner_tags"]:
                if tag.startswith("B-"):
                    entity_counts[tag[2:]] += 1

    # Phase 2: Fill remaining slots with random balanced samples
    while len(samples) < n_samples:
        # Find underrepresented entities
        min_count = min(entity_counts.values())
        underrep = [e for e in all_entities if entity_counts[e] < min_count + 5]

        required = set(random.sample(underrep, min(2, len(underrep))))
        sample = generate_sample(sample_id, required_entities=required)
        samples.append(sample)
        sample_id += 1

        for tag in sample["ner_tags"]:
            if tag.startswith("B-"):
                entity_counts[tag[2:]] += 1

    random.shuffle(samples)
    return samples


def generate_focused_training_data(
    n_samples: int = 2000,
    seed: int = 42,
    focus_entities: list[str] | None = None,
) -> list[dict]:
    """
    Generate training data with focus on underperforming entities.
    """
    random.seed(seed)

    # Default focus on entities with poor F1 scores
    focus = focus_entities or ["COLONY", "PLOT", "SUBAREA", "GALI", "KHASRA"]

    # Higher weights for focused entities
    weights = {
        "HOUSE_NUMBER": 0.60,
        "PLOT": 0.35,  # Increased
        "FLOOR": 0.45,
        "BLOCK": 0.30,
        "SECTOR": 0.20,
        "GALI": 0.50,  # Increased
        "COLONY": 0.60,  # Increased
        "SUBAREA": 0.70,  # Increased
        "AREA": 0.35,
        "KHASRA": 0.35,  # Increased
        "CITY": 0.95,
        "STATE": 0.15,
        "PINCODE": 0.90,
    }

    samples = []
    sample_id = 20000  # New training IDs start at 20000

    # 60% focused samples (guaranteed to have focus entities)
    n_focused = int(n_samples * 0.6)
    for i in range(n_focused):
        required = {random.choice(focus)}
        sample = generate_sample(sample_id, required_entities=required, entity_weights=weights)
        samples.append(sample)
        sample_id += 1

    # 40% random samples with elevated weights
    for i in range(n_samples - n_focused):
        sample = generate_sample(sample_id, entity_weights=weights)
        samples.append(sample)
        sample_id += 1

    random.shuffle(samples)
    return samples


def validate_sample(sample: dict) -> bool:
    """Check that tokens and ner_tags are aligned."""
    if len(sample["tokens"]) != len(sample["ner_tags"]):
        return False
    expected_text = " ".join(sample["tokens"])
    return sample["text"] == expected_text


def save_jsonl(samples: list[dict], path: str | Path) -> None:
    """Save samples to JSONL file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def print_entity_distribution(samples: list[dict], title: str) -> dict[str, int]:
    """Print entity distribution for a dataset."""
    counts = Counter()
    for sample in samples:
        for tag in sample["ner_tags"]:
            if tag.startswith("B-"):
                counts[tag[2:]] += 1

    print(f"\n{title} ({len(samples)} samples):")
    print("-" * 40)
    for entity, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {entity:15} {count:5}")

    return dict(counts)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate balanced training/test data")
    parser.add_argument("--n-train", type=int, default=2000, help="New focused training samples")
    parser.add_argument("--n-test", type=int, default=150, help="Balanced test samples")
    parser.add_argument("--min-per-entity", type=int, default=20, help="Minimum test samples per entity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-dir", default="data/processed", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    # Generate balanced test set
    print(f"Generating {args.n_test} balanced test samples (min {args.min_per_entity}/entity)...")
    test_samples = generate_balanced_test_set(
        n_samples=args.n_test,
        min_per_entity=args.min_per_entity,
        seed=args.seed + 5000,
    )

    valid = sum(1 for s in test_samples if validate_sample(s))
    print(f"Valid test samples: {valid}/{len(test_samples)}")

    test_path = data_dir / "test_balanced.jsonl"
    save_jsonl(test_samples, test_path)
    print(f"Saved to {test_path}")

    print_entity_distribution(test_samples, "TEST SET DISTRIBUTION")

    # Generate focused training data
    print(f"\nGenerating {args.n_train} focused training samples...")
    train_samples = generate_focused_training_data(
        n_samples=args.n_train,
        seed=args.seed,
        focus_entities=["COLONY", "PLOT", "SUBAREA", "GALI", "KHASRA"],
    )

    valid = sum(1 for s in train_samples if validate_sample(s))
    print(f"Valid training samples: {valid}/{len(train_samples)}")

    train_path = data_dir / "train_focused.jsonl"
    save_jsonl(train_samples, train_path)
    print(f"Saved to {train_path}")

    print_entity_distribution(train_samples, "FOCUSED TRAINING DISTRIBUTION")

    # Combine with existing training data
    existing_paths = [
        data_dir / "train_combined.jsonl",
        data_dir / "train_augmented.jsonl",
        data_dir / "train.jsonl",
    ]

    combined = list(train_samples)
    for path in existing_paths:
        if path.exists():
            existing = load_jsonl(path)
            combined.extend(existing)
            print(f"\nAdded {len(existing)} samples from {path.name}")
            break

    # Deduplicate by text
    seen_texts = set()
    deduped = []
    for sample in combined:
        if sample["text"] not in seen_texts:
            seen_texts.add(sample["text"])
            deduped.append(sample)

    combined_path = data_dir / "train_final.jsonl"
    save_jsonl(deduped, combined_path)
    print(f"\nSaved {len(deduped)} deduplicated samples to {combined_path}")

    print_entity_distribution(deduped, "FINAL TRAINING DISTRIBUTION")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Test set:     {len(test_samples):,} samples → {test_path}")
    print(f"Training set: {len(deduped):,} samples → {combined_path}")


if __name__ == "__main__":
    main()
