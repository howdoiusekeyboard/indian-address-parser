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
    components = []
    all_tokens = []
    all_labels = []

    # House number (80% chance)
    if random.random() < 0.8:
        house = generate_house_number()
        tokens, labels = tokenize_and_label(house, "HOUSE_NUMBER")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(house)

    # Floor (50% chance)
    if random.random() < 0.5:
        floor = random.choice(FLOOR_OPTIONS)
        tokens, labels = tokenize_and_label(floor, "FLOOR")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(floor)

    # Block (30% chance)
    if random.random() < 0.3:
        block = generate_block()
        tokens, labels = tokenize_and_label(block, "BLOCK")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(block)

    # Sector (20% chance)
    if random.random() < 0.2:
        sector = generate_sector()
        tokens, labels = tokenize_and_label(sector, "SECTOR")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(sector)

    # Gali (40% chance)
    if random.random() < 0.4:
        gali = generate_gali()
        tokens, labels = tokenize_and_label(gali, "GALI")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(gali)

    # Colony (50% chance)
    if random.random() < 0.5:
        colony = random.choice(COLONIES)
        tokens, labels = tokenize_and_label(colony, "COLONY")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(colony)

    # Subarea/Locality (70% chance)
    if random.random() < 0.7:
        locality = random.choice(LOCALITIES)
        tokens, labels = tokenize_and_label(locality, "SUBAREA")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(locality)

    # Area (40% chance)
    if random.random() < 0.4:
        area = random.choice(AREAS)
        tokens, labels = tokenize_and_label(area, "AREA")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(area)

    # City (always DELHI for Delhi addresses)
    city = random.choice(["DELHI", "NEW DELHI", "Delhi"])
    tokens, labels = tokenize_and_label(city, "CITY")
    all_tokens.extend(tokens)
    all_labels.extend(labels)
    components.append(city)

    # Pincode (90% chance)
    if random.random() < 0.9:
        pincode = random.choice(PINCODES)
        tokens, labels = tokenize_and_label(pincode, "PINCODE")
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        components.append(pincode)

    # Join with commas
    text = ", ".join(components)

    # Re-tokenize the full text properly
    final_tokens = []
    final_labels = []

    for comp_idx, (component, comp_tokens, comp_labels) in enumerate(zip(
        components,
        [tokenize_and_label(c, "O")[0] for c in components],
        [all_labels[sum(len(tokenize_and_label(components[i], "O")[0]) for i in range(j)):
                    sum(len(tokenize_and_label(components[i], "O")[0]) for i in range(j+1))]
         for j in range(len(components))]
    )):
        pass  # Complex re-alignment not needed for simple generation

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


def main():
    # Generate synthetic samples
    print("Generating synthetic training data...")
    synthetic = generate_dataset(n_samples=500)
    print(f"Generated {len(synthetic)} synthetic samples")

    # Load existing training data
    train_path = Path("data/processed/train.jsonl")
    existing = []
    if train_path.exists():
        with open(train_path, "r") as f:
            for line in f:
                existing.append(json.loads(line))
        print(f"Loaded {len(existing)} existing samples")

    # Combine
    combined = existing + synthetic
    print(f"Total: {len(combined)} samples")

    # Save augmented training data
    output_path = Path("data/processed/train_augmented.jsonl")
    with open(output_path, "w") as f:
        for sample in combined:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved augmented dataset to {output_path}")

    # Show some examples
    print("\nExample synthetic samples:")
    for sample in synthetic[:5]:
        print(f"  {sample['text']}")
        print(f"    Tokens: {sample['tokens'][:10]}...")
        print(f"    Labels: {sample['ner_tags'][:10]}...")
        print()


if __name__ == "__main__":
    main()
