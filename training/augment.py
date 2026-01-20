"""
Data augmentation for address NER training.

Expands the training dataset through various augmentation techniques:
- Abbreviation variations (H.NO vs HOUSE NO)
- Case variations
- Component reordering
- Hindi/English variations
- Typo injection
"""

import random
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class AugmentedSample:
    """An augmented training sample."""
    text: str
    tokens: list[str]
    labels: list[str]
    original_id: int
    augmentation_type: str


class AddressAugmenter:
    """
    Augments address training data.

    Techniques:
    1. Abbreviation expansion/contraction
    2. Case variations
    3. Typo injection (realistic OCR/typing errors)
    4. Punctuation variations
    5. Hindi transliteration variants
    """

    # Abbreviation pairs (full form, abbreviated form)
    ABBREVIATIONS = [
        ("HOUSE NO", "H.NO", "H NO", "HNO"),
        ("PLOT NO", "PLT NO", "P.NO"),
        ("KHASRA NO", "KH NO", "KH.NO", "KH"),
        ("GROUND FLOOR", "GF", "GRD FLR", "G.FLOOR"),
        ("FIRST FLOOR", "FF", "1ST FLR", "1ST FLOOR"),
        ("SECOND FLOOR", "SF", "2ND FLR", "2ND FLOOR"),
        ("THIRD FLOOR", "TF", "3RD FLR", "3RD FLOOR"),
        ("BLOCK", "BLK", "BL"),
        ("SECTOR", "SEC"),
        ("GALI NO", "GALI", "LANE NO"),
        ("COLONY", "COL"),
        ("NAGAR", "NGR"),
        ("EXTENSION", "EXTN", "EXT"),
        ("NEW DELHI", "N.DELHI", "N DELHI"),
        ("DELHI", "DL"),
    ]

    # Common typos (correct, typo variants)
    TYPOS = {
        "HOUSE": ["HAUSE", "HOSUE"],
        "FLOOR": ["FLOR", "FOOR", "FLORR"],
        "BLOCK": ["BLCOK", "BOCK"],
        "COLONY": ["COLNY", "COLOGY"],
        "NAGAR": ["NAGER", "NAGR"],
        "KHASRA": ["KHASARA", "KHASRAA"],
        "SECTOR": ["SECTR", "SECOTR"],
        "EXTENSION": ["EXTENTION", "EXTNSION"],
        "DELHI": ["DLEHI", "DEHLI"],
    }

    # Hindi terms with transliterations
    HINDI_VARIANTS = {
        "GALI": ["गली", "GALLI"],
        "MOHALLA": ["मोहल्ला", "MOHLA"],
        "NAGAR": ["नगर", "NAGR"],
        "VIHAR": ["विहार", "BIHAR"],
        "COLONY": ["कॉलोनी", "COLONI"],
    }

    def __init__(
        self,
        abbrev_prob: float = 0.3,
        case_prob: float = 0.2,
        typo_prob: float = 0.1,
        punct_prob: float = 0.2,
        seed: int = 42,
    ):
        """
        Initialize augmenter.

        Args:
            abbrev_prob: Probability of applying abbreviation changes
            case_prob: Probability of case variations
            typo_prob: Probability of typo injection
            punct_prob: Probability of punctuation variations
            seed: Random seed
        """
        self.abbrev_prob = abbrev_prob
        self.case_prob = case_prob
        self.typo_prob = typo_prob
        self.punct_prob = punct_prob

        random.seed(seed)

    def augment_text(self, text: str, n_augments: int = 3) -> list[tuple[str, str]]:
        """
        Generate augmented versions of text.

        Args:
            text: Original address text
            n_augments: Number of augmentations to generate

        Returns:
            List of (augmented_text, augmentation_type) tuples
        """
        augmented = []

        for _ in range(n_augments):
            aug_text = text
            aug_types = []

            # Apply augmentations with probability
            if random.random() < self.abbrev_prob:
                aug_text = self._apply_abbreviation(aug_text)
                aug_types.append("abbrev")

            if random.random() < self.case_prob:
                aug_text = self._apply_case_variation(aug_text)
                aug_types.append("case")

            if random.random() < self.typo_prob:
                aug_text = self._apply_typo(aug_text)
                aug_types.append("typo")

            if random.random() < self.punct_prob:
                aug_text = self._apply_punctuation_variation(aug_text)
                aug_types.append("punct")

            if aug_text != text:  # Only add if actually changed
                augmented.append((aug_text, "+".join(aug_types) if aug_types else "unchanged"))

        return augmented

    def _apply_abbreviation(self, text: str) -> str:
        """Apply abbreviation expansion/contraction."""
        for variants in self.ABBREVIATIONS:
            for i, variant in enumerate(variants):
                if variant in text.upper():
                    # Replace with random other variant
                    other_variants = [v for j, v in enumerate(variants) if j != i]
                    if other_variants:
                        replacement = random.choice(other_variants)
                        # Preserve case pattern
                        pattern = re.compile(re.escape(variant), re.IGNORECASE)
                        text = pattern.sub(replacement, text, count=1)
                        break
        return text

    def _apply_case_variation(self, text: str) -> str:
        """Apply case variations."""
        choice = random.choice(["upper", "lower", "title", "mixed"])

        if choice == "upper":
            return text.upper()
        elif choice == "lower":
            return text.lower()
        elif choice == "title":
            return text.title()
        else:  # mixed
            return "".join(
                c.upper() if random.random() > 0.5 else c.lower()
                for c in text
            )

    def _apply_typo(self, text: str) -> str:
        """Inject realistic typos."""
        for correct, typos in self.TYPOS.items():
            if correct in text.upper():
                if random.random() < 0.5:  # 50% chance per word
                    typo = random.choice(typos)
                    pattern = re.compile(re.escape(correct), re.IGNORECASE)
                    text = pattern.sub(typo, text, count=1)
                    break  # Only one typo per augmentation
        return text

    def _apply_punctuation_variation(self, text: str) -> str:
        """Apply punctuation variations."""
        choice = random.choice(["remove_commas", "add_commas", "remove_periods", "hyphen"])

        if choice == "remove_commas":
            text = text.replace(",", " ")
        elif choice == "add_commas":
            # Add commas after common terms
            for term in ["FLOOR", "BLOCK", "COLONY", "NAGAR"]:
                text = re.sub(rf'\b({term})\b(?!,)', r'\1,', text, flags=re.IGNORECASE)
        elif choice == "remove_periods":
            text = text.replace(".", "")
        elif choice == "hyphen":
            # Convert space to hyphen or vice versa in patterns like "H NO" or "H-NO"
            text = re.sub(r'(\w)\s+(\w)', r'\1-\2', text)

        return text

    def augment_bio_sample(
        self,
        tokens: list[str],
        labels: list[str],
        original_id: int,
        n_augments: int = 2
    ) -> list[AugmentedSample]:
        """
        Augment a BIO-formatted sample.

        This is more complex as we need to maintain token-label alignment.

        Args:
            tokens: Original tokens
            labels: BIO labels
            original_id: Original sample ID
            n_augments: Number of augmentations

        Returns:
            List of augmented samples
        """
        augmented_samples = []
        text = " ".join(tokens)

        for aug_text, aug_type in self.augment_text(text, n_augments):
            # Re-tokenize and try to align labels
            new_tokens = aug_text.split()

            if len(new_tokens) == len(tokens):
                # Same number of tokens - can reuse labels
                augmented_samples.append(AugmentedSample(
                    text=aug_text,
                    tokens=new_tokens,
                    labels=labels.copy(),
                    original_id=original_id,
                    augmentation_type=aug_type
                ))
            # If token count changed, skip this augmentation
            # (more complex alignment would be needed)

        return augmented_samples


def augment_dataset(
    samples: list[dict],
    augmenter: AddressAugmenter,
    target_size: int = 1500
) -> list[dict]:
    """
    Augment entire dataset to target size.

    Args:
        samples: Original samples with 'tokens' and 'ner_tags'
        augmenter: Augmenter instance
        target_size: Target dataset size

    Returns:
        Augmented dataset
    """
    augmented = list(samples)
    original_size = len(samples)
    needed = target_size - original_size

    if needed <= 0:
        return augmented

    # Calculate augmentations per sample
    augs_per_sample = max(1, needed // original_size)

    for sample in samples:
        aug_samples = augmenter.augment_bio_sample(
            tokens=sample["tokens"],
            labels=sample["ner_tags"],
            original_id=sample.get("id", 0),
            n_augments=augs_per_sample
        )

        for aug in aug_samples:
            augmented.append({
                "id": f"{aug.original_id}_aug_{len(augmented)}",
                "text": aug.text,
                "tokens": aug.tokens,
                "ner_tags": aug.labels,
                "augmentation": aug.augmentation_type
            })

            if len(augmented) >= target_size:
                break

        if len(augmented) >= target_size:
            break

    return augmented


if __name__ == "__main__":
    # Demo
    augmenter = AddressAugmenter()

    sample_address = "PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041"

    print("Original:", sample_address)
    print("\nAugmented versions:")

    for aug_text, aug_type in augmenter.augment_text(sample_address, n_augments=5):
        print(f"[{aug_type}]: {aug_text}")
