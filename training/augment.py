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

    def _realign_labels(
        self,
        original_tokens: list[str],
        original_labels: list[str],
        new_text: str,
    ) -> tuple[list[str], list[str]] | None:
        """
        Realign BIO labels after text augmentation changes token count.

        Uses character offsets to map new tokens back to original tokens,
        then transfers labels accordingly.

        Returns:
            (new_tokens, new_labels) or None if alignment fails.
        """
        old_text = " ".join(original_tokens)
        new_tokens = new_text.split()

        if not new_tokens:
            return None

        # Build char-offset-to-label map from original tokens
        char_labels = []
        pos = 0
        for token, label in zip(original_tokens, original_labels):
            for _ in token:
                char_labels.append(label)
                pos += 1
            # space separator
            char_labels.append("O")
            pos += 1

        # If augmentation changed length significantly, use a simpler approach:
        # map each new token to the original label at its approximate position
        old_len = len(old_text)
        new_len = len(new_text)

        if old_len == 0 or new_len == 0:
            return None

        new_labels = []
        new_pos = 0
        for token in new_tokens:
            # Map new position to old position proportionally
            ratio = new_pos / max(new_len, 1)
            old_pos = min(int(ratio * old_len), len(char_labels) - 1)

            # Find the label at this position in the original
            if old_pos < len(char_labels):
                mapped_label = char_labels[old_pos]
            else:
                mapped_label = "O"

            new_labels.append(mapped_label)
            new_pos += len(token) + 1  # +1 for space

        # Fix BIO consistency: ensure I- tags follow B- tags of same type
        for i in range(len(new_labels)):
            if new_labels[i].startswith("I-"):
                entity = new_labels[i][2:]
                # Check if previous is B- or I- of same entity
                if i == 0 or (not new_labels[i-1].endswith(entity)):
                    new_labels[i] = "B-" + entity

        return new_tokens, new_labels

    def augment_bio_sample(
        self,
        tokens: list[str],
        labels: list[str],
        original_id: int,
        n_augments: int = 2
    ) -> list[AugmentedSample]:
        """
        Augment a BIO-formatted sample.

        Handles both same-length and different-length augmentations
        by realigning labels using character offset mapping.

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
            new_tokens = aug_text.split()

            if len(new_tokens) == len(tokens):
                # Same number of tokens - can reuse labels directly
                augmented_samples.append(AugmentedSample(
                    text=aug_text,
                    tokens=new_tokens,
                    labels=labels.copy(),
                    original_id=original_id,
                    augmentation_type=aug_type
                ))
            else:
                # Token count changed - realign labels
                result = self._realign_labels(tokens, labels, aug_text)
                if result is not None:
                    new_toks, new_labs = result
                    augmented_samples.append(AugmentedSample(
                        text=aug_text,
                        tokens=new_toks,
                        labels=new_labs,
                        original_id=original_id,
                        augmentation_type=aug_type + "+realigned"
                    ))

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
