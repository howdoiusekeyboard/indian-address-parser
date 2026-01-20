"""Model configuration for address NER."""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for BERT-CRF NER model."""

    # Base model
    model_name: str = "bert-base-multilingual-cased"
    use_crf: bool = True

    # Architecture
    hidden_size: int = 768
    num_labels: int = 31  # O + 15 entity types * 2 (B-/I-)
    hidden_dropout_prob: float = 0.1
    classifier_dropout: float = 0.1

    # CRF settings
    crf_reduction: str = "mean"  # 'mean' or 'sum'

    # Training
    max_length: int = 128
    learning_rate: float = 5e-5
    crf_learning_rate: float = 1e-3  # Higher LR for CRF
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 1

    # Label smoothing
    label_smoothing: float = 0.0

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    # Paths
    output_dir: str = "./models"
    cache_dir: str | None = None

    # ONNX export
    onnx_opset_version: int = 14

    @classmethod
    def from_pretrained_name(cls, name: str) -> ModelConfig:
        """Create config for known pretrained models."""
        configs = {
            "mbert": cls(
                model_name="bert-base-multilingual-cased",
                hidden_size=768,
            ),
            "distilbert": cls(
                model_name="distilbert-base-multilingual-cased",
                hidden_size=768,
            ),
            "xlm-roberta": cls(
                model_name="xlm-roberta-base",
                hidden_size=768,
            ),
            "muril": cls(
                model_name="google/muril-base-cased",
                hidden_size=768,
            ),
        }
        return configs.get(name, cls())


# Entity label definitions (must match schemas.py)
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

# Generate BIO labels
BIO_LABELS = ["O"] + [f"B-{label}" for label in ENTITY_LABELS] + [f"I-{label}" for label in ENTITY_LABELS]
LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(BIO_LABELS)}
NUM_LABELS = len(BIO_LABELS)
