"""
Training script for Indian Address NER model.

Trains a mBERT-CRF model for address parsing using the converted
training data from Label Studio annotations.
"""

import json
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from collections import Counter
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from address_parser.models import BertCRFForTokenClassification, ModelConfig
from address_parser.models.config import LABEL2ID, ID2LABEL, BIO_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minority entities that need more sampling weight
MINORITY_ENTITIES = {"PLOT", "COLONY", "SUBAREA", "GALI", "KHASRA", "SECTOR", "BLOCK", "STATE"}


def compute_sample_weights(samples: list[dict], boost_factor: float = 3.0) -> list[float]:
    """
    Compute sampling weights for each sample based on entity presence.

    Samples containing minority entities get higher weight.

    Args:
        samples: List of sample dicts with 'ner_tags' field
        boost_factor: Weight multiplier for samples with minority entities

    Returns:
        List of sampling weights (one per sample)
    """
    # Count entity occurrences across all samples
    entity_counts = Counter()
    for sample in samples:
        entities_in_sample = set()
        for tag in sample.get("ner_tags", []):
            if tag.startswith("B-"):
                entities_in_sample.add(tag[2:])
        for entity in entities_in_sample:
            entity_counts[entity] += 1

    # Compute inverse frequency (rarer = higher weight)
    total_samples = len(samples)
    entity_weights = {}
    for entity, count in entity_counts.items():
        if count > 0:
            # Inverse document frequency: log(N/count) with smoothing
            entity_weights[entity] = max(1.0, total_samples / count)

    # Compute per-sample weight
    sample_weights = []
    for sample in samples:
        entities_in_sample = set()
        for tag in sample.get("ner_tags", []):
            if tag.startswith("B-"):
                entities_in_sample.add(tag[2:])

        # Base weight
        weight = 1.0

        # Boost for minority entities
        for entity in entities_in_sample:
            if entity in MINORITY_ENTITIES:
                weight = max(weight, boost_factor)
                # Additional boost based on rarity
                if entity_weights.get(entity, 1.0) > 5:
                    weight = max(weight, boost_factor * 1.5)

        sample_weights.append(weight)

    # Log weight distribution
    boosted = sum(1 for w in sample_weights if w > 1.0)
    logger.info(f"Sample weighting: {boosted}/{len(samples)} samples boosted")
    logger.info(f"Entity counts: {dict(sorted(entity_counts.items(), key=lambda x: x[1]))}")

    return sample_weights


class AddressNERDataset(Dataset):
    """Dataset for address NER training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 128,
        label2id: dict = None
    ):
        """
        Initialize dataset.

        Args:
            data_path: Path to JSONL file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            label2id: Label to ID mapping
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id or LABEL2ID

        # Load data
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                self.samples.append(sample)

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        # Align labels with tokenized input
        word_ids = encoding.word_ids()
        labels = []
        previous_word_id = None

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)  # Ignore index for special tokens
            elif word_id != previous_word_id:
                # First token of word
                label = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                labels.append(self.label2id.get(label, 0))
            else:
                # Subsequent tokens of word - use I- tag or ignore
                label = ner_tags[word_id] if word_id < len(ner_tags) else "O"
                if label.startswith("B-"):
                    label = "I-" + label[2:]
                labels.append(self.label2id.get(label, 0))

            previous_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    """Evaluate model on validation set."""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            predictions = model.decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Convert to label strings
            for pred, label, mask in zip(predictions, labels, attention_mask):
                pred_labels = []
                true_labels = []

                for p, l, m in zip(pred, label.tolist(), mask.tolist()):
                    if m == 1 and l != -100:  # Valid token with label
                        pred_labels.append(id2label.get(p, "O"))
                        true_labels.append(id2label.get(l, "O"))

                all_predictions.append(pred_labels)
                all_labels.append(true_labels)

    # Calculate metrics
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    report = classification_report(all_labels, all_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
    }


def get_layer_wise_lr_params(model, base_lr, lr_decay, classifier_lr, crf_lr, use_crf):
    """
    Build optimizer parameter groups with layer-wise learning rate decay.

    Deeper (later) layers get higher LR, earlier layers get lower LR.
    Formula: lr_layer = base_lr * decay^(num_layers - layer_index)
    """
    # Try to get encoder layers
    encoder = None
    if hasattr(model.bert, 'encoder'):
        encoder = model.bert.encoder
    elif hasattr(model.bert, 'transformer'):
        encoder = model.bert.transformer

    if encoder is None or lr_decay >= 1.0:
        # Fallback: no layer-wise decay
        params = [
            {"params": list(model.bert.parameters()), "lr": base_lr},
            {"params": list(model.classifier.parameters()), "lr": classifier_lr},
        ]
        if use_crf and model.crf is not None:
            params.append({"params": list(model.crf.parameters()), "lr": crf_lr})
        return params

    # Get layer modules
    if hasattr(encoder, 'layer'):
        layers = list(encoder.layer)
    elif hasattr(encoder, 'layers'):
        layers = list(encoder.layers)
    else:
        # Can't find layers, fallback
        params = [
            {"params": list(model.bert.parameters()), "lr": base_lr},
            {"params": list(model.classifier.parameters()), "lr": classifier_lr},
        ]
        if use_crf and model.crf is not None:
            params.append({"params": list(model.crf.parameters()), "lr": crf_lr})
        return params

    num_layers = len(layers)
    layer_param_ids = set()
    optimizer_params = []

    # Embeddings get lowest LR
    embeddings_lr = base_lr * (lr_decay ** num_layers)
    embeddings_params = []
    if hasattr(model.bert, 'embeddings'):
        embeddings_params = list(model.bert.embeddings.parameters())
    layer_param_ids.update(id(p) for p in embeddings_params)
    if embeddings_params:
        optimizer_params.append({"params": embeddings_params, "lr": embeddings_lr})

    # Each encoder layer: deeper = higher LR
    for layer_idx, layer in enumerate(layers):
        layer_lr = base_lr * (lr_decay ** (num_layers - layer_idx - 1))
        layer_params = list(layer.parameters())
        layer_param_ids.update(id(p) for p in layer_params)
        optimizer_params.append({"params": layer_params, "lr": layer_lr})

    # Any remaining BERT params (pooler, etc.) get base LR
    remaining = [p for p in model.bert.parameters() if id(p) not in layer_param_ids]
    if remaining:
        optimizer_params.append({"params": remaining, "lr": base_lr})

    # Classifier head
    optimizer_params.append({"params": list(model.classifier.parameters()), "lr": classifier_lr})

    # CRF
    if use_crf and model.crf is not None:
        optimizer_params.append({"params": list(model.crf.parameters()), "lr": crf_lr})

    # Log LR summary
    logger.info(f"Layer-wise LR decay={lr_decay}: embeddings={embeddings_lr:.2e}, "
                f"layer_0={base_lr * lr_decay**(num_layers-1):.2e}, "
                f"layer_{num_layers-1}={base_lr:.2e}, "
                f"classifier={classifier_lr:.2e}, crf={crf_lr:.2e}")

    return optimizer_params


def train(
    train_data: str,
    val_data: str,
    output_dir: str,
    model_name: str = "ai4bharat/IndicBERTv2-SS",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    crf_learning_rate: float = 1e-3,
    max_length: int = 128,
    device: str = None,
    use_crf: bool = True,
    early_stopping_patience: int = 5,
    lr_decay: float = 0.95,
    warmup_ratio: float = 0.1,
    test_data: str = None,
    sample_boost: float = 3.0,
):
    """
    Main training function.

    Args:
        train_data: Path to training JSONL
        val_data: Path to validation JSONL
        output_dir: Directory to save model
        model_name: Pretrained model name
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate for BERT
        crf_learning_rate: Learning rate for CRF layer
        max_length: Maximum sequence length
        device: Device to train on
        use_crf: Whether to use CRF layer
        early_stopping_patience: Early stopping patience
        lr_decay: Layer-wise LR decay factor (1.0 = no decay)
        warmup_ratio: Fraction of total steps for LR warmup
        test_data: Optional path to test JSONL for final evaluation
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    train_dataset = AddressNERDataset(train_data, tokenizer, max_length)
    val_dataset = AddressNERDataset(val_data, tokenizer, max_length)

    # Compute sample weights for balanced sampling
    sample_weights = compute_sample_weights(train_dataset.samples, boost_factor=sample_boost)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler instead of shuffle
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Initialize model
    config = ModelConfig(
        model_name=model_name,
        num_labels=len(BIO_LABELS),
        use_crf=use_crf,
        max_length=max_length,
        learning_rate=learning_rate,
        crf_learning_rate=crf_learning_rate,
    )

    model = BertCRFForTokenClassification(config)
    model.to(device)

    # Setup optimizer with layer-wise learning rate decay
    classifier_lr = learning_rate * 10
    optimizer_params = get_layer_wise_lr_params(
        model, learning_rate, lr_decay, classifier_lr, crf_learning_rate, use_crf
    )

    optimizer = AdamW(optimizer_params, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    logger.info(f"Training config: model={model_name}, lr={learning_rate}, "
                f"crf_lr={crf_learning_rate}, lr_decay={lr_decay}, "
                f"warmup_ratio={warmup_ratio}, patience={early_stopping_patience}")
    logger.info(f"Data: {len(train_dataset)} train, {len(val_dataset)} val")
    logger.info(f"Steps: {total_steps} total, {warmup_steps} warmup, "
                f"{len(train_loader)} steps/epoch")

    # Training loop
    best_f1 = 0
    patience_counter = 0

    for epoch in range(epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        logger.info(f"{'='*50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logger.info(f"Training loss: {train_loss:.4f}")

        # Evaluate
        metrics = evaluate(model, val_loader, device, ID2LABEL)
        logger.info(f"Validation - P: {metrics['precision']:.4f}, R: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        logger.info(f"\nPer-entity classification report:\n{metrics['report']}")

        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            patience_counter = 0

            model.save_pretrained(str(output_path))
            tokenizer.save_pretrained(str(output_path))
            logger.info(f"New best model saved! F1: {best_f1:.4f}")

            # Save training info
            with open(output_path / "training_info.json", "w") as f:
                json.dump({
                    "best_f1": best_f1,
                    "best_precision": metrics["precision"],
                    "best_recall": metrics["recall"],
                    "epoch": epoch + 1,
                    "model_name": model_name,
                    "use_crf": use_crf,
                    "learning_rate": learning_rate,
                    "crf_learning_rate": crf_learning_rate,
                    "lr_decay": lr_decay,
                    "warmup_ratio": warmup_ratio,
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                }, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to {output_path}")

    # Test set evaluation
    if test_data:
        logger.info(f"\n{'='*50}")
        logger.info("Evaluating on test set...")
        logger.info(f"{'='*50}")

        # Reload best model for test evaluation
        best_model = BertCRFForTokenClassification.from_pretrained(str(output_path), device=device)
        best_model.to(device)

        test_dataset = AddressNERDataset(test_data, tokenizer, max_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        test_metrics = evaluate(best_model, test_loader, device, ID2LABEL)
        logger.info(f"Test - P: {test_metrics['precision']:.4f}, R: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}")
        logger.info(f"\nTest per-entity classification report:\n{test_metrics['report']}")

        # Save test results
        with open(output_path / "test_results.json", "w") as f:
            json.dump({
                "test_f1": test_metrics["f1"],
                "test_precision": test_metrics["precision"],
                "test_recall": test_metrics["recall"],
                "test_report": test_metrics["report"],
            }, f, indent=2)

    return best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Train address NER model",
        color=True,
        suggest_on_error=True,
    )
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument("--val", required=True, help="Path to validation JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="ai4bharat/IndicBERTv2-SS", help="Pretrained model (default: IndicBERTv2-SS)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--crf-lr", type=float, default=1e-3, help="CRF learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--no-crf", action="store_true", help="Disable CRF layer")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr-decay", type=float, default=0.95, help="Layer-wise LR decay factor (1.0=no decay)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--test", default=None, help="Path to test JSONL for final evaluation")
    parser.add_argument("--sample-boost", type=float, default=3.0, help="Boost factor for minority entity samples")

    args = parser.parse_args()

    train(
        train_data=args.train,
        val_data=args.val,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        crf_learning_rate=args.crf_lr,
        max_length=args.max_length,
        device=args.device,
        use_crf=not args.no_crf,
        early_stopping_patience=args.patience,
        lr_decay=args.lr_decay,
        warmup_ratio=args.warmup_ratio,
        test_data=args.test,
        sample_boost=args.sample_boost,
    )


if __name__ == "__main__":
    main()
