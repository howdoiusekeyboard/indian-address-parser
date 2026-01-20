"""
Training script for Indian Address NER model.

Trains a mBERT-CRF model for address parsing using the converted
training data from Label Studio annotations.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
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


def train(
    train_data: str,
    val_data: str,
    output_dir: str,
    model_name: str = "bert-base-multilingual-cased",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 5e-5,
    crf_learning_rate: float = 1e-3,
    max_length: int = 128,
    device: str = None,
    use_crf: bool = True,
    early_stopping_patience: int = 3,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
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

    # Setup optimizer with different learning rates
    bert_params = list(model.bert.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer_params = [
        {"params": bert_params, "lr": learning_rate},
        {"params": classifier_params, "lr": learning_rate * 10},
    ]

    if use_crf and model.crf is not None:
        crf_params = list(model.crf.parameters())
        optimizer_params.append({"params": crf_params, "lr": crf_learning_rate})

    optimizer = AdamW(optimizer_params, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

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
        logger.info(f"\n{metrics['report']}")

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
                }, f, indent=2)
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    logger.info(f"\nTraining complete! Best F1: {best_f1:.4f}")
    logger.info(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train address NER model")
    parser.add_argument("--train", required=True, help="Path to training JSONL")
    parser.add_argument("--val", required=True, help="Path to validation JSONL")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--model", default="bert-base-multilingual-cased", help="Pretrained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--crf-lr", type=float, default=1e-3, help="CRF learning rate")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu/mps)")
    parser.add_argument("--no-crf", action="store_true", help="Disable CRF layer")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")

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
    )


if __name__ == "__main__":
    main()
