"""
BERT-CRF Model for Indian Address NER.

Combines a multilingual BERT encoder with a Conditional Random Field (CRF)
layer for improved sequence labeling performance.
"""

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from address_parser.models.config import ID2LABEL, LABEL2ID, ModelConfig


class CRF(nn.Module):
    """
    Conditional Random Field layer for sequence labeling.

    Implements the forward algorithm for computing log-likelihood
    and Viterbi decoding for inference.
    """

    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        Initialize CRF layer.

        Args:
            num_tags: Number of output tags
            batch_first: If True, input is (batch, seq, features)
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # Transition matrix: transitions[i, j] = score of transitioning from tag i to tag j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

        # Start and end transition scores
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

        self._init_transitions()

    def _init_transitions(self):
        """Initialize transition parameters."""
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.ByteTensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.

        Args:
            emissions: Emission scores from BERT (batch, seq, num_tags)
            tags: Gold standard tags (batch, seq)
            mask: Mask for valid tokens (batch, seq)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            Negative log-likelihood loss
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Compute log-likelihood
        numerator = self._compute_score(emissions, tags, mask)
        denominator = self._compute_normalizer(emissions, mask)
        llh = numerator - denominator

        if reduction == "mean":
            return -llh.mean()
        elif reduction == "sum":
            return -llh.sum()
        else:
            return -llh

    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.ByteTensor | None = None,
    ) -> list[list[int]]:
        """
        Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions: Emission scores (batch, seq, num_tags)
            mask: Mask for valid tokens (batch, seq)

        Returns:
            List of best tag sequences for each sample
        """
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.LongTensor,
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Compute the score of a given tag sequence."""
        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score
        score = self.start_transitions[tags[0]]

        for i in range(seq_length - 1):
            current_tag = tags[i]
            next_tag = tags[i + 1]

            # Emission score
            score += emissions[i, torch.arange(batch_size), current_tag] * mask[i]

            # Transition score
            score += self.transitions[current_tag, next_tag] * mask[i + 1]

        # Last emission score
        last_tag_idx = mask.long().sum(dim=0) - 1
        last_tags = tags.gather(0, last_tag_idx.unsqueeze(0)).squeeze(0)
        score += emissions[last_tag_idx, torch.arange(batch_size), last_tags]

        # End transition score
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Compute log-sum-exp of all possible tag sequences (partition function)."""
        seq_length = emissions.shape[0]

        # Initialize with start transitions
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score and transitions for all combinations
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute next scores
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Log-sum-exp
            next_score = torch.logsumexp(next_score, dim=1)

            # Mask
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # Add end transitions
        score += self.end_transitions

        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.BoolTensor
    ) -> list[list[int]]:
        """Viterbi decoding to find best tag sequence."""
        seq_length, batch_size, num_tags = emissions.shape

        # Initialize
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)

            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Find best previous tag for each current tag
            next_score, indices = next_score.max(dim=1)

            # Apply mask
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # Add end transitions
        score += self.end_transitions

        # Backtrack
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for batch_idx in range(batch_size):
            # Best last tag
            _, best_last_tag = score[batch_idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # Backtrack through history
            for hist in reversed(history[:seq_ends[batch_idx]]):
                best_last_tag = hist[batch_idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


class BertCRFForTokenClassification(nn.Module):
    """
    BERT model with CRF layer for token classification.

    This combines a multilingual BERT encoder with a CRF layer
    for improved sequence labeling on NER tasks.
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize BERT-CRF model.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels

        # Load pretrained BERT
        self.bert = AutoModel.from_pretrained(
            config.model_name,
            cache_dir=config.cache_dir,
        )

        # Dropout
        self.dropout = nn.Dropout(config.classifier_dropout)

        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # CRF layer
        if config.use_crf:
            self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        else:
            self.crf = None

        # Label mappings
        self.id2label = ID2LABEL
        self.label2id = LABEL2ID

        # PyTorch 2.9: Lazy compilation for optimized inference
        self._compiled_forward: nn.Module | None = None

    def _get_compiled_forward(self):
        """Lazy compile forward pass on first inference call."""
        # Skip torch.compile on Windows without MSVC or when explicitly disabled
        # The inductor backend requires a C++ compiler (cl on Windows, gcc/clang on Linux)
        import os
        import sys

        skip_compile = (
            os.environ.get("TORCH_COMPILE_DISABLE", "0") == "1"
            or sys.platform == "win32"  # Skip on Windows to avoid cl requirement
        )

        if self._compiled_forward is None:
            if not skip_compile and hasattr(torch, "compile"):
                try:
                    self._compiled_forward = torch.compile(
                        self.forward,
                        backend="inductor",
                        mode="reduce-overhead",
                        dynamic=True,
                    )
                except Exception:
                    self._compiled_forward = self.forward
            else:
                self._compiled_forward = self.forward
        return self._compiled_forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)
            token_type_ids: Token type IDs (batch, seq)
            labels: Gold standard labels for training (batch, seq)
            return_dict: Return as dict or tuple

        Returns:
            TokenClassifierOutput with loss, logits, hidden states
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        # Classification logits
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.crf is not None:
                # CRF loss - need to handle -100 (ignore_index) labels
                mask = attention_mask.bool() if attention_mask is not None else None
                # Replace -100 with 0 (will be masked out anyway)
                crf_labels = labels.clone()
                crf_labels[crf_labels == -100] = 0
                loss = self.crf(logits, crf_labels, mask=mask, reduction=self.config.crf_reduction)
            else:
                # Standard cross-entropy
                loss_fct = nn.CrossEntropyLoss()
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> list[list[int]]:
        """
        Decode input to tag sequences using compiled forward pass.

        Args:
            input_ids: Input token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)
            token_type_ids: Token type IDs (batch, seq)

        Returns:
            List of predicted tag sequences
        """
        self.eval()
        with torch.no_grad():
            # Use compiled forward for optimized inference (PyTorch 2.9+)
            forward_fn = self._get_compiled_forward()
            outputs = forward_fn(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            logits = outputs.logits

            if self.crf is not None:
                mask = attention_mask.bool() if attention_mask is not None else None
                predictions = self.crf.decode(logits, mask=mask)
            else:
                predictions = logits.argmax(dim=-1).tolist()

        return predictions

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import json
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save config
        config_dict = {
            "model_name": self.config.model_name,
            "num_labels": self.config.num_labels,
            "use_crf": self.config.use_crf,
            "hidden_size": self.config.hidden_size,
            "classifier_dropout": self.config.classifier_dropout,
            "id2label": self.id2label,
            "label2id": self.label2id,
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = "cpu"):
        """Load model from directory."""
        import json

        with open(f"{model_path}/config.json") as f:
            config_dict = json.load(f)

        config = ModelConfig(
            model_name=config_dict["model_name"],
            num_labels=config_dict["num_labels"],
            use_crf=config_dict["use_crf"],
            hidden_size=config_dict["hidden_size"],
            classifier_dropout=config_dict["classifier_dropout"],
        )

        model = cls(config)
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        return model
