"""Model architectures for address NER."""

from address_parser.models.bert_crf import BertCRFForTokenClassification
from address_parser.models.config import ModelConfig

__all__ = ["BertCRFForTokenClassification", "ModelConfig"]
