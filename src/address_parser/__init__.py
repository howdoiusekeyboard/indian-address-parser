"""
Indian Address Parser - Production-grade NER for Indian addresses.

A modern NLP system for parsing unstructured Indian addresses into
structured components using mBERT-CRF architecture with Hindi+English support.
"""

__version__ = "2.1.0"
__author__ = "Kushagra"

from address_parser.pipeline import AddressParser
from address_parser.schemas import (
    AddressEntity,
    ParsedAddress,
    ParseRequest,
    ParseResponse,
)

__all__ = [
    "AddressParser",
    "AddressEntity",
    "ParsedAddress",
    "ParseRequest",
    "ParseResponse",
    "__version__",
]
