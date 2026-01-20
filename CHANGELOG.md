# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-01-21

### Added
- Production-ready mBERT-CRF model for address NER
- FastAPI REST service with batch processing
- Gradio interactive demo
- Hindi transliteration support
- Delhi locality gazetteer with fuzzy matching
- Rule-based post-processor for entity refinement
- Data augmentation utilities for training
- ONNX export support (Python <3.14)
- Comprehensive test suite (43 tests)

### Changed
- Upgraded to Python 3.14
- Migrated to PyTorch 2.9.1 with torch.compile optimization
- Updated to Transformers 4.57.6
- Migrated Pydantic to v2.12 with ConfigDict
- Updated Gradio to v6.3

### Technical
- 94%+ F1 score on test data
- <30ms inference time per address
- 15 entity types for Indian address components
- Support for mixed Hindi-English addresses

## [1.0.0] - 2024

### Added
- Initial release from BSES Delhi internship
- Basic spaCy-based NER approach
- Label Studio annotation workflow
