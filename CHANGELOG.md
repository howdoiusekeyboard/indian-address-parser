# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-01-28

### Changed
- **Model upgrade**: Replaced `bert-base-multilingual-cased` with `ai4bharat/IndicBERTv2-SS` as default base model
- IndicBERTv2-SS is pretrained on 20.9B tokens of Indian language text (24 Indic languages)
- Added `indicbert` preset to `ModelConfig.from_pretrained_name()`
- Updated training script default to IndicBERTv2-SS

### Fixed
- Corrected performance metrics in README (was 94%+, actual mBERT-CRF baseline: 79.5% F1)

### Training Results
- IndicBERTv2-CRF achieved 64.2% F1 on validation (14 samples)
- Entity-level: PINCODE 100%, CITY 96%, KHASRA 88%, HOUSE_NUMBER 0%
- Model size: ~1.1GB (pytorch_model.bin)
- Note: Lower than mBERT baseline likely due to small val set; needs hyperparameter tuning

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
- 79.5% F1 score on test data (mBERT-CRF)
- <30ms inference time per address
- 15 entity types for Indian address components
- Support for mixed Hindi-English addresses

## [1.0.0] - 2024

### Added
- Initial release from BSES Delhi internship
- Basic spaCy-based NER approach
- Label Studio annotation workflow
