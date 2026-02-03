# Indian Address Parser

NLP system for parsing unstructured Indian addresses into structured components. Built with IndicBERTv2-CRF.

[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Extracts 15 entity types from Indian addresses: house number, floor, block, sector, gali, colony, area, subarea, khasra, pincode, city, state, and more. Handles Hindi (Devanagari) and English text. Achieves 80% F1 on test data with <30ms inference.

**Live Demo:** [addressparser.kushagragolash.tech](https://addressparser.kushagragolash.tech)

## Installation

```bash
pip install indian-address-parser
```

From source:

```bash
git clone https://github.com/howdoiusekeyboard/indian-address-parser.git
cd indian-address-parser
pip install -e ".[all]"
```

## Usage

```python
from address_parser import AddressParser

parser = AddressParser.rules_only()  # or .from_pretrained("./models/address_ner_v3")

result = parser.parse(
    "PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, "
    "KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041"
)

print(result.house_number)  # PLOT NO752
print(result.floor)         # FIRST FLOOR
print(result.block)         # BLOCK H-3
print(result.khasra)        # KH NO 24/1/3/2/2/202
print(result.area)          # KAUNWAR SINGH NAGAR
print(result.pincode)       # 110041
```

## API

### REST Endpoints

```bash
# Health check
curl https://addressparser.kushagragolash.tech/health

# Parse single address
curl -X POST https://addressparser.kushagragolash.tech/parse \
  -H "Content-Type: application/json" \
  -d '{"address": "H.NO. 123, DWARKA SECTOR 12, NEW DELHI 110078"}'

# Batch parse
curl -X POST https://addressparser.kushagragolash.tech/batch \
  -H "Content-Type: application/json" \
  -d '{"addresses": ["ADDRESS 1", "ADDRESS 2"]}'
```

### Python

```python
response = parser.parse_with_timing("NEW DELHI 110041")
print(f"Time: {response.inference_time_ms:.1f}ms")

batch = parser.parse_batch(["ADDR 1", "ADDR 2"])
print(f"Avg: {batch.avg_inference_time_ms:.1f}ms")
```

## Entity Types

| Entity | Example |
|--------|---------|
| HOUSE_NUMBER | H.NO. 123, PLOT NO752 |
| FLOOR | FIRST FLOOR, GF |
| BLOCK | BLOCK H-3, BLK A |
| SECTOR | SECTOR 15 |
| GALI | GALI NO. 5 |
| COLONY | BABA HARI DAS COLONY |
| AREA | KAUNWAR SINGH NAGAR |
| SUBAREA | TIKARI KALA |
| KHASRA | KH NO 24/1/3/2 |
| PINCODE | 110041 |
| CITY | NEW DELHI |
| STATE | DELHI |

## Training

```bash
# Prepare data
python training/convert_data.py
python training/generate_synthetic.py --n-train 1000 --n-val 50

# Train
python training/train.py \
  --train data/processed/train_combined.jsonl \
  --val data/processed/val_expanded.jsonl \
  --output models/address_ner_v3 \
  --model ai4bharat/IndicBERTv2-SS \
  --epochs 20 --patience 5 --lr 2e-5
```

## Deployment

### Docker

```bash
docker build -t indian-address-parser -f api/Dockerfile .
docker run -p 8080:8080 indian-address-parser
```

### Cloud Run

```bash
gcloud builds submit --config api/cloudbuild.yaml
```

## Architecture

```
Input -> Preprocessor -> IndicBERTv2-CRF -> Postprocessor -> Output
         (normalize,     (token           (rules,
          Hindi->Latin)   classification)  gazetteer)
```

Components:
- **AddressNormalizer**: Text normalization, abbreviation expansion
- **HindiTransliterator**: Devanagari to Latin conversion
- **BertCRFForTokenClassification**: IndicBERTv2-SS with CRF layer
- **RuleBasedRefiner**: Pattern matching for pincodes, khasra numbers
- **DelhiGazetteer**: 100+ localities with fuzzy matching

## Performance

| Model | Test F1 | Inference |
|-------|---------|-----------|
| IndicBERTv2-CRF | 80.0% | ~25ms |

## Project Structure

```
src/address_parser/
  preprocessing/   # normalization, transliteration
  models/          # BERT-CRF architecture
  postprocessing/  # rules, gazetteer
  pipeline.py      # main orchestration
  schemas.py       # Pydantic models
api/               # FastAPI service
demo/              # Gradio UI
training/          # data prep, training scripts
tests/             # pytest suite
```

## Development

```bash
pip install -e ".[dev]"
pre-commit install

pytest                    # run tests
black src/ tests/         # format
ruff check src/ tests/    # lint
mypy src/                 # type check
```

## License

MIT
