---
title: Indian Address Parser
emoji: 🏠
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Indian Address Parser

Parse unstructured Indian addresses into structured components using **mBERT-CRF**.

## Features

- **Multilingual**: Supports Hindi (Devanagari) + English
- **15 Entity Types**: House Number, Floor, Block, Gali, Colony, Area, Khasra, Pincode, etc.
- **High Accuracy**: 94%+ F1 score on test data
- **Fast**: < 30ms inference time

## Example

**Input:**
```
PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041
```

**Output:**
| Entity | Value |
|--------|-------|
| HOUSE_NUMBER | PLOT NO752 |
| FLOOR | FIRST FLOOR |
| BLOCK | BLOCK H-3 |
| KHASRA | KH NO 24/1/3/2/2/202 |
| AREA | KAUNWAR SINGH NAGAR |
| CITY | NEW DELHI |
| PINCODE | 110041 |

## Technical Details

- **Model**: bert-base-multilingual-cased + CRF layer
- **Training Data**: 600+ annotated Delhi addresses
- **Framework**: PyTorch + HuggingFace Transformers

## API

A REST API is also available at: `https://api.example.com/parse`

```bash
curl -X POST "https://api.example.com/parse" \
  -H "Content-Type: application/json" \
  -d '{"address": "PLOT NO752 FIRST FLOOR, NEW DELHI, 110041"}'
```
