# Indian Address Parser Demo

Parse unstructured Indian addresses into structured components using **IndicBERTv2-CRF**.

## Features

- **Multilingual**: Supports Hindi (Devanagari) + English
- **15 Entity Types**: House Number, Floor, Block, Gali, Colony, Area, Khasra, Pincode, etc.
- **~80% F1 score** on held-out test data
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

- **Model**: ai4bharat/IndicBERTv2-SS + CRF layer
- **Training Data**: 600+ annotated Delhi addresses
- **Framework**: PyTorch + HuggingFace Transformers + Pydantic v2
