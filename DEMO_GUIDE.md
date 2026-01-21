# Demo Guide: Running Indian Address Parser

## Quick Start

### Option 1: Interactive Gradio Demo (Recommended for Showcasing)

```bash
# Navigate to project
cd indian-address-parser

# Install dependencies with uv
uv sync

# Run the demo
uv run python demo/app.py
```

The Gradio interface will open at `http://localhost:7860`

**Features in the demo:**
- Real-time address parsing
- Color-coded entity highlighting
- Confidence scores for each entity
- Example addresses to try
- Structured JSON output
- Entity type legend

### Option 2: Python API

```python
from src.address_parser import AddressParser

# Load model
parser = AddressParser.from_pretrained("models/address_ner")

# Parse an address
address = "PLOT NO752 FIRST FLOOR, BLOCK H-3, NEW DELHI, 110041"
result = parser.parse(address)

# Access extracted components
print(f"House Number: {result.house_number}")
print(f"Floor: {result.floor}")
print(f"Block: {result.block}")
print(f"Area: {result.area}")
print(f"Pincode: {result.pincode}")

# All entities with confidence
for entity in result.entities:
    print(f"{entity.label}: {entity.value} ({entity.confidence:.0%})")
```

### Option 3: REST API

```bash
# Start the API server
uv run python api/main.py
```

Server runs at `http://localhost:8000`

**Parse single address:**
```bash
curl -X POST "http://localhost:8000/parse" \
  -H "Content-Type: application/json" \
  -d '{"address": "PLOT NO752 FIRST FLOOR, NEW DELHI, 110041"}'
```

**Batch parse:**
```bash
curl -X POST "http://localhost:8000/batch_parse" \
  -H "Content-Type: application/json" \
  -d '{
    "addresses": [
      "PLOT NO752 FIRST FLOOR, NEW DELHI, 110041",
      "H.NO. 123, GALI NO. 5, SOUTH DELHI, 110024"
    ]
  }'
```

**Health check:**
```bash
curl "http://localhost:8000/health"
```

## Testing the Parser

### Run Unit Tests
```bash
uv run pytest tests/ -v
```

### Coverage Report
```bash
uv run pytest tests/ --cov=address_parser --cov-report=html
open htmlcov/index.html
```

## Example Addresses

The demo includes these test addresses:

1. **Plot with multiple components:**
   ```
   PLOT NO752 FIRST FLOOR, BLOCK H-3 KH NO 24/1/3/2/2/202, KAUNWAR SINGH NAGAR NEW DELHI, DELHI, 110041
   ```

2. **Simple flat address:**
   ```
   H.NO. 123, GALI NO. 5, LAJPAT NAGAR, SOUTH DELHI, 110024
   ```

3. **Sector-based (New Delhi suburbs):**
   ```
   FLAT NO A-501, SECTOR 15, DWARKA, NEW DELHI, 110078
   ```

4. **Village address:**
   ```
   KHASRA NO 45/2, VILLAGE MUNDKA, OUTER DELHI, 110041
   ```

5. **Complex ground floor:**
   ```
   S-3/166, GROUND FLOOR, KH NO 98/4, GALI NO-6, SWARN PARK MUNDKA, Delhi, 110041
   ```

6. **With khasra and colony:**
   ```
   PLOT NO A5 GROUND FLOOR, KHASRA NO 15/20/2 BABA HARI DAS COLONY, TIKARI KALA, DELHI, 110041
   ```

## Understanding the Output

### Highlighted View
Shows the address with color-coded entities:
- **Red**: House number/Plot
- **Teal**: Floor
- **Blue**: Block
- **Green**: Sector
- **Yellow**: Gali/Area
- **Purple**: Colony/Khasra

### Entity Table
Lists extracted entities with:
- Entity type
- Extracted value
- Confidence score (0-100%)

### Structured JSON
Clean JSON output with only extracted fields:
```json
{
  "house_number": "PLOT NO752",
  "floor": "FIRST FLOOR",
  "block": "BLOCK H-3",
  "khasra": "KH NO 24/1/3/2/2/202",
  "area": "KAUNWAR SINGH NAGAR",
  "city": "NEW DELHI",
  "pincode": "110041"
}
```

## Troubleshooting

### Port Already in Use
```bash
# Run on different port
PORT=8001 uv run python demo/app.py
# or
PORT=8001 uv run python api/main.py
```

### Model Not Found
The demo automatically falls back to rules-only mode if the model binary isn't present. Full NER requires the model file.

### Dependencies Issues
```bash
# Fresh install
uv sync --refresh
```

## Performance Notes

- **Inference time**: < 30ms per address (with torch.compile)
- **Batch processing**: Process 100 addresses in ~2.5 seconds
- **Memory**: ~800MB for model loading
- **GPU support**: Automatic fallback to CPU if GPU unavailable

## Deployment

### HuggingFace Spaces
```yaml
# See demo/README.md for HuggingFace Spaces config
# File: space.yaml
```

### Docker
```bash
docker build -t address-parser api/
docker run -p 8000:8000 address-parser
```

## Next Steps

- Customize entity types for your use case
- Fine-tune model on your domain data
- Integrate into your application
- See CONTRIBUTING.md for development setup
