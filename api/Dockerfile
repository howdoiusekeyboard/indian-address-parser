# Indian Address Parser - Multi-stage Docker build
# Bakes model from HuggingFace Hub into image

# Stage 1: Build dependencies with uv
FROM python:3.14-slim AS builder

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install --python /opt/venv/bin/python ".[api,demo]"

# Stage 2: Runtime
FROM python:3.14-slim

RUN groupadd -r appuser && useradd -r -g appuser appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/address_ner_v3 \
    DEVICE=cpu \
    PORT=8080 \
    TORCHDYNAMO_DISABLE=1 \
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache

# Download models at build time
ARG HF_MODEL_REPO=x2aqq/indian-address-parser-model
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('${HF_MODEL_REPO}', local_dir='/app/models/address_ner_v3')"
RUN python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('ai4bharat/IndicBERTv2-SS'); AutoTokenizer.from_pretrained('ai4bharat/IndicBERTv2-SS')"

COPY --from=builder /app/src/ ./src/
COPY demo/ ./demo/

RUN mkdir -p /tmp/hf_cache && chown -R appuser:appuser /app /tmp/hf_cache

USER appuser
EXPOSE 8080
CMD ["python", "demo/app.py"]
