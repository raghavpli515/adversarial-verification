FROM python:3.11-slim

WORKDIR /app

# System deps for chromadb / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/
COPY eval/ ./eval/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir -e .

# Build the vector index at image-build time so container startup is fast
# and doesn't depend on the corpus files being writable at runtime.
RUN python scripts/build_index.py

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
