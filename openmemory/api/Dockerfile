FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

WORKDIR /app

# Install Python dependencies
COPY openmemory/api/requirements.txt .
RUN pip install -r requirements.txt

# Install local mem0 source to override PyPI version (includes onnx reranker etc.)
COPY pyproject.toml /tmp/mem0_src/pyproject.toml
COPY mem0/ /tmp/mem0_src/mem0/
RUN pip install /tmp/mem0_src/ && rm -rf /tmp/mem0_src/

# Copy application code
COPY openmemory/api/ .

EXPOSE 8765
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}"]
