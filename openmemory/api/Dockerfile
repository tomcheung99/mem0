FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

WORKDIR /app

# Install mem0 from local source so all local patches are included
COPY mem0/ /tmp/mem0_src/
RUN pip install /tmp/mem0_src && rm -rf /tmp/mem0_src

# Install remaining API dependencies (mem0ai removed to avoid overwriting local install)
COPY openmemory/api/requirements.txt .
RUN pip install -r requirements.txt

COPY openmemory/api/config.json .
COPY openmemory/api/ .

EXPOSE 8765
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}"]
