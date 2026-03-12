FROM python:3.12-slim

LABEL org.opencontainers.image.name="mem0/openmemory-mcp"

WORKDIR /app

COPY openmemory/api/requirements.txt .
RUN pip install -r requirements.txt

COPY openmemory/api/config.json .
COPY openmemory/api/ .

EXPOSE 8765
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8765}"]
