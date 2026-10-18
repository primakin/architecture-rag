# Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_pipeline.py .
COPY chroma_db ./chroma_db

CMD ["python", "rag_pipeline.py"]