FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV BASE_PDF_PATH="/app/docs/"
ENV CHUNK_SIZE=450
ENV CHUNK_OVERLAP=100
ENV EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
ENV QDRANT_PATH=":memory:"
ENV COLLECTION_NAME="collection_name"
ENV LLM_MODEL_NAME="facebook/bart-large-cnn"
ENV LLM_MODEL_TASK="summarization"

CMD ["python", "rag_pipeline.py"]
