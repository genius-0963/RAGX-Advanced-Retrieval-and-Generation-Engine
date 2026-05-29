# Dockerfile for Hugging Face Spaces
FROM python:3.11-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files
COPY . .

# Install dependencies (standard install, not editable)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir .

# Create data directories and set permissions for Hugging Face
RUN mkdir -p data/raw data/processed data/vectorstore/faiss data/vectorstore/chroma data/logs
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

ENV PYTHONPATH=/app
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD ["uvicorn", "ragx.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
