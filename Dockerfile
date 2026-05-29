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

# Install dependencies using standard pip (since HF environments are isolated)
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -e .[dev]

# Create data directories and set permissions for Hugging Face
# Hugging Face runs containers as a non-root user (UID 1000)
RUN mkdir -p data/raw data/processed data/vectorstore/faiss data/vectorstore/chroma data/logs
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Hugging Face strictly requires applications to run on port 7860
ENV PORT=7860
EXPOSE 7860

# Start the FastAPI server on port 7860
CMD uvicorn ragx.api.main:app --host 0.0.0.0 --port 7860
