# RAGX — Advanced Retrieval and Generation Engine

<p align="center">
  <strong>A production-grade RAG platform with multi-format ingestion, hybrid retrieval, multi-LLM generation, evaluation, and monitoring.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/langchain-1.3+-green.svg" alt="LangChain">
  <img src="https://img.shields.io/badge/fastapi-latest-teal.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
</p>

---

## Architecture

```
                  User Query
                       │
                       ▼
                API Gateway (FastAPI)
                       │
                       ▼
                Query Processor
                       │
                       ▼
                 Retriever
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
      BM25                    Vector Search
         │                           │
         └────── Hybrid Search ──────┘
                       │
                       ▼
                   Re-ranker
                       │
                       ▼
              Context Builder
                       │
                       ▼
                     LLM
                       │
                       ▼
          Answer + Sources + Score
                       │
                       ▼
                 Monitoring
```

## Features

### Phase 1: Document Ingestion
- **Multi-format loading**: PDF, DOCX, TXT, CSV, Markdown, Web pages
- **Text cleaning**: Header/footer removal, deduplication, Unicode normalization
- **Metadata generation**: document_id, source, upload_time, page_number, section
- **Batch processing**: Concurrent ingestion with progress tracking

### Phase 2: Embedding Generation
- **Intelligent chunking**: Recursive text splitting (500 tokens, 100 overlap) + semantic chunking
- **Multiple embedding models**: OpenAI, BGE, Sentence Transformers
- **Vector stores**: ChromaDB (default) and FAISS
- **Incremental updates**: Only process new/modified documents

### Phase 3: Retrieval Engine
- **Search strategies**: Similarity, BM25, Hybrid (Reciprocal Rank Fusion)
- **Advanced retrieval**: Multi-query, parent-child, context compression
- **Re-ranking**: Cross-encoder (ms-marco-MiniLM) + Cohere Rerank API
- **Evaluation metrics**: Precision@K, Recall@K, MRR, NDCG@K

### Phase 4: LLM Integration
- **Multi-provider**: GPT-4o, Claude, Gemini, Llama (Ollama)
- **Anti-hallucination**: Grounded prompts with citation enforcement
- **Conversation memory**: Session-based sliding window
- **Source attribution**: Automatic citation extraction and formatting

### Phase 5: Production Deployment
- **FastAPI backend**: RESTful API with Swagger docs
- **Evaluation**: RAGAS + DeepEval metrics
- **Monitoring**: Prometheus metrics + Grafana dashboards
- **Deployment**: Docker, Docker Compose, Kubernetes

---

## Quick Start

### 1. Clone and Install

```bash
# Clone the repository
cd "RAGX – Advanced Retrieval and Generation Engine"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -e ".[dev]"
```

### 2. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
# At minimum, set one LLM provider key (OpenAI, Anthropic, Google, or use Ollama)
```

### 3. Run the API Server

```bash
# Development mode (with auto-reload)
make run

# Or directly:
uvicorn ragx.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API Docs

Navigate to **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest/file` | Upload and process a document |
| `POST` | `/api/v1/ingest/url` | Ingest from a web URL |
| `POST` | `/api/v1/ingest/batch` | Batch upload multiple documents |
| `GET` | `/api/v1/documents` | List ingested documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document |
| `POST` | `/api/v1/query` | Query with retrieval + generation |
| `POST` | `/api/v1/chat` | Conversational chat with memory |
| `GET` | `/api/v1/query/history` | Query history |
| `GET` | `/api/v1/admin/stats` | System statistics |
| `POST` | `/api/v1/admin/reindex` | Trigger re-indexing |
| `GET` | `/api/v1/admin/config` | Current configuration |
| `POST` | `/api/v1/feedback` | Submit feedback |
| `GET` | `/api/v1/feedback/report` | Feedback metrics |
| `GET` | `/health` | Health check |
| `GET` | `/metrics` | Prometheus metrics |

---

## Usage Examples

### Python SDK

```python
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.embeddings.pipeline import EmbeddingPipeline
from ragx.retrieval.engine import RetrievalEngine
from ragx.generation.pipeline import GenerationPipeline

# 1. Ingest documents
ingestion = IngestionPipeline()
docs = ingestion.ingest_file("data/raw/document.pdf")
docs += ingestion.ingest_url("https://example.com/article")

# 2. Embed and store
embeddings = EmbeddingPipeline()
embeddings.process(docs)

# 3. Retrieve
retrieval = RetrievalEngine(vectorstore=embeddings.get_vectorstore())

# 4. Generate answers
generation = GenerationPipeline(retrieval_engine=retrieval)
result = generation.query("What are the key findings?")

print(result["answer"])
print(result["sources"])
print(f"Confidence: {result['confidence_score']}")
```

### cURL

```bash
# Ingest a file
curl -X POST http://localhost:8000/api/v1/ingest/file \
  -F "file=@document.pdf"

# Query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key findings?", "top_k": 5}'

# Chat (with memory)
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me more about that", "session_id": "my-session"}'
```

---

## Docker Deployment

```bash
# Build and start all services (API + Prometheus + Grafana)
make docker-build
make docker-up

# Services:
# - RAGX API:    http://localhost:8000
# - Prometheus:  http://localhost:9090
# - Grafana:     http://localhost:3000 (admin/ragx_admin)
```

---

## Project Structure

```
ragx/
├── config/          # Settings & logging
├── ingestion/       # Phase 1: Document loaders & preprocessing
│   └── loaders/     # PDF, DOCX, TXT, CSV, Markdown, Web
├── embeddings/      # Phase 2: Chunking, embedding models, vector stores
│   ├── chunking/    # Recursive & semantic splitting
│   ├── models/      # OpenAI, BGE, Sentence Transformers
│   └── vectorstore/ # FAISS & ChromaDB
├── retrieval/       # Phase 3: Search, strategies, reranking
│   ├── search/      # Similarity, BM25, Hybrid
│   ├── strategies/  # Multi-query, parent-child, compression
│   └── reranking/   # Cross-encoder, Cohere
├── generation/      # Phase 4: LLM integration
│   ├── llm/         # OpenAI, Anthropic, Gemini, Ollama
│   └── prompts/     # Templates & formatting
├── evaluation/      # Phase 5: RAGAS, DeepEval, custom metrics
├── api/             # FastAPI backend
│   └── routes/      # Ingest, query, admin, feedback
└── monitoring/      # Prometheus metrics & query logging
```

---

## Configuration

All settings are managed via environment variables (`.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `RAGX_EMBEDDING_PROVIDER` | `sentence-transformer` | Embedding model provider |
| `RAGX_VECTORSTORE_TYPE` | `chroma` | Vector store backend |
| `RAGX_CHUNK_SIZE` | `500` | Chunk size in tokens |
| `RAGX_RETRIEVAL_STRATEGY` | `hybrid` | Search strategy |
| `RAGX_RERANKER` | `cross-encoder` | Reranking method |
| `RAGX_LLM_PROVIDER` | `openai` | LLM provider |
| `RAGX_LLM_MODEL` | `gpt-4o` | LLM model name |

See [.env.example](.env.example) for the complete list.

---

## Testing

```bash
# Run all tests
make test

# Phase-specific tests
make test-phase1  # Ingestion
make test-phase2  # Embeddings
make test-phase3  # Retrieval
make test-phase4  # Generation
make test-phase5  # API
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Framework | LangChain 1.3+ |
| Document Parsing | PyPDF, python-docx, BeautifulSoup, Pandas |
| Embeddings | OpenAI, BGE, Sentence Transformers |
| Vector Stores | ChromaDB, FAISS |
| Re-ranking | Cross-Encoders, Cohere |
| LLMs | GPT-4o, Claude, Gemini, Llama |
| API | FastAPI + Uvicorn |
| Evaluation | RAGAS, DeepEval |
| Monitoring | Prometheus + Grafana |
| Deployment | Docker, Kubernetes |

---

## License

MIT License
