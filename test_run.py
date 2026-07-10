"""
RAGX — End-to-End Test Run Script

This script performs a complete test run of the RAGX platform without starting the API server.
It verifies that all phases (Ingestion -> Embedding -> Retrieval -> Generation) work together.

To run this script:
    python test_run.py
"""

import os
import sys
import tempfile
from pathlib import Path
from pprint import pprint

# Ensure the ragx package is in the Python path
sys.path.insert(0, str(Path(__file__).parent))

from ragx.config.settings import get_settings
from ragx.ingestion.pipeline import IngestionPipeline
from ragx.embeddings.pipeline import EmbeddingPipeline
from ragx.retrieval.engine import RetrievalEngine
from ragx.generation.pipeline import GenerationPipeline


def main():
    print("🚀 Starting RAGX End-to-End Test Run...\n")

    # ── Configuration: fully local, no API key required ──────────────────────
    os.environ["RAGX_EMBEDDING_PROVIDER"] = "sentence-transformer"
    os.environ["RAGX_EMBEDDING_MODEL"]    = "all-MiniLM-L6-v2"
    os.environ["RAGX_VECTORSTORE_TYPE"]   = "chroma"
    os.environ["RAGX_LLM_PROVIDER"]       = "ollama"
    os.environ["RAGX_RETRIEVAL_STRATEGY"] = "hybrid"
    os.environ["RAGX_RERANKER"]           = "cross-encoder"
    # OLLAMA_MODEL is read from settings (no RAGX_ prefix)
    os.environ["OLLAMA_MODEL"]            = "llama3.2:3b"
    os.environ["OLLAMA_BASE_URL"]         = "http://localhost:11434"

    # Must clear lru_cache so env overrides take effect
    get_settings.cache_clear()
    settings = get_settings()

    print(f"🔧 Configuration:")
    print(f"  - Embeddings: {settings.embedding_provider.value} ({settings.embedding_model})")
    print(f"  - Vector Store: {settings.vectorstore_type.value}")
    print(f"  - LLM: {settings.llm_provider.value} ({settings.llm_model})\n")

    # Create a temporary Markdown file to ingest
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
        temp_file.write("# RAGX Test Document\n\n")
        temp_file.write("RAGX is a highly advanced Retrieval-Augmented Generation engine built in Python. ")
        temp_file.write("It supports multiple document formats, advanced chunking strategies, and hybrid retrieval. ")
        temp_file.write("The default vector store is ChromaDB, and it uses sentence-transformers for local embeddings.")
        temp_path = temp_file.name

    try:
        # Phase 1: Ingestion
        print("📄 Phase 1: Ingesting Document...")
        ingestion_pipeline = IngestionPipeline(settings=settings)
        documents = ingestion_pipeline.ingest_file(temp_path)
        print(f"  ✓ Ingested {len(documents)} document(s).\n")

        # Phase 2: Embedding & Storage
        print("🧠 Phase 2: Generating Embeddings & Storing...")
        embedding_pipeline = EmbeddingPipeline(settings=settings)
        ids = embedding_pipeline.process(documents)
        print(f"  ✓ Stored {len(ids)} chunk(s) in Vector Database.\n")

        # Phase 3: Retrieval
        print("🔍 Phase 3: Initializing Retrieval Engine...")
        vectorstore = embedding_pipeline.get_vectorstore()
        # Pass the embedding model so context compression works too
        embeddings = embedding_pipeline._get_embedding_model().get_langchain_embeddings()
        retrieval_engine = RetrievalEngine(
            settings=settings,
            vectorstore=vectorstore,
            embeddings=embeddings,
        )
        print("  ✓ Retrieval Engine ready.\n")

        # Phase 4: Generation
        print("🤖 Phase 4: Generating Answer...")
        generation_pipeline = GenerationPipeline(settings=settings, retrieval_engine=retrieval_engine)
        
        query = "What is RAGX and what vector store does it use by default?"
        print(f"  ➤ Query: '{query}'")
        
        try:
            response = generation_pipeline.query(query)
            print("\n✅ Generation Successful!\n")
            print("========================================")
            print("ANSWER:")
            print("========================================")
            print(response["answer"])
            print("========================================")
            print(f"Confidence Score: {response['confidence_score']:.2f}")
            print(f"Latency: {response['latency_ms']} ms")
            print("Sources:")
            for idx, source in enumerate(response["sources"]):
                print(f"  [{idx + 1}] {source['source']} (Excerpt: {source['excerpt'][:50]}...)")
            print("========================================\n")
        except Exception as e:
            if "connection" in str(e).lower() or "11434" in str(e):
                print(f"\n❌ Generation failed — Ollama server not responding.")
                print(f"   Make sure Ollama is running: ollama serve")
                print(f"   And the model is downloaded: ollama pull llama3.2:3b")
            elif "model" in str(e).lower() and "not found" in str(e).lower():
                print(f"\n❌ Model not found locally.")
                print(f"   Run: ollama pull llama3.2:3b")
            else:
                print(f"\n❌ Generation failed: {e}")

    finally:
        # Cleanup temporary file
        os.unlink(temp_path)
        print("🧹 Cleaned up temporary test files.")


if __name__ == "__main__":
    main()
