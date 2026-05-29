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

    # Override settings for local testing
    os.environ["RAGX_EMBEDDING_PROVIDER"] = "sentence-transformer"
    os.environ["RAGX_VECTORSTORE_TYPE"] = "chroma"
    os.environ["RAGX_LLM_PROVIDER"] = "gemini"
    os.environ["RAGX_LLM_MODEL"] = "gemini-1.5-flash"
    
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
        retrieval_engine = RetrievalEngine(settings=settings, vectorstore=vectorstore)
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
            print(f"\n❌ Generation failed. (Note: Ensure Ollama is running locally if using 'ollama' provider.)")
            print(f"Error: {e}")

    finally:
        # Cleanup temporary file
        os.unlink(temp_path)
        print("🧹 Cleaned up temporary test files.")


if __name__ == "__main__":
    main()
