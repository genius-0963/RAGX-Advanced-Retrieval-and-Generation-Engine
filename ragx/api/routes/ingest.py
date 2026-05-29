"""
RAGX Ingestion Routes — Document upload and management endpoints.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile

from ragx.api.dependencies import get_embedding_pipeline, get_ingestion_pipeline
from ragx.api.schemas import DocumentInfo, DocumentListResponse, IngestURLRequest
from ragx.config.logging_config import get_logger
from ragx.config.settings import get_settings

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ingest/file", response_model=dict)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload and process a single document."""
    settings = get_settings()
    raw_dir = Path(settings.data_raw_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    file_path = raw_dir / file.filename
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Process in background
    background_tasks.add_task(_process_file, str(file_path))

    return {
        "status": "processing",
        "filename": file.filename,
        "message": "Document queued for ingestion.",
    }


@router.post("/ingest/url", response_model=dict)
async def ingest_url(
    request: IngestURLRequest,
    background_tasks: BackgroundTasks,
):
    """Ingest a document from a URL."""
    background_tasks.add_task(_process_url, request.url)
    return {
        "status": "processing",
        "url": request.url,
        "message": "URL queued for ingestion.",
    }


@router.post("/ingest/batch", response_model=dict)
async def ingest_batch(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
):
    """Batch upload multiple documents."""
    settings = get_settings()
    raw_dir = Path(settings.data_raw_path)
    raw_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for file in files:
        file_path = raw_dir / file.filename
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(str(file_path))
        except Exception as e:
            logger.error("batch_save_failed", filename=file.filename, error=str(e))

    background_tasks.add_task(_process_files_batch, saved_files)

    return {
        "status": "processing",
        "files_queued": len(saved_files),
        "message": f"{len(saved_files)} documents queued for ingestion.",
    }


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """List all ingested documents."""
    settings = get_settings()
    processed_dir = Path(settings.data_processed_path)

    documents: list[DocumentInfo] = []
    if processed_dir.exists():
        import json
        for json_file in processed_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    first = data[0] if data else {}
                    meta = first.get("metadata", {})
                    documents.append(DocumentInfo(
                        id=meta.get("document_id", json_file.stem),
                        source=meta.get("source", str(json_file)),
                        file_type=meta.get("file_type", "unknown"),
                        upload_time=meta.get("upload_time", ""),
                        chunk_count=len(data),
                    ))
            except Exception as e:
                logger.warning("failed_reading_doc_info", file=str(json_file), error=str(e))

    return DocumentListResponse(documents=documents, total=len(documents))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document by ID."""
    settings = get_settings()
    processed_dir = Path(settings.data_processed_path)

    # Find and delete the document file
    deleted = False
    for json_file in processed_dir.glob("*.json"):
        if document_id in json_file.stem:
            json_file.unlink()
            deleted = True
            break

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    return {"status": "deleted", "document_id": document_id}


# ── Background Tasks ─────────────────────────────────────────────────────────


def _process_file(file_path: str) -> None:
    """Background task to ingest and embed a single file."""
    try:
        pipeline = get_ingestion_pipeline()
        documents = pipeline.ingest_file(file_path)
        if documents:
            embedding_pipeline = get_embedding_pipeline()
            embedding_pipeline.process(documents)
        logger.info("file_ingested", file=file_path, docs=len(documents))
    except Exception as e:
        logger.error("file_ingestion_failed", file=file_path, error=str(e))


def _process_url(url: str) -> None:
    """Background task to ingest and embed a URL."""
    try:
        pipeline = get_ingestion_pipeline()
        documents = pipeline.ingest_url(url)
        if documents:
            embedding_pipeline = get_embedding_pipeline()
            embedding_pipeline.process(documents)
        logger.info("url_ingested", url=url, docs=len(documents))
    except Exception as e:
        logger.error("url_ingestion_failed", url=url, error=str(e))


def _process_files_batch(file_paths: list[str]) -> None:
    """Background task to ingest and embed multiple files."""
    try:
        pipeline = get_ingestion_pipeline()
        all_docs = []
        for fp in file_paths:
            try:
                docs = pipeline.ingest_file(fp)
                all_docs.extend(docs)
            except Exception as e:
                logger.error("batch_file_failed", file=fp, error=str(e))

        if all_docs:
            embedding_pipeline = get_embedding_pipeline()
            embedding_pipeline.process(all_docs)
        logger.info("batch_ingested", total_docs=len(all_docs))
    except Exception as e:
        logger.error("batch_ingestion_failed", error=str(e))
