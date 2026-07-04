"""Document ingestion: chunk the raw text and add to the Chroma vector store."""

from __future__ import annotations

from ..config import REPO_ROOT
from ..db import crud
from ..db.session import SessionLocal
from ..utils.logging import get_logger

log = get_logger(__name__)


def _chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Simple character-window chunking."""
    step = max(1, size - overlap)
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def run_ingestion(job_id: str, text: str, dataset_id: str | None, chunk_size: int, chunk_overlap: int) -> None:
    """Chunk, embed, and persist a document into the Chroma index."""
    db = SessionLocal()
    try:
        crud.update_ingestion_job(db, job_id, status="running")
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document

        from utils.lc_retriever import EMBED_MODEL

        chunks = _chunk_text(text, chunk_size, chunk_overlap)
        if not chunks:
            crud.update_ingestion_job(db, job_id, status="completed", chunks_added=0)
            return

        documents = [
            Document(
                page_content=chunk,
                metadata={"ingested": True, "chunk_index": index, "dataset_id": dataset_id},
            )
            for index, chunk in enumerate(chunks)
        ]

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

        persist_dir = REPO_ROOT / "data" / "chroma_db"
        persist_dir.mkdir(parents=True, exist_ok=True)
        if any(persist_dir.iterdir()):
            store = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)
            store.add_documents(documents)
        else:
            Chroma.from_documents(documents, embedding=embeddings, persist_directory=str(persist_dir))

        crud.update_ingestion_job(db, job_id, status="completed", chunks_added=len(documents))
    except Exception as exc:
        log.exception("ingestion job %s failed", job_id)
        crud.update_ingestion_job(db, job_id, status="failed", error=str(exc))
    finally:
        db.close()
