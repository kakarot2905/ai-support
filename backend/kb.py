"""Knowledge base: text extraction, chunking, embedding + vector search (ChromaDB)."""
import io
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from docx import Document
from pypdf import PdfReader

from db import DATA_DIR

SUPPORTED_TYPES = {".pdf", ".txt", ".docx"}
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
TOP_K = 4
# Calibration knob: passages scoring below this cosine similarity (0..1) aren't sent to the LLM.
# Measured on samples/ with all-MiniLM-L6-v2: specific questions 0.37-0.72, broad on-topic ones
# ("what are company policies") 0.15-0.17, off-topic ("weather", "football") 0.07-0.09.
MIN_SIMILARITY = 0.10

_client = chromadb.PersistentClient(path=str(DATA_DIR / "chroma"))
# DefaultEmbeddingFunction = all-MiniLM-L6-v2 run locally via ONNX (no API calls, no torch).
_collection = _client.get_or_create_collection(
    "knowledge_base",
    embedding_function=DefaultEmbeddingFunction(),
    configuration={"hnsw": {"space": "cosine"}},
)


class ExtractionError(Exception):
    pass


def extract_text(filename: str, data: bytes) -> str:
    ext = Path(filename).suffix.lower()
    try:
        if ext == ".txt":
            text = data.decode("utf-8", errors="replace")
        elif ext == ".pdf":
            text = "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        elif ext == ".docx":
            text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
        else:
            raise ExtractionError(f"Unsupported file type '{ext}'")
    except ExtractionError:
        raise
    except Exception as e:
        raise ExtractionError(f"Could not read file: {e}") from e
    if not text.strip():
        raise ExtractionError("No readable text found (empty or scanned/image-only document)")
    return text


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding window of ~size chars with overlap, breaking on whitespace so words stay whole."""
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + size // 2, end)
            cut = max(cut, text.rfind("\n", start + size // 2, end))
            if cut != -1:
                end = cut
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - overlap
        while start < end and not text[start - 1].isspace():  # snap to the next word start
            start += 1
    return chunks


def add_document(doc_id: int, filename: str, chunks: list[str]) -> None:
    _collection.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[{"source": filename, "doc_id": doc_id} for _ in chunks],
    )


def search(query: str, k: int = TOP_K) -> list[dict]:
    """Top-k chunks for the query, best first, dropping any below MIN_SIMILARITY.

    Chroma returns cosine *distance*; similarity = 1 - distance.
    """
    count = _collection.count()
    if count == 0:
        return []
    res = _collection.query(query_texts=[query], n_results=min(k, count))
    chunks = [
        {"text": doc, "source": meta["source"], "similarity": 1 - dist}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    return [c for c in chunks if c["similarity"] >= MIN_SIMILARITY]
