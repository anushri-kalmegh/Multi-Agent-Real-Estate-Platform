"""Build page-aware legal chunks and a persistent Chroma semantic index."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from PyPDF2 import PdfReader


ROOT = Path(__file__).resolve().parent
LEGAL_DIR = ROOT / "legal_docs"
RAG_DIR = ROOT / "rag"
CHUNKS_PATH = RAG_DIR / "legal_chunks.json"
CHROMA_PATH = RAG_DIR / "chroma"
COLLECTION_NAME = "propwise_legal_v1"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def page_chunks(text: str, size: int = 280, overlap: int = 60):
    words = clean_text(text).split()
    step = max(1, size - overlap)
    for start in range(0, len(words), step):
        chunk = words[start : start + size]
        if len(chunk) >= 40:
            yield " ".join(chunk)
        if start + size >= len(words):
            break


def extract_chunks() -> list[dict]:
    chunks = []
    for pdf_path in sorted(LEGAL_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for page_chunk_index, text_chunk in enumerate(page_chunks(text)):
                chunks.append({
                    "chunk_id": len(chunks),
                    "source_file": pdf_path.name,
                    "page": page_number,
                    "page_chunk": page_chunk_index,
                    "text": text_chunk,
                })
    return chunks


def build_chroma(chunks: list[dict], rebuild: bool = True) -> int:
    import chromadb

    if rebuild and CHROMA_PATH.exists():
        shutil.rmtree(CHROMA_PATH)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "PropWise page-aware legal evidence", "hnsw:space": "cosine"},
    )
    batch_size = 100
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[f"legal-{item['chunk_id']}" for item in batch],
            documents=[item["text"] for item in batch],
            metadatas=[{
                "chunk_id": item["chunk_id"],
                "source_file": item["source_file"],
                "page": item["page"],
                "page_chunk": item["page_chunk"],
            } for item in batch],
        )
    return collection.count()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-only", action="store_true")
    args = parser.parse_args()
    RAG_DIR.mkdir(exist_ok=True)
    chunks = extract_chunks()
    CHUNKS_PATH.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    indexed = 0 if args.chunks_only else build_chroma(chunks)
    print(json.dumps({
        "pdfs": len(list(LEGAL_DIR.glob("*.pdf"))),
        "page_aware_chunks": len(chunks),
        "chroma_records": indexed,
        "collection": COLLECTION_NAME,
    }, indent=2))


if __name__ == "__main__":
    main()
