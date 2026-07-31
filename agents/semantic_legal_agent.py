"""Semantic legal RAG with evidence citations and keyword fallback."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from agents.config import CHROMA_DIR, RAG_DIR
from agents.legal_agent import answer_legal_query as keyword_answer
from agents.llm_service import generate_legal_answer


COLLECTION_NAME = "propwise_legal_v1"


def semantic_search(question: str, top_k: int = 4) -> list[dict]:
    if not CHROMA_DIR.exists():
        return []
    import chromadb

    collection = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_collection(
        name=COLLECTION_NAME
    )
    count = collection.count()
    if not count:
        return []
    candidate_count = min(max(top_k * 4, 12), count)
    result = collection.query(
        query_texts=[question],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )
    stop_words = {
        "what", "when", "where", "which", "does", "under", "have", "must",
        "are", "the", "and", "for", "with", "from", "into", "that", "this",
    }
    query_terms = {
        word for word in re.findall(r"[a-z]{3,}", question.casefold())
        if word not in stop_words
    }
    evidence = []
    for document, metadata, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        document_terms = set(re.findall(r"[a-z]{3,}", document.casefold()))
        lexical = len(query_terms & document_terms) / max(1, len(query_terms))
        hybrid_score = similarity * 0.78 + lexical * 0.22
        evidence.append({
            "text": document,
            "source_file": metadata["source_file"],
            "page": int(metadata["page"]),
            "chunk_id": int(metadata["chunk_id"]),
            "similarity": round(similarity, 4),
            "hybrid_score": round(hybrid_score, 4),
        })
    return sorted(evidence, key=lambda item: item["hybrid_score"], reverse=True)[:top_k]


def answer_legal_query(question: str) -> dict:
    try:
        evidence = semantic_search(question)
    except Exception:
        evidence = []
    if not evidence:
        legacy = keyword_answer(question)
        legacy["retrieval_method"] = "keyword_fallback"
        legacy["sources"] = [] if legacy.get("chunk_id") == -1 else [{
            "source_file": legacy.get("source_document", "Unknown"),
            "page": None,
            "chunk_id": legacy.get("chunk_id"),
            "similarity": None,
        }]
        return legacy

    answer, generation_source = generate_legal_answer(question, evidence)
    confidence = round(
        sum(item["similarity"] for item in evidence[:3]) / min(3, len(evidence)) * 100,
        2,
    )
    return {
        "answer": answer,
        "source_document": evidence[0]["source_file"],
        "chunk_id": evidence[0]["chunk_id"],
        "confidence_score": confidence,
        "retrieval_method": "chroma_semantic",
        "generation_source": generation_source,
        "sources": [{
            "source_file": item["source_file"],
            "page": item["page"],
            "chunk_id": item["chunk_id"],
            "similarity": item["similarity"],
        } for item in evidence],
    }
