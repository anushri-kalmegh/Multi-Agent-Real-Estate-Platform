import uuid

from agents.phase2_orchestrator import run_conversation
from agents.semantic_legal_agent import answer_legal_query, semantic_search


def test_semantic_retrieval_has_page_citations(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    evidence = semantic_search("When must a real estate project be registered?", top_k=4)
    assert len(evidence) == 4
    assert all(item["page"] > 0 for item in evidence)
    assert all(item["source_file"].endswith(".pdf") for item in evidence)


def test_semantic_answer_is_grounded(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    result = answer_legal_query("What rights does a home buyer have under RERA?")
    assert result["retrieval_method"] == "chroma_semantic"
    assert result["generation_source"] == "extractive_fallback"
    assert result["sources"]
    assert "[1]" in result["answer"]


def test_langgraph_uses_semantic_legal_agent(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    result = run_conversation(
        "What are a promoter's duties under RERA?", str(uuid.uuid4())
    )
    assert result["type"] == "legal"
    assert result["data"]["retrieval_method"] == "chroma_semantic"
