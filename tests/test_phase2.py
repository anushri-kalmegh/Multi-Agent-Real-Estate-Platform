import uuid

from agents.phase2_orchestrator import run_conversation


def test_clarification_memory(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    thread = str(uuid.uuid4())
    first = run_conversation("I need a 2 BHK under 80 lakh for investment", thread)
    assert first["status"] == "needs_input"
    assert first["requirements"]["bhk"] == 2
    second = run_conversation("Pune", thread)
    assert second["status"] == "ok"
    assert second["requirements"]["city"] == "Pune"
    assert len(second["data"]["recommendations"]) == 4


def test_legal_route(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    result = run_conversation("What is RERA registration?", str(uuid.uuid4()))
    assert result["type"] == "legal"
    assert result["data"]["source_document"] == "rera book.pdf"


def test_general_route(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    result = run_conversation("hello", str(uuid.uuid4()))
    assert result["status"] == "needs_input"
    assert result["type"] == "text"
