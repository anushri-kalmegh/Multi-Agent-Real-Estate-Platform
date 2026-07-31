"""Optional Groq conversation intelligence with deterministic fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv


load_dotenv()


SUPPORTED_CITIES = [
    "Pune", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata",
    "Mumbai", "Thane", "Kalyan", "Agartala", "Palghar", "Bhiwandi",
    "Gurgaon", "Nagpur",
]


def groq_enabled() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))


def deterministic_extract(query: str, previous: dict | None = None) -> dict:
    previous = previous or {}
    lowered = query.casefold()
    result: dict[str, Any] = {
        "intent": previous.get("intent", "general"),
        "city": previous.get("city"),
        "bhk": previous.get("bhk"),
        "max_budget": previous.get("max_budget"),
        "purpose": previous.get("purpose", "self-use"),
        "locality": previous.get("locality"),
        "language": previous.get("language", "English"),
        "financial_assumptions": previous.get("financial_assumptions", {}),
    }
    legal_words = ("rera", "legal", "law", "act", "compliance", "registration", "promoter")
    if any(word in lowered for word in legal_words):
        result["intent"] = "legal"
    elif any(word in lowered for word in ("bhk", "flat", "property", "house", "apartment")):
        result["intent"] = "property"

    for city in SUPPORTED_CITIES:
        if city.casefold() in lowered:
            result["city"] = city
            break
    bhk = re.search(r"(\d+)\s*bhk", query, re.IGNORECASE)
    if bhk:
        result["bhk"] = int(bhk.group(1))
    budget = re.search(r"(\d+(?:\.\d+)?)\s*(lakhs?|crores?|cr)\b", query, re.IGNORECASE)
    if budget:
        value, unit = float(budget.group(1)), budget.group(2).casefold()
        result["max_budget"] = int(value * (100_000 if "lakh" in unit else 10_000_000))
    else:
        plain = re.search(r"\b(\d{5,10})\b", query)
        if plain:
            result["max_budget"] = int(plain.group(1))
    if any(word in lowered for word in ("invest", "roi", "rental", "rent", "yield")):
        result["purpose"] = "investment"
    elif any(word in lowered for word in ("self use", "self-use", "living", "family")):
        result["purpose"] = "self-use"
    if result["intent"] == "general" and any(
        result.get(field) is not None for field in ("city", "bhk", "max_budget")
    ):
        result["intent"] = "property"
    financial = dict(result.get("financial_assumptions") or {})
    patterns = {
        "down_payment_pct": r"(?:down payment|downpayment)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
        "annual_interest_rate_pct": r"(?:interest|loan rate)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%",
        "loan_tenure_years": r"(\d+)\s*(?:year|yr)s?\s*(?:home\s*)?loan",
        "holding_period_years": r"(?:hold|holding period)\s*(?:for\s*)?(\d+)\s*(?:year|yr)s?",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, lowered)
        if match:
            value = float(match.group(1))
            financial[name] = int(value) if name.endswith("_years") else value
    result["financial_assumptions"] = financial
    return result


QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["property", "legal", "general"]},
        "city": {"type": ["string", "null"]},
        "bhk": {"type": ["integer", "null"]},
        "max_budget": {"type": ["integer", "null"]},
        "purpose": {"type": "string", "enum": ["investment", "self-use"]},
        "locality": {"type": ["string", "null"]},
        "language": {"type": "string"},
        "financial_assumptions": {
            "type": "object",
            "properties": {
                "down_payment_pct": {"type": ["number", "null"]},
                "annual_interest_rate_pct": {"type": ["number", "null"]},
                "loan_tenure_years": {"type": ["integer", "null"]},
                "holding_period_years": {"type": ["integer", "null"]},
            },
            "required": [
                "down_payment_pct", "annual_interest_rate_pct",
                "loan_tenure_years", "holding_period_years",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "intent", "city", "bhk", "max_budget", "purpose", "locality",
        "language", "financial_assumptions",
    ],
    "additionalProperties": False,
}


def extract_requirements(query: str, previous: dict | None = None) -> tuple[dict, str]:
    fallback = deterministic_extract(query, previous)
    if not groq_enabled():
        return fallback, "deterministic"
    try:
        from groq import Groq

        response = Groq().chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract real-estate requirements. Merge the new message with previous "
                        "requirements. Budget must be integer INR. Never invent missing values."
                    ),
                },
                {"role": "user", "content": json.dumps({"previous": previous or {}, "message": query})},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "property_requirements", "strict": True, "schema": QUERY_SCHEMA},
            },
        )
        return json.loads(response.choices[0].message.content), "groq"
    except Exception:
        return fallback, "deterministic_fallback"


def generate_summary(query: str, result: dict, language: str = "English") -> tuple[str, str]:
    if not groq_enabled() or result.get("status") != "ok":
        return "The multi-agent review is complete. Here are the strongest matches.", "template"
    try:
        from groq import Groq

        compact = []
        for item in result.get("recommendations", []):
            compact.append({
                "rank": item["rank"],
                "locality": item["property"]["locality"],
                "price": item["property"]["price"],
                "match_score": item["rank_score"],
                "yield": item["roi_analysis"]["rental_yield_pct"],
                "risk": item["roi_analysis"]["risk_level"],
                "price_verdict": item["price_analysis"]["verdict"],
            })
        response = Groq().chat.completions.create(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            temperature=0.2,
            max_tokens=220,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are PropWise AI. Answer in {language}. Summarize only supplied "
                        "agent evidence in 3 concise sentences. Do not add facts or guarantees."
                    ),
                },
                {"role": "user", "content": json.dumps({"query": query, "ranked_results": compact})},
            ],
        )
        return response.choices[0].message.content.strip(), "groq"
    except Exception:
        return "The multi-agent review is complete. Here are the strongest matches.", "template_fallback"


def generate_legal_answer(question: str, evidence: list[dict]) -> tuple[str, str]:
    """Answer only from retrieved evidence; use an extractive fallback offline."""
    if groq_enabled():
        try:
            from groq import Groq

            numbered = [
                {
                    "citation": f"[{index}]",
                    "source": item["source_file"],
                    "page": item["page"],
                    "text": item["text"],
                }
                for index, item in enumerate(evidence, start=1)
            ]
            response = Groq().chat.completions.create(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
                temperature=0,
                max_tokens=420,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Answer the legal question only from the supplied evidence. "
                            "Cite claims with [1], [2] notation. If evidence is insufficient, "
                            "say so. Do not provide personal legal advice."
                        ),
                    },
                    {"role": "user", "content": json.dumps({"question": question, "evidence": numbered})},
                ],
            )
            return response.choices[0].message.content.strip(), "groq_grounded"
        except Exception:
            pass

    text = re.sub(r"\s+", " ", evidence[0]["text"]).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    selected = [sentence for sentence in sentences if len(sentence) > 25][:4]
    answer = " ".join(selected) if selected else text[:700]
    if len(answer) > 900:
        answer = answer[:897].rsplit(" ", 1)[0] + "..."
    return f"{answer} [1]", "extractive_fallback"
