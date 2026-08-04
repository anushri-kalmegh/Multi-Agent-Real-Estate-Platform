import math
import uuid

from agents.financial_agent import analyze_financials, monthly_emi
from agents.llm_service import deterministic_extract
from agents.phase2_orchestrator import run_conversation


PROPERTY = {"price": 8_000_000}
LOCALITY = {
    "nearest_metro_km": 2.0,
    "schools_nearby": 10,
    "hospitals_nearby": 6,
    "traffic_index": 5,
    "connectivity_score": 8,
    "rent_yield_pct": 4.0,
    "data_confidence": 1.0,
}


def test_emi_formula():
    emi = monthly_emi(6_400_000, 8.5, 20)
    assert round(emi, 0) == 55541


def test_financial_scenarios_are_ordered():
    result = analyze_financials(PROPERTY, LOCALITY)
    scenarios = result["scenarios"]
    assert scenarios["conservative"]["future_property_value"] < scenarios["base"]["future_property_value"]
    assert scenarios["base"]["future_property_value"] < scenarios["optimistic"]["future_property_value"]
    assert result["model_info"]["type"] == "derived_proxy_not_historical"
    assert result["upfront_cash_required"] == 2_160_000


def test_financial_assumptions_parse_and_reach_pipeline(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    query = (
        "2 BHK in Pune under 80 lakh for investment with down payment 25%, "
        "interest 8%, 15 year loan and hold for 7 years"
    )
    parsed = deterministic_extract(query)
    assert parsed["financial_assumptions"] == {
        "down_payment_pct": 25.0,
        "annual_interest_rate_pct": 8.0,
        "loan_tenure_years": 15,
        "holding_period_years": 7,
    }
    result = run_conversation(query, str(uuid.uuid4()))
    assumptions = result["data"]["recommendations"][0]["financial_analysis"]["assumptions"]
    assert assumptions["down_payment_pct"] == 25.0
    assert assumptions["holding_period_years"] == 7
