"""Advanced, assumption-driven property financial scenario analysis."""

from __future__ import annotations

import math
import pickle

import pandas as pd

from agents.config import ROI_MODEL_PATH


DEFAULT_ASSUMPTIONS = {
    "down_payment_pct": 20.0,
    "annual_interest_rate_pct": 8.5,
    "loan_tenure_years": 20,
    "holding_period_years": 5,
    "acquisition_cost_pct": 7.0,
    "annual_maintenance_pct": 1.0,
    "vacancy_pct": 5.0,
    "annual_rent_growth_pct": 5.0,
    "selling_cost_pct": 2.0,
}


def monthly_emi(principal: float, annual_rate_pct: float, years: int) -> float:
    months = max(1, int(years * 12))
    monthly_rate = annual_rate_pct / 1200
    if monthly_rate == 0:
        return principal / months
    factor = (1 + monthly_rate) ** months
    return principal * monthly_rate * factor / (factor - 1)


def remaining_balance(
    principal: float, annual_rate_pct: float, years: int, paid_months: int
) -> float:
    payment = monthly_emi(principal, annual_rate_pct, years)
    rate = annual_rate_pct / 1200
    if rate == 0:
        return max(0.0, principal - payment * paid_months)
    return max(
        0.0,
        principal * (1 + rate) ** paid_months
        - payment * (((1 + rate) ** paid_months - 1) / rate),
    )


def predict_proxy_appreciation(locality: dict) -> tuple[float, dict]:
    if not ROI_MODEL_PATH.exists():
        return 5.0, {"type": "fallback_assumption", "version": None}
    with ROI_MODEL_PATH.open("rb") as handle:
        bundle = pickle.load(handle)
    row = pd.DataFrame([{
        "nearest_metro_km": locality["nearest_metro_km"],
        "schools_nearby": locality["schools_nearby"],
        "hospitals_nearby": locality["hospitals_nearby"],
        "traffic_index": locality["traffic_index"],
        "connectivity_score": locality["connectivity_score"],
        "rent_yield_pct": locality["rent_yield_pct"],
        "locality_data_confidence": locality.get("data_confidence", 1.0),
    }])
    rate = float(bundle["model"].predict(row[bundle["features"]])[0])
    return round(max(2.0, min(10.0, rate)), 2), {
        "type": bundle["metadata"]["target_type"],
        "version": bundle["metadata"]["version"],
        "model_name": bundle["metadata"]["model_name"],
    }


def analyze_financials(
    selected_property: dict,
    locality_analysis: dict,
    assumptions: dict | None = None,
) -> dict:
    config = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    price = float(selected_property["price"])
    down_payment = price * config["down_payment_pct"] / 100
    acquisition_cost = price * config["acquisition_cost_pct"] / 100
    loan_principal = price - down_payment
    emi = monthly_emi(
        loan_principal,
        config["annual_interest_rate_pct"],
        config["loan_tenure_years"],
    )
    gross_annual_rent = price * float(locality_analysis["rent_yield_pct"]) / 100
    effective_annual_rent = gross_annual_rent * (1 - config["vacancy_pct"] / 100)
    annual_maintenance = price * config["annual_maintenance_pct"] / 100
    net_operating_income = effective_annual_rent - annual_maintenance
    annual_debt_service = emi * 12
    annual_cash_flow = net_operating_income - annual_debt_service
    upfront_cash = down_payment + acquisition_cost
    cash_on_cash = annual_cash_flow / upfront_cash * 100 if upfront_cash else 0
    net_rental_yield = net_operating_income / price * 100

    base_appreciation, model_info = predict_proxy_appreciation(locality_analysis)
    holding_years = int(config["holding_period_years"])
    scenario_rates = {
        "conservative": max(0.0, base_appreciation - 2.0),
        "base": base_appreciation,
        "optimistic": min(15.0, base_appreciation + 2.0),
    }
    scenarios = {}
    paid_months = min(holding_years * 12, int(config["loan_tenure_years"] * 12))
    balance = remaining_balance(
        loan_principal,
        config["annual_interest_rate_pct"],
        config["loan_tenure_years"],
        paid_months,
    )
    total_rent = sum(
        effective_annual_rent * (1 + config["annual_rent_growth_pct"] / 100) ** year
        for year in range(holding_years)
    )
    total_maintenance = annual_maintenance * holding_years
    total_emi = emi * paid_months
    for name, annual_rate in scenario_rates.items():
        future_value = price * (1 + annual_rate / 100) ** holding_years
        selling_cost = future_value * config["selling_cost_pct"] / 100
        sale_equity = future_value - selling_cost - balance
        net_profit = sale_equity + total_rent - total_maintenance - total_emi - upfront_cash
        total_return = net_profit / upfront_cash * 100 if upfront_cash else 0
        annualized = (
            ((max(0.01, upfront_cash + net_profit) / upfront_cash) ** (1 / holding_years) - 1) * 100
            if upfront_cash else 0
        )
        scenarios[name] = {
            "annual_appreciation_pct": round(annual_rate, 2),
            "future_property_value": round(future_value, 2),
            "remaining_loan_balance": round(balance, 2),
            "net_profit": round(net_profit, 2),
            "total_return_pct": round(total_return, 2),
            "annualized_return_pct": round(annualized, 2),
        }

    break_even_years = None
    if net_operating_income > 0:
        break_even_years = round(upfront_cash / net_operating_income, 1)
    return {
        "assumptions": config,
        "model_info": model_info,
        "proxy_appreciation_pct": base_appreciation,
        "down_payment": round(down_payment, 2),
        "acquisition_cost": round(acquisition_cost, 2),
        "upfront_cash_required": round(upfront_cash, 2),
        "loan_principal": round(loan_principal, 2),
        "monthly_emi": round(emi, 2),
        "gross_monthly_rent": round(gross_annual_rent / 12, 2),
        "net_operating_income": round(net_operating_income, 2),
        "net_rental_yield_pct": round(net_rental_yield, 2),
        "annual_cash_flow_after_emi": round(annual_cash_flow, 2),
        "cash_on_cash_return_pct": round(cash_on_cash, 2),
        "break_even_years_unlevered": break_even_years,
        "scenarios": scenarios,
        "warning": (
            "Scenario output uses configurable assumptions and a proxy appreciation "
            "model, not historical resale validation or financial advice."
        ),
    }
