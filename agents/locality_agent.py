"""Locality intelligence agent backed by the Phase 1 data repository."""

from __future__ import annotations

import sys

from agents.data_repository import get_locality


def _bounded(value: float) -> float:
    return max(1.0, min(10.0, value))


def get_locality_insights(city, locality):
    """Return infrastructure signals, provenance and a weighted 1–10 score."""
    try:
        if not city or not locality:
            raise ValueError("Both city and locality must be provided.")

        row = get_locality(str(city), str(locality))
        if not row:
            raise ValueError(f"Locality '{locality}' in city '{city}' was not found.")

        metro_km = float(row["nearest_metro_km"])
        schools = int(row["schools_nearby"])
        hospitals = int(row["hospitals_nearby"])
        traffic = int(row["traffic_index"])
        connectivity = int(row["connectivity_score"])
        rent_yield = float(row["rent_yield_pct"])

        metro_score = _bounded(10.0 - 9.0 * (metro_km - 0.5) / 7.5)
        schools_score = _bounded(1.0 + 9.0 * (schools - 2) / 13.0)
        hospitals_score = _bounded(1.0 + 9.0 * (hospitals - 1) / 9.0)
        connectivity_score = _bounded(float(connectivity))
        rent_yield_score = _bounded(1.0 + 9.0 * (rent_yield - 2.5) / 2.5)

        raw_locality_score = round(
            metro_score * 0.25
            + schools_score * 0.20
            + hospitals_score * 0.15
            + connectivity_score * 0.25
            + rent_yield_score * 0.15,
            2,
        )
        data_source = row.get("data_source", "legacy")
        data_confidence = float(row.get("data_confidence", 1.0))
        # Pull uncertain estimates toward a neutral 5/10 so they cannot carry
        # the same ranking influence as curated locality observations.
        locality_score = round(
            5.0 + (raw_locality_score - 5.0) * data_confidence,
            2,
        )
        return {
            "city": str(row["city"]),
            "locality": str(row["locality"]),
            "nearest_metro_km": metro_km,
            "schools_nearby": schools,
            "hospitals_nearby": hospitals,
            "traffic_index": traffic,
            "connectivity_score": connectivity,
            "rent_yield_pct": rent_yield,
            "locality_score": locality_score,
            "raw_locality_score": raw_locality_score,
            "data_source": data_source,
            "data_confidence": data_confidence,
        }
    except Exception as error:
        print(f"Error in get_locality_insights: {error}", file=sys.stderr)
        return None


if __name__ == "__main__":
    print(get_locality_insights("Pune", "Alandi Road"))
