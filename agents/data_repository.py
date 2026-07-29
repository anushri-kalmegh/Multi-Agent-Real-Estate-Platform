"""Read-only repository for SQLite-backed agent data with CSV fallback."""

from __future__ import annotations

import sqlite3

import pandas as pd

from agents.config import DATA_DIR, DATABASE_PATH


PROPERTY_COLUMNS = [
    "property_id", "city", "locality", "bhk", "area_sqft",
    "price", "price_per_sqft", "possession_status", "amenities",
]


def database_available() -> bool:
    return DATABASE_PATH.exists()


def load_properties() -> pd.DataFrame:
    if database_available():
        with sqlite3.connect(DATABASE_PATH) as connection:
            return pd.read_sql_query("SELECT * FROM properties", connection)
    enriched = DATA_DIR / "properties_enriched.csv"
    source = enriched if enriched.exists() else DATA_DIR / "properties_clean.csv"
    return pd.read_csv(source)


def search_properties(
    city: str,
    max_budget: float,
    bhk: int | None = None,
    locality: str | None = None,
    top_n: int = 100,
) -> pd.DataFrame:
    if database_available():
        clauses = ["LOWER(city) = LOWER(?)", "price <= ?"]
        parameters: list[object] = [city.strip(), float(max_budget)]
        if bhk is not None:
            clauses.append("bhk = ?")
            parameters.append(float(bhk))
        if locality:
            clauses.append("LOWER(locality) = LOWER(?)")
            parameters.append(locality.strip())
        parameters.extend([float(max_budget), int(top_n)])
        query = f"""
            SELECT {", ".join(PROPERTY_COLUMNS)}
            FROM properties
            WHERE {" AND ".join(clauses)}
            ORDER BY ABS(? - price) ASC, price_per_sqft ASC
            LIMIT ?
        """
        with sqlite3.connect(DATABASE_PATH) as connection:
            return pd.read_sql_query(query, connection, params=parameters)

    df = load_properties()
    match = df["city"].astype(str).str.casefold().eq(city.strip().casefold())
    match &= df["price"].le(max_budget)
    if bhk is not None:
        match &= df["bhk"].eq(bhk)
    if locality:
        match &= df["locality"].astype(str).str.casefold().eq(locality.strip().casefold())
    result = df.loc[match, PROPERTY_COLUMNS].copy()
    result["price_diff"] = (max_budget - result["price"]).abs()
    return result.sort_values(["price_diff", "price_per_sqft"]).head(top_n).drop(
        columns="price_diff"
    )


def get_locality(city: str, locality: str) -> dict | None:
    if database_available():
        with sqlite3.connect(DATABASE_PATH) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT * FROM locality_scores
                WHERE LOWER(city) = LOWER(?) AND LOWER(locality) = LOWER(?)
                LIMIT 1
                """,
                (city.strip(), locality.strip()),
            ).fetchone()
            return dict(row) if row else None

    complete = DATA_DIR / "locality_scores_complete.csv"
    source = complete if complete.exists() else DATA_DIR / "locality_scores.csv"
    df = pd.read_csv(source)
    match = df[
        df["city"].astype(str).str.casefold().eq(city.strip().casefold())
        & df["locality"].astype(str).str.casefold().eq(locality.strip().casefold())
    ]
    return match.iloc[0].to_dict() if not match.empty else None
