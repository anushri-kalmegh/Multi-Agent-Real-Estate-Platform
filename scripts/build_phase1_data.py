"""Build the normalized Phase 1 data layer without modifying raw source files.

Outputs:
  data/properties_enriched.csv
  data/locality_scores_complete.csv
  data/propwise.db
  data/phase1_quality_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROPERTY_SOURCE = DATA_DIR / "properties_clean.csv"
LOCALITY_SOURCE = DATA_DIR / "locality_scores.csv"
PROPERTY_OUTPUT = DATA_DIR / "properties_enriched.csv"
LOCALITY_OUTPUT = DATA_DIR / "locality_scores_complete.csv"
DATABASE_OUTPUT = DATA_DIR / "propwise.db"
REPORT_OUTPUT = DATA_DIR / "phase1_quality_report.json"

PROPERTY_COLUMNS = [
    "property_id", "city", "locality", "bhk", "area_sqft", "price",
    "price_per_sqft", "possession_status", "amenities",
]
LOCALITY_METRICS = [
    "nearest_metro_km", "schools_nearby", "hospitals_nearby",
    "traffic_index", "connectivity_score", "rent_yield_pct",
]


def normalize_text(value: object, fallback: str = "Unknown") -> str:
    if pd.isna(value):
        return fallback
    cleaned = re.sub(r"\s+", " ", str(value)).strip()
    return cleaned.title() if cleaned else fallback


def normalize_amenities(value: object) -> str:
    if pd.isna(value):
        return ""
    items = {
        re.sub(r"\s+", " ", item).strip()
        for item in str(value).split(",")
        if item.strip()
    }
    return ", ".join(sorted(items, key=str.casefold))


def fingerprint(row: pd.Series) -> str:
    values = (
        row["city"].casefold(),
        row["locality"].casefold(),
        f"{row['bhk']:.1f}",
        f"{row['area_sqft']:.1f}",
        f"{row['price']:.0f}",
        row["possession_status"].casefold(),
        row["amenities"].casefold(),
    )
    return hashlib.sha1("|".join(values).encode("utf-8")).hexdigest()[:20]


def prepare_properties(source: Path) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(source)
    missing = set(PROPERTY_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Property source is missing columns: {sorted(missing)}")

    before = len(raw)
    df = raw[PROPERTY_COLUMNS].copy()
    df["city"] = df["city"].map(normalize_text)
    df["locality"] = df["locality"].map(normalize_text)
    df["possession_status"] = df["possession_status"].map(
        lambda value: normalize_text(value, "Unknown")
    )
    df["amenities"] = df["amenities"].map(normalize_amenities)
    for column in ("bhk", "area_sqft", "price", "price_per_sqft"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    valid = (
        df["bhk"].between(1, 20)
        & df["area_sqft"].between(100, 100_000)
        & df["price"].between(100_000, 2_000_000_000)
        & df["price_per_sqft"].between(100, 1_000_000)
        & df["city"].ne("Unknown")
        & df["locality"].ne("Unknown")
    )
    invalid_rows = int((~valid).sum())
    df = df.loc[valid].copy()

    df["listing_fingerprint"] = df.apply(fingerprint, axis=1)
    duplicate_counts = df.groupby("listing_fingerprint")["listing_fingerprint"].transform("size")
    df["duplicate_source_count"] = duplicate_counts.astype(int)
    df = (
        df.sort_values("property_id")
        .drop_duplicates("listing_fingerprint", keep="first")
        .reset_index(drop=True)
    )

    df["price_lakh"] = (df["price"] / 100_000).round(2)
    df["price_crore"] = (df["price"] / 10_000_000).round(3)
    df["amenity_count"] = df["amenities"].map(
        lambda value: 0 if not value else len(value.split(", "))
    )
    df["is_ready_to_move"] = (
        df["possession_status"].str.casefold().eq("ready to move").astype(int)
    )
    df["data_quality_score"] = (
        55
        + df["possession_status"].ne("Unknown").astype(int) * 15
        + df["amenity_count"].gt(0).astype(int) * 15
        + df["locality"].ne("Other").astype(int) * 15
    ).clip(0, 100)

    report = {
        "source_rows": before,
        "invalid_rows_removed": invalid_rows,
        "duplicate_rows_removed": int(before - invalid_rows - len(df)),
        "output_rows": len(df),
        "unique_cities": int(df["city"].nunique()),
        "unique_city_localities": int(df[["city", "locality"]].drop_duplicates().shape[0]),
    }
    return df, report


def prepare_localities(properties: pd.DataFrame, source: Path) -> tuple[pd.DataFrame, dict]:
    curated = pd.read_csv(source)
    required = {"city", "locality", *LOCALITY_METRICS}
    missing = required - set(curated.columns)
    if missing:
        raise ValueError(f"Locality source is missing columns: {sorted(missing)}")

    curated = curated[list(required)].copy()
    curated["city"] = curated["city"].map(normalize_text)
    curated["locality"] = curated["locality"].map(normalize_text)
    for column in LOCALITY_METRICS:
        curated[column] = pd.to_numeric(curated[column], errors="coerce")
    curated = curated.dropna(subset=LOCALITY_METRICS)
    curated = curated.drop_duplicates(["city", "locality"], keep="first")
    curated["data_source"] = "curated"
    curated["data_confidence"] = 1.0

    property_localities = properties[["city", "locality"]].drop_duplicates()
    covered = curated[["city", "locality"]]
    missing_pairs = property_localities.merge(
        covered, on=["city", "locality"], how="left", indicator=True
    )
    missing_pairs = missing_pairs[missing_pairs["_merge"].eq("left_only")][["city", "locality"]]

    global_medians = curated[LOCALITY_METRICS].median(numeric_only=True)
    city_medians = curated.groupby("city")[LOCALITY_METRICS].median(numeric_only=True)
    estimates = []
    for row in missing_pairs.itertuples(index=False):
        metrics = city_medians.loc[row.city] if row.city in city_medians.index else global_medians
        record = {"city": row.city, "locality": row.locality}
        for column in LOCALITY_METRICS:
            record[column] = float(metrics[column])
        record["data_source"] = "city_estimate" if row.city in city_medians.index else "global_estimate"
        record["data_confidence"] = 0.35 if row.city in city_medians.index else 0.20
        estimates.append(record)

    complete = pd.concat([curated, pd.DataFrame(estimates)], ignore_index=True)
    complete["nearest_metro_km"] = complete["nearest_metro_km"].round(2)
    complete["rent_yield_pct"] = complete["rent_yield_pct"].round(2)
    for column in ("schools_nearby", "hospitals_nearby", "traffic_index", "connectivity_score"):
        complete[column] = complete[column].round().astype(int)
    complete = complete.sort_values(["city", "locality"]).reset_index(drop=True)

    covered_pairs = set(map(tuple, complete[["city", "locality"]].to_numpy()))
    required_pairs = set(map(tuple, property_localities.to_numpy()))
    report = {
        "curated_rows": len(curated),
        "estimated_rows": len(estimates),
        "total_rows": len(complete),
        "coverage_pct": round(len(required_pairs & covered_pairs) / len(required_pairs) * 100, 2),
        "estimation_policy": (
            "Missing localities inherit medians from curated rows in the same city; "
            "cities without curated rows use global medians. Estimates are explicitly labelled."
        ),
    }
    return complete, report


def build_database(properties: pd.DataFrame, localities: pd.DataFrame, target: Path) -> None:
    temporary = target.with_suffix(".tmp.db")
    temporary.unlink(missing_ok=True)
    with sqlite3.connect(temporary) as connection:
        properties.to_sql("properties", connection, index=False, if_exists="replace")
        localities.to_sql("locality_scores", connection, index=False, if_exists="replace")
        connection.executescript(
            """
            CREATE UNIQUE INDEX idx_properties_id ON properties(property_id);
            CREATE INDEX idx_properties_search
              ON properties(city, bhk, price, locality);
            CREATE INDEX idx_properties_fingerprint
              ON properties(listing_fingerprint);
            CREATE UNIQUE INDEX idx_locality_lookup
              ON locality_scores(city, locality);

            CREATE VIEW property_search AS
            SELECT p.*, l.nearest_metro_km, l.schools_nearby,
                   l.hospitals_nearby, l.traffic_index,
                   l.connectivity_score, l.rent_yield_pct,
                   l.data_source AS locality_data_source,
                   l.data_confidence AS locality_data_confidence
            FROM properties p
            JOIN locality_scores l
              ON p.city = l.city AND p.locality = l.locality;
            """
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PropWise Phase 1 data assets.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    properties, property_report = prepare_properties(PROPERTY_SOURCE)
    localities, locality_report = prepare_localities(properties, LOCALITY_SOURCE)
    report = {
        "phase": 1,
        "properties": property_report,
        "localities": locality_report,
    }
    if not args.validate_only:
        properties.to_csv(PROPERTY_OUTPUT, index=False)
        localities.to_csv(LOCALITY_OUTPUT, index=False)
        build_database(properties, localities, DATABASE_OUTPUT)
        REPORT_OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
