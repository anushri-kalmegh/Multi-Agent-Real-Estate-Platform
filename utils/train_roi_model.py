"""Train the Phase 4 proxy appreciation scenario model.

The target is a documented heuristic because the repository does not contain
historical resale transactions. This artifact must not be described as a
historically validated forecast.
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.config import DATABASE_PATH, ROI_MODEL_METADATA_PATH, ROI_MODEL_PATH


FEATURES = [
    "nearest_metro_km", "schools_nearby", "hospitals_nearby",
    "traffic_index", "connectivity_score", "rent_yield_pct",
    "locality_data_confidence",
]


def proxy_target(df: pd.DataFrame) -> np.ndarray:
    raw = (
        2.4
        + df["connectivity_score"] * 0.28
        + df["rent_yield_pct"] * 0.48
        + df["schools_nearby"] * 0.035
        + df["hospitals_nearby"] * 0.04
        - df["traffic_index"] * 0.10
        - df["nearest_metro_km"] * 0.13
    )
    confidence_adjusted = 5.0 + (raw - 5.0) * df["locality_data_confidence"]
    return confidence_adjusted.clip(2.0, 10.0).to_numpy()


def main():
    with sqlite3.connect(DATABASE_PATH) as connection:
        df = pd.read_sql_query(
            f"SELECT {', '.join(FEATURES)} FROM property_search", connection
        ).dropna()
    # Locality signals repeat across listings; deduplicate feature rows so large
    # localities do not dominate the proxy model.
    df = df.drop_duplicates(FEATURES).reset_index(drop=True)
    target = proxy_target(df)
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES], target, test_size=0.2, random_state=42
    )
    model = HistGradientBoostingRegressor(
        max_iter=180, max_depth=5, learning_rate=0.06, random_state=42
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    metadata = {
        "model_name": "PropWise proxy appreciation scenario model",
        "version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": len(df),
        "features": FEATURES,
        "target_type": "derived_proxy_not_historical",
        "target_range_pct": [2.0, 10.0],
        "mae_pct_points": round(float(mean_absolute_error(y_test, prediction)), 4),
        "r2_on_proxy_target": round(float(r2_score(y_test, prediction)), 4),
        "warning": (
            "The target is derived from locality and rental signals. It is not "
            "trained on historical resale prices and is for scenario analysis only."
        ),
    }
    ROI_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ROI_MODEL_PATH.open("wb") as handle:
        pickle.dump({"model": model, "features": FEATURES, "metadata": metadata}, handle)
    ROI_MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
