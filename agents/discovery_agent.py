import os
import re
import pandas as pd
import numpy as np
from agents.config import DATA_DIR
from agents.data_repository import search_properties

data_path = DATA_DIR / "properties_clean.csv"

def clean_locality_name(val):
    """
    Standardizes input locality name to match clean data standards.
    """
    if pd.isna(val):
        return ""
    val_str = str(val).strip()
    val_str = re.sub(r'\s+', ' ', val_str)
    return val_str.title()

def discover_properties(city, max_budget, bhk=None, locality=None, top_n=10):
    """
    Discovers properties in the master dataset based on city, budget, BHK, and locality.

    Parameters:
    - city (str): Name of the city to search in.
    - max_budget (float): Maximum budget for the property in INR.
    - bhk (int, optional): Exact BHK count.
    - locality (str, optional): Locality name (case-insensitive, exact match).
    - top_n (int): Number of top results to return. Default is 10.

    Returns:
    - list of dicts: Standardized property listings matching the criteria.
    """
    try:
        if not city:
            raise ValueError("City name must be provided.")
        if max_budget is None or max_budget <= 0:
            raise ValueError("max_budget must be a positive number.")
        df_filtered = search_properties(
            city=city,
            max_budget=max_budget,
            bhk=bhk,
            locality=clean_locality_name(locality) if locality else None,
            top_n=top_n,
        )

        # If no properties match, return an empty list
        if df_filtered.empty:
            return []

        # Select target columns
        target_cols = ['property_id', 'city', 'locality', 'bhk', 'area_sqft', 'price', 'price_per_sqft']
        df_top = df_filtered.head(top_n)[target_cols]

        # Convert NaN values to None to keep valid JSON/dict formats
        df_top = df_top.replace({np.nan: None})

        # Convert BHK and area_sqft to standard types if needed
        # Convert to dictionary format
        results = df_top.to_dict(orient='records')
        return results

    except Exception as e:
        print(f"Error discovering properties: {e}")
        return []

if __name__ == "__main__":
    results = discover_properties(
        city="Pune",
        max_budget=8000000,
        bhk=2
    )

    print(results[:5])
