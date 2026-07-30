import os
import sys
import pandas as pd
from agents.config import DATA_DIR

# Ensure the workspace directory is in the python path
# Import functions from peer agents
try:
    from agents.price_agent import analyze_price
    from agents.locality_agent import get_locality_insights
except ImportError as e:
    print(f"Import Error: Make sure peer agents are located in the agents/ folder. Details: {e}", file=sys.stderr)

def analyze_roi(selected_property, price_analysis, locality_analysis):
    """
    Performs Return on Investment (ROI) and risk analysis on a property.

    Parameters:
    - selected_property (dict): Details of the property. Must contain 'price'.
    - price_analysis (dict): Output from price_agent. Must contain 'verdict' and 'actual_price'.
    - locality_analysis (dict): Output from locality_agent. Must contain 'locality_score' and 'rent_yield_pct'.

    Returns:
    - dict: ROI analysis containing expected_monthly_rent, annual_rent, rental_yield_pct,
            investment_score, risk_level, and recommendation.
            Returns None if an error occurs.
    """
    try:
        # 1. Input Validation
        if not selected_property or not price_analysis or not locality_analysis:
            raise ValueError("All inputs (selected_property, price_analysis, locality_analysis) must be provided.")

        # Get actual price
        actual_price = price_analysis.get("actual_price")
        if actual_price is None:
            actual_price = selected_property.get("price")

        if actual_price is None or actual_price <= 0:
            raise ValueError("Property price must be a positive number.")

        # Get rent yield percentage
        rent_yield_pct = locality_analysis.get("rent_yield_pct")
        if rent_yield_pct is None or rent_yield_pct < 0:
            raise ValueError("locality_analysis must contain a non-negative 'rent_yield_pct'.")

        # Get locality score
        locality_score = locality_analysis.get("locality_score")
        if locality_score is None or not (1 <= locality_score <= 10):
            raise ValueError("locality_analysis must contain a 'locality_score' between 1 and 10.")

        # Get price verdict
        verdict = price_analysis.get("verdict")
        if not verdict:
            raise ValueError("price_analysis must contain a 'verdict'.")

        # 2. ROI Calculations
        # Expected monthly rent: (actual_price * rent_yield_pct / 100) / 12
        expected_monthly_rent = (actual_price * (rent_yield_pct / 100.0)) / 12.0
        expected_monthly_rent = round(expected_monthly_rent, 2)

        # Annual rent: expected_monthly_rent * 12
        annual_rent = round(expected_monthly_rent * 12.0, 2)

        # 3. Component Scores
        # Price Score: Undervalued = 10, Fairly Priced = 7, Overpriced = 4
        verdict_clean = str(verdict).strip().lower()
        if "undervalued" in verdict_clean:
            price_score = 10.0
        elif "fairly priced" in verdict_clean:
            price_score = 7.0
        elif "overpriced" in verdict_clean:
            price_score = 4.0
        else:
            print(f"Warning: Unknown verdict '{verdict}'. Defaulting price score to 5.0.")
            price_score = 5.0

        # Rental Yield Score:
        # >= 4.5 -> 10
        # >= 4.0 -> 8
        # >= 3.5 -> 7
        # >= 3.0 -> 6
        # else -> 5
        if rent_yield_pct >= 4.5:
            rental_yield_score = 10.0
        elif rent_yield_pct >= 4.0:
            rental_yield_score = 8.0
        elif rent_yield_pct >= 3.5:
            rental_yield_score = 7.0
        elif rent_yield_pct >= 3.0:
            rental_yield_score = 6.0
        else:
            rental_yield_score = 5.0

        # 4. Investment Score (Weighted: 40% Locality, 30% Price, 30% Rental Yield)
        investment_score = (locality_score * 0.40) + (price_score * 0.30) + (rental_yield_score * 0.30)
        investment_score = round(investment_score, 2)

        # 5. Determine Risk Level
        # investment_score >= 8 -> Low Risk
        # investment_score >= 6 -> Medium Risk
        # else -> High Risk
        if investment_score >= 8.0:
            risk_level = "Low Risk"
        elif investment_score >= 6.0:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        # 6. Determine Recommendation
        # investment_score >= 8 -> Strong Buy
        # investment_score >= 6 -> Consider
        # else -> Avoid
        if investment_score >= 8.0:
            recommendation = "Strong Buy"
        elif investment_score >= 6.0:
            recommendation = "Consider"
        else:
            recommendation = "Avoid"

        # 7. Return standard dict
        return {
            "expected_monthly_rent": float(expected_monthly_rent),
            "annual_rent": float(annual_rent),
            "rental_yield_pct": float(rent_yield_pct),
            "investment_score": float(investment_score),
            "risk_level": risk_level,
            "recommendation": recommendation
        }

    except Exception as e:
        print(f"Error in analyze_roi: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("Testing ROI Agent...")

    properties_clean_path = DATA_DIR / "properties_clean.csv"
    locality_scores_path = DATA_DIR / "locality_scores.csv"

    if os.path.exists(properties_clean_path) and os.path.exists(locality_scores_path):
        # 1. Load clean properties dataset
        df_properties = pd.read_csv(properties_clean_path)
        # Load locality scores dataset to find matching properties
        df_scores = pd.read_csv(locality_scores_path)

        # Clean columns to merge/match
        df_properties['city_lower'] = df_properties['city'].str.strip().str.lower()
        df_properties['locality_lower'] = df_properties['locality'].str.strip().str.lower()

        df_scores['city_lower'] = df_scores['city'].str.strip().str.lower()
        df_scores['locality_lower'] = df_scores['locality'].str.strip().str.lower()

        # Find properties in properties_clean.csv that exist in locality_scores.csv
        merged = df_properties.merge(df_scores, on=['city_lower', 'locality_lower'], suffixes=('', '_score'))

        if not merged.empty:
            # Select the first property
            test_row = merged.iloc[0]
            selected_property = {
                "property_id": test_row["property_id"],
                "city": test_row["city"],
                "locality": test_row["locality"],
                "bhk": int(test_row["bhk"]),
                "area_sqft": float(test_row["area_sqft"]),
                "price": float(test_row["price"])
            }

            print("\n" + "="*40)
            print("1. Selected Property Details:")
            print("="*40)
            for k, v in selected_property.items():
                print(f"  {k}: {v}")
            print("="*40)

            # 2. Call price_agent
            print("\n2. Calling Price Agent...")
            price_analysis = analyze_price(selected_property)
            if price_analysis:
                for k, v in price_analysis.items():
                    print(f"  {k}: {v}")
            else:
                print("  Failed to analyze price.")
                sys.exit(1)

            # 3. Call locality_agent
            print("\n3. Calling Locality Agent...")
            locality_analysis = get_locality_insights(selected_property["city"], selected_property["locality"])
            if locality_analysis:
                for k, v in locality_analysis.items():
                    print(f"  {k}: {v}")
            else:
                print("  Failed to analyze locality.")
                sys.exit(1)

            # 4. Call ROI Agent
            print("\n4. Calling ROI Agent...")
            roi_analysis = analyze_roi(selected_property, price_analysis, locality_analysis)
            if roi_analysis:
                print("\n" + "="*40)
                print("5. Final ROI Analysis Result:")
                print("="*40)
                for k, v in roi_analysis.items():
                    print(f"  {k}: {v}")
                print("="*40)
            else:
                print("  Failed to analyze ROI.")
                sys.exit(1)
        else:
            print("Could not find any properties in properties_clean.csv matching localities in locality_scores.csv.")
    else:
        print("Required datasets are missing. Run previous extraction/generation scripts first.")
