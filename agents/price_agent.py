import os
import pandas as pd
import numpy as np
from agents.config import DATA_DIR, MODEL_DIR
from agents.data_repository import load_properties

data_path = DATA_DIR / "properties_clean.csv"

def analyze_price(selected_property):
    """
    Analyzes the price of a selected property compared to comparable listings in the dataset.

    Parameters:
    - selected_property (dict): Dictionary representing the property details.

    Returns:
    - dict: Pricing analysis results containing fair_price_per_sqft, predicted_price,
            actual_price, deviation_pct, and verdict.
    """
    try:
        # Load the normalized SQLite data layer (CSV fallback is handled by repository).
        df = load_properties()

        # Extract fields
        city = selected_property.get('city')
        locality = selected_property.get('locality')
        bhk = selected_property.get('bhk')
        area_sqft = selected_property.get('area_sqft')
        actual_price = selected_property.get('price')
        prop_id = selected_property.get('property_id')

        if not city or bhk is None or not area_sqft or actual_price is None:
            raise ValueError("selected_property must contain 'city', 'bhk', 'area_sqft', and 'price'.")

        city_lower = str(city).strip().lower()
        locality_lower = str(locality).strip().lower() if locality else ""

        # 2. Find comparable properties: same city, same locality, same bhk
        # Exclude the selected property itself if it's already in the dataset
        comp_df = df[
            (df['city'].str.lower() == city_lower) &
            (df['locality'].str.lower() == locality_lower) &
            (df['bhk'] == bhk)
        ]

        if prop_id:
            comp_df = comp_df[comp_df['property_id'] != prop_id]

        comparable_count = len(comp_df)

        # 3. If comparable properties < 3: use city + bhk average
        if comparable_count < 3:
            comp_df = df[
                (df['city'].str.lower() == city_lower) &
                (df['bhk'] == bhk)
            ]
            if prop_id:
                comp_df = comp_df[comp_df['property_id'] != prop_id]
            comparable_count = len(comp_df)

        if comp_df.empty:
            raise ValueError(f"No comparable properties found in city '{city}' with BHK {bhk} even after fallback.")

        # 4. Outlier Analysis & IQR Cleaning
        comparable_count_before = len(comp_df)
        old_median = float(comp_df['price_per_sqft'].median())

        Q1 = comp_df['price_per_sqft'].quantile(0.25)
        Q3 = comp_df['price_per_sqft'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        clean_comp_df = comp_df[(comp_df['price_per_sqft'] >= lower_bound) & (comp_df['price_per_sqft'] <= upper_bound)]
        if clean_comp_df.empty:
            clean_comp_df = comp_df

        comparable_count_after = len(clean_comp_df)
        new_median = float(clean_comp_df['price_per_sqft'].median())

        # Print requested debug logging
        print(f"comparable_count_before: {comparable_count_before}")
        print(f"comparable_count_after: {comparable_count_after}")
        print(f"old_median: {old_median}")
        print(f"new_median: {new_median}")

        # 5. Calculate fair_price_per_sqft, predicted_price, deviation_pct using ML model if available
        model_path = MODEL_DIR / "price_predictor.pkl"
        use_ml = False
        if os.path.exists(model_path):
            try:
                import pickle
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                use_ml = True
            except Exception as ml_err:
                print(f"Warning: Failed to load ML model, falling back to median logic: {ml_err}")

        if use_ml:
            try:
                # Prepare features DataFrame for ML model
                test_df = pd.DataFrame([{
                    "city": city,
                    "locality": locality if locality else "Unknown",
                    "bhk": float(bhk),
                    "area_sqft": float(area_sqft)
                }])

                # Predict price_per_sqft
                predicted_pps = float(model.predict(test_df)[0])

                fair_price_per_sqft = predicted_pps
                predicted_price = fair_price_per_sqft * area_sqft
                deviation_pct = ((actual_price - predicted_price) / predicted_price) * 100
                print(f"ML Model prediction used: {predicted_pps:.2f} INR/sqft")
            except Exception as pred_err:
                print(f"Error predicting with ML model, falling back: {pred_err}")
                use_ml = False

        if not use_ml:
            fair_price_per_sqft = new_median
            predicted_price = fair_price_per_sqft * area_sqft
            deviation_pct = ((actual_price - predicted_price) / predicted_price) * 100

        # 6. Determine Verdict
        if deviation_pct < -10:
            verdict = "Undervalued"
        elif -10 <= deviation_pct <= 10:
            verdict = "Fairly Priced"
        else:
            verdict = "Overpriced"

        # Calculate confidence score
        confidence_score = min(comparable_count_after * 5, 100)

        # 7. Return standard dict
        return {
            "comparable_properties": int(comparable_count_after),
            "confidence_score": int(confidence_score),
            "fair_price_per_sqft": float(fair_price_per_sqft),
            "predicted_price": float(predicted_price),
            "actual_price": float(actual_price),
            "deviation_pct": float(deviation_pct),
            "verdict": verdict
        }

    except Exception as e:
        print(f"Error in analyze_price: {e}")
        return None

if __name__ == "__main__":
    print("Testing Price Agent...")

    # Test using a selected Pune property from properties_clean.csv
    if os.path.exists(data_path):
        df_clean = pd.read_csv(data_path)
        pune_df = df_clean[df_clean['city'] == 'Pune']

        if not pune_df.empty:
            test_row = pune_df.iloc[0]
            selected_property = test_row.to_dict()

            print("\nSelected Property for Evaluation:")
            for k, v in selected_property.items():
                print(f"  {k}: {v}")

            analysis = analyze_price(selected_property)
            print("\nAnalysis Result:")
            if analysis:
                for k, v in analysis.items():
                    print(f"  {k}: {v}")
            else:
                print("  Analysis failed.")
        else:
            print("No Pune properties found in dataset.")
    else:
        print(f"Cleaned dataset not found at {data_path}.")
