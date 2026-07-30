import sys
import json
import pandas as pd
from agents.config import REPORTS_DIR

# Ensure the workspace directory is in the python path
# Import other agents
try:
    from agents.discovery_agent import discover_properties
    from agents.price_agent import analyze_price
    from agents.locality_agent import get_locality_insights
    from agents.roi_agent import analyze_roi
except ImportError as e:
    print(f"Import Error: Make sure peer agents are located in the agents/ folder. Details: {e}", file=sys.stderr)

def generate_property_report(
    selected_property, price_analysis, locality_analysis, roi_analysis,
    financial_analysis=None,
):
    """
    Generates a consolidated investment report from peer agent outputs.

    Parameters:
    - selected_property (dict): Details of the evaluated property.
    - price_analysis (dict): Pricing analysis results.
    - locality_analysis (dict): Locality infrastructure insights.
    - roi_analysis (dict): ROI analysis results.

    Returns:
    - dict: Consolidated report dictionary.
            Returns None if an error occurs.
    """
    try:
        # 1. Validation
        if not selected_property or not price_analysis or not locality_analysis or not roi_analysis:
            raise ValueError("All inputs (selected_property, price_analysis, locality_analysis, roi_analysis) must be provided.")

        # 2. Determine final verdict based on ROI recommendation
        # Strong Buy -> Recommended Investment
        # Consider -> Moderate Opportunity
        # Avoid -> High Risk Investment
        recommendation = roi_analysis.get("recommendation", "")
        rec_clean = str(recommendation).strip().lower()

        if "strong buy" in rec_clean:
            final_verdict = "Recommended Investment"
        elif "consider" in rec_clean:
            final_verdict = "Moderate Opportunity"
        elif "avoid" in rec_clean:
            final_verdict = "High Risk Investment"
        else:
            final_verdict = "Unknown"

        # 3. Structure the report
        report = {
            "property_summary": selected_property,
            "price_summary": price_analysis,
            "locality_summary": locality_analysis,
            "roi_summary": roi_analysis,
            "financial_summary": financial_analysis or {},
            "final_verdict": final_verdict
        }

        # 4. Create reports folder if missing and save report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_file_path = REPORTS_DIR / "sample_report.json"

        with open(report_file_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Report saved successfully to {report_file_path}")
        return report

    except Exception as e:
        print(f"Error in generate_property_report: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("Testing Report Agent...")

    # Find a valid city/locality to test
    from agents.config import DATA_DIR
    locality_scores_path = DATA_DIR / "locality_scores.csv"
    if locality_scores_path.exists():
        df_scores = pd.read_csv(locality_scores_path)
        if not df_scores.empty:
            # We select the first record for testing
            first_row = df_scores.iloc[0]
            test_city = first_row["city"]
            test_locality = first_row["locality"]

            print(f"\n1. Disclosing properties in '{test_locality}', '{test_city}' using Discovery Agent...")
            discovered_list = discover_properties(city=test_city, max_budget=10000000, locality=test_locality)

            if discovered_list:
                selected_property = discovered_list[0]
                print(f"   Successfully discovered property ID: {selected_property['property_id']}")

                # 2. Call Price Agent
                print("\n2. Calling Price Agent...")
                price_analysis = analyze_price(selected_property)

                # 3. Call Locality Agent
                print("\n3. Calling Locality Agent...")
                locality_analysis = get_locality_insights(test_city, test_locality)

                # 4. Call ROI Agent
                print("\n4. Calling ROI Agent...")
                roi_analysis = analyze_roi(selected_property, price_analysis, locality_analysis)

                # 5. Generate Report
                print("\n5. Generating Investment Report...")
                report = generate_property_report(selected_property, price_analysis, locality_analysis, roi_analysis)

                if report:
                    print("\n" + "="*50)
                    print("PRETTY PRINTED INVESTMENT REPORT")
                    print("="*50)
                    print(json.dumps(report, indent=4))
                    print("="*50)
                else:
                    print("Failed to generate report.")
            else:
                print(f"No properties found for {test_locality}, {test_city} with max budget 10000000.")
        else:
            print("Locality scores file is empty.")
    else:
        print(f"Required file {locality_scores_path} is missing. Please run generation scripts first.")
