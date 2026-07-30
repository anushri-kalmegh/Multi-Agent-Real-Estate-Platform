import sys

# Import other agents
try:
    from agents.discovery_agent import discover_properties
    from agents.price_agent import analyze_price
    from agents.locality_agent import get_locality_insights
    from agents.roi_agent import analyze_roi
    from agents.financial_agent import analyze_financials
    from agents.report_agent import generate_property_report
except ImportError as e:
    print(f"Import Error: Make sure all peer agents are in the agents/ folder. Details: {e}", file=sys.stderr)

def run_property_pipeline(
    city, max_budget, bhk=None, locality=None, purpose="self-use",
    recommendation_count=4, financial_assumptions=None,
):
    """
    Master orchestrator connecting all real estate agents to identify and analyze
    the best property investment opportunity.

    Parameters:
    - city (str): The search city.
    - max_budget (float): Maximum price budget.
    - bhk (int, optional): Exact BHK.
    - locality (str, optional): Target locality.

    Returns:
    - dict: Orchestrator results containing best_property details, investment_score,
            recommendation, and the full consolidated report dict.
            Returns None if an error occurs.
    """
    try:
        # 1. Validation
        if not city or max_budget is None or max_budget <= 0:
            raise ValueError("A valid city and positive max_budget must be provided.")

        print(f"Pipeline: Discovering properties in '{city}' under budget {max_budget} (BHK: {bhk}, Locality: {locality})...")

        # 2. Call discover_properties (fetches top 10 matching properties)
        # Pull a wider discovery pool because locality intelligence does not cover
        # every raw listing; stop once enough fully evaluated options are available.
        properties = discover_properties(city=city, max_budget=max_budget, bhk=bhk, locality=locality, top_n=100)

        if not properties:
            print("Pipeline: No properties matching the criteria were found.")
            return None

        print(f"Pipeline: Discovered {len(properties)} properties. Evaluating each...")

        # 3. Evaluate each property
        evaluated_opportunities = []
        for prop in properties:
            prop_id = prop.get("property_id")
            prop_locality = prop.get("locality")

            # Step A: Analyze Price
            price_analysis = analyze_price(prop)
            if not price_analysis:
                print(f"  [Skip] Price analysis failed for property {prop_id}.")
                continue

            # Step B: Analyze Locality (must exist in locality_scores.csv)
            locality_analysis = get_locality_insights(city, prop_locality)
            if not locality_analysis:
                # Some properties might not have locality score data
                print(f"  [Skip] Locality '{prop_locality}' has no score data in our database.")
                continue

            # Step C: Analyze ROI
            roi_analysis = analyze_roi(prop, price_analysis, locality_analysis)
            if not roi_analysis:
                print(f"  [Skip] ROI analysis failed for property {prop_id}.")
                continue
            financial_analysis = analyze_financials(
                prop, locality_analysis, financial_assumptions
            )

            evaluated_opportunities.append({
                "property": prop,
                "price_analysis": price_analysis,
                "locality_analysis": locality_analysis,
                "roi_analysis": roi_analysis,
                "financial_analysis": financial_analysis,
                "investment_score": roi_analysis["investment_score"],
                "recommendation": roi_analysis["recommendation"]
            })
            if len(evaluated_opportunities) >= 12:
                break

        if not evaluated_opportunities:
            raise ValueError("Pipeline: No properties could be fully evaluated. Ensure localities are present in the scores database.")

        # Investment searches emphasize ROI; self-use searches give locality more influence.
        for opportunity in evaluated_opportunities:
            roi = opportunity["roi_analysis"]["investment_score"]
            locality_score = opportunity["locality_analysis"]["locality_score"]
            price_deviation = abs(opportunity["price_analysis"]["deviation_pct"])
            price_fit = max(0.0, 10.0 - min(price_deviation, 50.0) / 5.0)
            if purpose == "investment":
                rank_score = (roi * 0.60) + (locality_score * 0.25) + (price_fit * 0.15)
            else:
                rank_score = (roi * 0.35) + (locality_score * 0.45) + (price_fit * 0.20)
            opportunity["rank_score"] = round(rank_score, 2)

        # 4. Rank fully evaluated properties and retain 3-4 options for comparison.
        evaluated_opportunities.sort(key=lambda x: x["rank_score"], reverse=True)
        top_opportunities = evaluated_opportunities[:max(1, recommendation_count)]

        # 5. Select the best property
        best_opp = top_opportunities[0]
        best_prop = best_opp["property"]
        print(f"Pipeline: Selected best property {best_prop['property_id']} in '{best_prop['locality']}' with Investment Score: {best_opp['investment_score']}")

        recommendations = []
        for index, opportunity in enumerate(top_opportunities, start=1):
            recommendations.append({
                "rank": index,
                **opportunity,
                "report": generate_property_report(
                    opportunity["property"],
                    opportunity["price_analysis"],
                    opportunity["locality_analysis"],
                    opportunity["roi_analysis"],
                    opportunity["financial_analysis"],
                ),
            })
        report = recommendations[0]["report"]

        # 7. Return summary plus every visible agent hand-off.
        return {
            "status": "ok",
            "best_property": best_prop,
            "investment_score": best_opp["investment_score"],
            "recommendation": best_opp["recommendation"],
            "report": report,
            "recommendations": recommendations,
            "agent_trace": [
                {"agent": "Query Understanding Agent", "status": "complete",
                 "output": f"{bhk or 'Any'} BHK · {city} · budget up to ₹{max_budget:,.0f} · {purpose}"},
                {"agent": "Discovery Agent", "status": "complete",
                 "output": f"{len(properties)} matching listings shortlisted from the property dataset"},
                {"agent": "Price Agent", "status": "complete",
                 "output": f"ML/fair-value predictions completed for {len(evaluated_opportunities)} eligible listings"},
                {"agent": "Locality Agent", "status": "complete",
                 "output": "Connectivity, metro, schools, hospitals and rental-yield signals scored"},
                {"agent": "ROI Agent", "status": "complete",
                 "output": "Expected rent, yield, investment score and risk predicted"},
                {"agent": "Financial Scenario Agent", "status": "complete",
                 "output": (
                     "EMI, costs, net cash flow and conservative/base/optimistic "
                     f"{(financial_assumptions or {}).get('holding_period_years', 5)}-year returns calculated"
                 )},
                {"agent": "Recommendation Agent", "status": "complete",
                 "output": f"Top {len(recommendations)} properties ranked for {purpose}"},
                {"agent": "Report Agent", "status": "complete",
                 "output": "Comparison and final recommendation generated"},
            ],
        }

    except Exception as e:
        print(f"Pipeline Error: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    print("Testing Master Orchestrator Pipeline...")

    # Run the test pipeline
    result = run_property_pipeline(
        city="Mumbai",
        max_budget=10000000,
        bhk=3
    )

    if result:
        print("\n" + "="*50)
        print("PIPELINE EXECUTION RESULTS")
        print("="*50)
        print(f"Top Property Selected : {result['best_property']['property_id']} in {result['best_property']['locality']}")
        print(f"Price                 : INR {result['best_property']['price']:,}")
        print(f"Investment Score      : {result['investment_score']} / 10")
        print(f"Recommendation        : {result['recommendation']}")
        print(f"Final Verdict         : {result['report']['final_verdict']}")
        print("="*50)
    else:
        print("\nPipeline execution failed.")
