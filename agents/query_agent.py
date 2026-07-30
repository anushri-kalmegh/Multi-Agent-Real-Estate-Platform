import sys
import re

# Import the property pipeline orchestrator
try:
    from agents.pipeline import run_property_pipeline
except ImportError as e:
    print(f"Import Error: Make sure pipeline.py is in the agents/ folder. Details: {e}", file=sys.stderr)

def parse_query(user_query):
    """
    Parses a natural language query to extract city, bhk, and max_budget.

    Parameters:
    - user_query (str): The raw query input by the user.

    Returns:
    - dict: Parsed parameters containing city, bhk, and max_budget.
    """
    if not user_query or not isinstance(user_query, str):
        raise ValueError("User query must be a non-empty string.")

    query_clean = user_query.strip()
    query_lower = query_clean.lower()

    # 1. Extract City (case-insensitive search from matching supported list)
    supported_cities = [
        'Pune', 'Bangalore', 'Chennai', 'Delhi', 'Hyderabad',
        'Kolkata', 'Mumbai', 'Thane', 'Kalyan', 'Agartala',
        'Palghar', 'Bhiwandi', 'Gurgaon', 'Nagpur'
    ]
    city = None
    for c in supported_cities:
        if c.lower() in query_lower:
            city = c
            break

    # 2. Extract BHK (looking for digits followed by BHK, e.g. "3 BHK", "2bhk")
    bhk_match = re.search(r'(\d+)\s*bhk', query_clean, re.IGNORECASE)
    bhk = int(bhk_match.group(1)) if bhk_match else None

    # 3. Extract Budget (handling lakh/lakhs/crore/crores and raw numeric values)
    max_budget = None
    # Match decimal or integer numbers followed by a multiplier unit
    budget_match = re.search(r'(\d+(?:\.\d+)?)\s*(lakhs?|crores?|cr?)', query_clean, re.IGNORECASE)
    if budget_match:
        value = float(budget_match.group(1))
        unit = budget_match.group(2).lower()
        if 'lakh' in unit:
            max_budget = int(value * 100_000)
        elif 'crore' in unit or 'cr' in unit:
            max_budget = int(value * 10_000_000)
    else:
        # Fallback: check for large plain numbers (at least 5 digits)
        plain_match = re.search(r'\b(\d{5,10})\b', query_clean)
        if plain_match:
            max_budget = int(plain_match.group(1))

    # 4. Validations
    if not city:
        raise ValueError("Could not extract a supported city from the query.")
    if max_budget is None or max_budget <= 0:
        raise ValueError("Could not extract a valid budget from the query.")

    purpose = "investment" if any(
        word in query_lower for word in ("invest", "investment", "roi", "rental", "rent", "yield")
    ) else "self-use"

    return {
        "city": city,
        "bhk": bhk,
        "max_budget": max_budget,
        "purpose": purpose,
    }

def run_query(user_query):
    """
    Parses a user query and runs the property investment pipeline.

    Parameters:
    - user_query (str): The raw query input by the user.

    Returns:
    - dict: Pipeline execution results containing best_property, score, recommendation, and report.
            Returns None if processing fails.
    """
    try:
        print(f"Query Agent: Parsing user query -> '{user_query}'")
        parsed = parse_query(user_query)
        print(f"Query Agent: Extracted Parameters -> {parsed}")

        # Execute pipeline
        result = run_property_pipeline(
            city=parsed["city"],
            max_budget=parsed["max_budget"],
            bhk=parsed["bhk"],
            purpose=parsed["purpose"],
        )
        if result:
            result["query_understanding"] = parsed
        return result

    except ValueError as e:
        return {"status": "needs_input", "message": str(e)}
    except Exception as e:
        print(f"Query Agent Error: {e}", file=sys.stderr)
        return {"status": "error", "message": "The agent pipeline could not process this query."}

if __name__ == "__main__":
    print("Testing Query Agent...")

    # Target query for demonstration
    test_query = "Find me a 3 BHK in Mumbai under 1 crore"

    print("-" * 50)
    result = run_query(test_query)
    print("-" * 50)

    if result:
        print("\n" + "="*50)
        print("QUERY AGENT SUCCESSFUL EXECUTION")
        print("="*50)
        print(f"Target Query          : '{test_query}'")
        print(f"Top Property Selected : {result['best_property']['property_id']} in {result['best_property']['locality']}")
        print(f"Listing Price         : INR {result['best_property']['price']:,}")
        print(f"Investment Score      : {result['investment_score']} / 10")
        print(f"Recommendation        : {result['recommendation']}")
        print(f"Final Verdict         : {result['report']['final_verdict']}")
        print("="*50)
    else:
        print("\nQuery execution failed.")
