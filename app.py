"""PropWise AI conversational multi-agent property advisor."""

from pathlib import Path
import json
import sys
import uuid

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.phase2_orchestrator import run_conversation
from agents.llm_service import groq_enabled
from agents.app_logging import get_logger
from agents.app_repository import (
    add_favourite, authenticate, create_session, create_user, list_favourites,
    list_reports, revoke_session, save_message, save_report, save_search,
    clear_conversation_history,
)

logger = get_logger()


st.set_page_config(page_title="PropWise AI", page_icon="🏡", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at 10% 0%, #eef6ff 0, #f8fafc 28%, #f8fafc 100%);}
    .stMain, .stMainBlockContainer {color:#0f172a;}
    .stMain p, .stMain li, .stMain label, .stMain h1, .stMain h2, .stMain h3,
    .stMain h4, .stMain span:not([data-testid="stIconMaterial"]) {color:#0f172a;}
    [data-testid="stSidebar"] {background: #0b1739;}
    [data-testid="stSidebar"] * {color: #e5ecff;}
    [data-testid="stSidebar"] input {
        color:#0f172a !important;background:#ffffff !important;
        -webkit-text-fill-color:#0f172a !important;
    }
    [data-testid="stSidebar"] input::placeholder {
        color:#64748b !important;opacity:1;
    }
    [data-testid="stSidebar"] [data-baseweb="tab-list"] {
        background:#13234b;border-radius:10px;padding:.2rem;
    }
    [data-testid="stSidebar"] button[data-baseweb="tab"] {
        color:#cbd5e1 !important;
    }
    [data-testid="stSidebar"] button[data-baseweb="tab"][aria-selected="true"] {
        color:#ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stForm"] {
        border:1px solid #334b7d;border-radius:12px;padding:.7rem;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] {
        background:#13234b !important;
        border:1px solid #334b7d !important;
        color:#f8fafc !important;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] summary,
    [data-testid="stSidebar"] div[data-testid="stExpander"] summary span,
    [data-testid="stSidebar"] div[data-testid="stExpander"] p,
    [data-testid="stSidebar"] div[data-testid="stExpander"] li,
    [data-testid="stSidebar"] div[data-testid="stExpander"] div,
    [data-testid="stSidebar"] div[data-testid="stExpander"] small {
        color:#f8fafc !important;
        -webkit-text-fill-color:#f8fafc !important;
    }
    [data-testid="stSidebar"] div[data-testid="stExpander"] svg {
        fill:#f8fafc !important;color:#f8fafc !important;
    }
    .hero {padding: 1.4rem 1.6rem; border-radius: 22px; color: white;
           background: linear-gradient(120deg,#172554,#1d4ed8 65%,#06b6d4);
           box-shadow: 0 18px 50px rgba(30,64,175,.20); margin-bottom: 1.2rem;}
    .hero h1 {margin:0; font-size:2.25rem}.hero p {margin:.35rem 0 0;color:#dbeafe}
    .agent-row {border:1px solid #dbe4f0;background:white;border-radius:12px;
                padding:.7rem .85rem;margin:.35rem 0;box-shadow:0 3px 14px rgba(15,23,42,.04)}
    .agent-ok {color:#047857;font-weight:700}.muted {color:#64748b;font-size:.88rem}
    .property-card {border:1px solid #dbe4f0;border-radius:16px;padding:1rem;
                    background:white;box-shadow:0 7px 24px rgba(15,23,42,.06);min-height:205px}
    .rank {display:inline-block;background:#dbeafe;color:#1d4ed8;border-radius:999px;
           padding:.18rem .55rem;font-weight:700;font-size:.78rem}
    .score {font-size:1.8rem;font-weight:800;color:#172554}
    .pill {display:inline-block;border-radius:999px;padding:.2rem .55rem;margin:.1rem;
           background:#ecfdf5;color:#047857;font-size:.78rem;font-weight:650}
    div[data-testid="stChatMessage"] {background:rgba(255,255,255,.78);border:1px solid #e2e8f0;
                                     border-radius:18px;padding:.35rem .7rem;margin:.55rem 0}
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li,
    div[data-testid="stChatMessage"] h1, div[data-testid="stChatMessage"] h2,
    div[data-testid="stChatMessage"] h3, div[data-testid="stChatMessage"] strong {
        color:#0f172a !important;
    }
    div[data-testid="stAlert"] {color:#0f172a !important;}
    div[data-testid="stAlert"] p, div[data-testid="stAlert"] span {
        color:#0f5132 !important;
    }
    div[data-testid="stExpander"] {
        background:#ffffff;border:1px solid #dbe4f0;border-radius:12px;color:#0f172a;
    }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary span {
        color:#0f172a !important;font-weight:700;
    }
    div[data-testid="stMetric"] {
        background:#ffffff;border:1px solid #dbe4f0;border-radius:12px;padding:.75rem;
    }
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {
        color:#0f172a !important;
    }
    button[data-baseweb="tab"] {color:#334155 !important;}
    button[data-baseweb="tab"][aria-selected="true"] {color:#1d4ed8 !important;}
    div[data-testid="stChatInput"] {background:#ffffff;border-top:1px solid #dbe4f0;}
    div[data-testid="stChatInput"] textarea {
        color:#0f172a !important;background:#ffffff !important;
    }
    div[data-testid="stChatInput"] textarea::placeholder {color:#64748b !important;opacity:1;}
    .stMain .hero h1, .stMain .hero p,
    .stMainBlockContainer .hero h1, .stMainBlockContainer .hero p {
        color:#ffffff !important;
        -webkit-text-fill-color:#ffffff !important;
    }
    .property-card h3, .property-card p, .property-card span {color:#0f172a;}
    .property-card .rank {color:#1d4ed8;}
    .property-card .pill {color:#047857;}
    .property-card .muted, .agent-row .muted {color:#64748b !important;}
    .agent-row .agent-ok {color:#047857 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)


def money(value):
    if value is None:
        return "N/A"
    value = float(value)
    if value >= 10_000_000:
        return f"₹{value / 10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"₹{value / 100_000:.1f} L"
    return f"₹{value:,.0f}"


def query_type(query):
    lowered = query.lower()
    legal_words = ("rera", "legal", "law", "act", "compliance", "registration", "promoter", "builder")
    property_words = ("bhk", "budget", "lakh", "crore", "property", "flat", "apartment", "house")
    if any(word in lowered for word in legal_words) and not any(word in lowered for word in property_words):
        return "legal"
    return "property"


def render_trace(result):
    with st.expander("🧠 Multi-agent execution trace", expanded=True):
        for step in result.get("agent_trace", []):
            st.markdown(
                f"""<div class="agent-row"><span class="agent-ok">✓ {step['agent']}</span><br>
                <span class="muted">{step['output']}</span></div>""",
                unsafe_allow_html=True,
            )


def render_property_card(item):
    prop = item["property"]
    price = item["price_analysis"]
    locality = item["locality_analysis"]
    roi = item["roi_analysis"]
    st.markdown(
        f"""<div class="property-card">
        <span class="rank">#{item['rank']} MATCH</span>
        <h3 style="margin:.55rem 0 .1rem">{prop['locality']}</h3>
        <div class="muted">{prop['city']} · {float(prop['bhk']):g} BHK · {float(prop['area_sqft']):,.0f} sq ft</div>
        <div style="margin:.65rem 0"><span class="score">{money(prop['price'])}</span></div>
        <span class="pill">{price['verdict']}</span>
        <span class="pill">{roi['recommendation']}</span>
        <p><b>{item['rank_score']}/10</b> match · <b>{roi['rental_yield_pct']:.2f}%</b> yield<br>
        <span class="muted">Predicted fair value: {money(price['predicted_price'])}</span></p>
        </div>""",
        unsafe_allow_html=True,
    )
    user = st.session_state.get("user")
    if user and st.button(
        "☆ Save to favourites",
        key=f"fav_{prop['property_id']}_{id(item)}",
        use_container_width=True,
    ):
        if add_favourite(user["id"], prop):
            st.success("Property saved.")
        else:
            st.info("This property is already in your favourites.")


def render_recommendations(result):
    understood = result.get("query_understanding", {})
    st.success(
        f"Found {len(result.get('recommendations', []))} ranked options for "
        f"{understood.get('purpose', 'your requirement')} in {understood.get('city', '')}."
    )
    render_trace(result)
    recommendations = result.get("recommendations", [])
    if not recommendations:
        st.warning("No fully evaluated recommendations were available.")
        return

    st.markdown("### Recommended properties")
    columns = st.columns(min(2, len(recommendations)))
    for index, item in enumerate(recommendations):
        with columns[index % len(columns)]:
            render_property_card(item)

    st.markdown("### Agent predictions")
    labels = [f"#{item['rank']} {item['property']['locality']}" for item in recommendations]
    tabs = st.tabs(labels)
    for tab, item in zip(tabs, recommendations):
        prop, price = item["property"], item["price_analysis"]
        locality, roi = item["locality_analysis"], item["roi_analysis"]
        financial = item.get("financial_analysis", {})
        with tab:
            a, b, c, d = st.columns(4)
            a.metric("Match score", f"{item['rank_score']}/10")
            b.metric("Investment score", f"{roi['investment_score']}/10")
            c.metric("Expected rent", f"{money(roi['expected_monthly_rent'])}/mo")
            d.metric("Locality score", f"{locality['locality_score']}/10")
            left, right = st.columns(2)
            with left:
                st.markdown("**🔎 Discovery Agent**")
                st.write(
                    f"Property `{prop['property_id']}` is a {float(prop['bhk']):g} BHK, "
                    f"{float(prop['area_sqft']):,.0f} sq ft listing in {prop['locality']}."
                )
                st.markdown("**📈 Price Agent prediction**")
                st.write(
                    f"Listed at {money(prop['price'])}; predicted fair value is "
                    f"{money(price['predicted_price'])} ({price['deviation_pct']:+.1f}% deviation). "
                    f"Verdict: **{price['verdict']}**, based on {price['comparable_properties']} comparables "
                    f"with {price['confidence_score']}% confidence."
                )
            with right:
                st.markdown("**📍 Locality Agent**")
                st.write(
                    f"Metro {locality['nearest_metro_km']:.2f} km · connectivity "
                    f"{locality['connectivity_score']}/10 · {locality['schools_nearby']} schools · "
                    f"{locality['hospitals_nearby']} hospitals · traffic {locality['traffic_index']}/10."
                )
                source_label = locality.get("data_source", "legacy").replace("_", " ").title()
                st.caption(
                    f"Locality data: {source_label} · "
                    f"{locality.get('data_confidence', 1.0) * 100:.0f}% confidence. "
                    "Estimated records are adjusted toward a neutral score."
                )
                st.markdown("**💰 ROI Agent prediction**")
                st.write(
                    f"Expected annual rent {money(roi['annual_rent'])}, rental yield "
                    f"{roi['rental_yield_pct']:.2f}%, and **{roi['risk_level']}**. "
                    f"Recommendation: **{roi['recommendation']}**."
                )
            if financial:
                st.markdown("**🏦 Financial Scenario Agent**")
                f1, f2, f3, f4 = st.columns(4)
                f1.metric("Upfront cash", money(financial["upfront_cash_required"]))
                f2.metric("Estimated EMI", f"{money(financial['monthly_emi'])}/mo")
                f3.metric("Net rental yield", f"{financial['net_rental_yield_pct']:.2f}%")
                f4.metric(
                    "Cash flow after EMI",
                    money(financial["annual_cash_flow_after_emi"]) + "/yr",
                )
                scenarios = financial["scenarios"]
                scenario_rows = []
                for name in ("conservative", "base", "optimistic"):
                    scenario = scenarios[name]
                    scenario_rows.append({
                        "Scenario": name.title(),
                        "Annual appreciation": f"{scenario['annual_appreciation_pct']:.2f}%",
                        f"Value after {financial['assumptions']['holding_period_years']} years":
                            money(scenario["future_property_value"]),
                        "Net profit": money(scenario["net_profit"]),
                        "Total return": f"{scenario['total_return_pct']:.2f}%",
                        "Annualized return": f"{scenario['annualized_return_pct']:.2f}%",
                    })
                st.dataframe(scenario_rows, use_container_width=True, hide_index=True)
                with st.expander("Financial assumptions and model limitations"):
                    assumptions = financial["assumptions"]
                    st.write(
                        f"Down payment {assumptions['down_payment_pct']}% · interest "
                        f"{assumptions['annual_interest_rate_pct']}% · loan "
                        f"{assumptions['loan_tenure_years']} years · holding period "
                        f"{assumptions['holding_period_years']} years · acquisition cost "
                        f"{assumptions['acquisition_cost_pct']}% · maintenance "
                        f"{assumptions['annual_maintenance_pct']}%/year · vacancy "
                        f"{assumptions['vacancy_pct']}% · rent growth "
                        f"{assumptions['annual_rent_growth_pct']}%/year · selling cost "
                        f"{assumptions['selling_cost_pct']}%."
                    )
                    st.warning(financial["warning"])

    winner = recommendations[0]
    wp, wr, wl = winner["property"], winner["roi_analysis"], winner["locality_analysis"]
    st.info(
        f"⭐ **Final recommendation:** {wp['locality']} ranks first with a "
        f"{winner['rank_score']}/10 match score, {wr['rental_yield_pct']:.2f}% projected yield, "
        f"and {wl['locality_score']}/10 locality quality. Predictions are decision-support estimates, "
        "not a guarantee; verify the listing, title, approvals, and physical condition before purchase."
    )
    report_bytes = json.dumps(
        result, indent=2, ensure_ascii=False, default=str
    ).encode("utf-8")
    st.download_button(
        "Download complete JSON report",
        data=report_bytes,
        file_name=f"propwise-report-{result['best_property']['property_id']}.json",
        mime="application/json",
        key=f"download_{result['best_property']['property_id']}_{id(result)}",
    )


def render_legal(result):
    if result.get("chunk_id") == -1:
        st.warning(result.get("answer", "No relevant legal passage was found."))
        return
    st.markdown("### ⚖️ Legal RAG Agent")
    st.write(result.get("answer"))
    a, b = st.columns(2)
    a.metric("Retrieval confidence", f"{result.get('confidence_score', 0):.0f}%")
    b.info(f"Retrieval: {result.get('retrieval_method', 'keyword').replace('_', ' ').title()}")
    sources = result.get("sources", [])
    if sources:
        st.markdown("**Evidence citations**")
        for index, source in enumerate(sources, start=1):
            page = f", page {source['page']}" if source.get("page") else ""
            similarity = (
                f" · semantic similarity {source['similarity'] * 100:.1f}%"
                if source.get("similarity") is not None else ""
            )
            st.caption(
                f"[{index}] {source.get('source_file', 'Unknown')}{page} "
                f"· chunk {source.get('chunk_id', 'N/A')}{similarity}"
            )
    st.caption("Legal information is educational and should be verified with a qualified professional.")


if "user" not in st.session_state:
    st.session_state.user = None
if "session_token" not in st.session_state:
    st.session_state.session_token = None

with st.sidebar:
    st.markdown("## 🏡 PropWise AI")
    st.caption("MULTI-AGENT PROPERTY INTELLIGENCE")
    if groq_enabled():
        st.success("Groq conversation intelligence: active")
    else:
        st.info("Groq: offline fallback active")
    if st.session_state.user:
        st.markdown("### 👤 Account")
        st.success(f"Signed in as {st.session_state.user['display_name']}")
        if st.button("Sign out", use_container_width=True):
            if st.session_state.session_token:
                revoke_session(st.session_state.session_token)
            st.session_state.user = None
            st.session_state.session_token = None
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
    else:
        st.markdown("### 👤 Account")
        login_tab, register_tab = st.tabs(["Login", "Register"])
        with login_tab:
            with st.form("login_form"):
                login_email = st.text_input("Email", key="login_email")
                login_password = st.text_input("Password", type="password", key="login_password")
                if st.form_submit_button("Login", use_container_width=True):
                    user = authenticate(login_email, login_password)
                    if user:
                        st.session_state.user = user
                        st.session_state.session_token = create_session(user["id"])
                        logger.info(f"login_success user_id={user['id']}")
                        st.rerun()
                    else:
                        logger.warning("login_failed")
                        st.error("Invalid email or password.")
        with register_tab:
            with st.form("register_form"):
                register_name = st.text_input("Display name")
                register_email = st.text_input("Email", key="register_email")
                register_password = st.text_input("Password", type="password", key="register_password")
                if st.form_submit_button("Create account", use_container_width=True):
                    try:
                        user = create_user(register_email, register_name, register_password)
                        st.session_state.user = user
                        st.session_state.session_token = create_session(user["id"])
                        logger.info(f"registration_success user_id={user['id']}")
                        st.rerun()
                    except ValueError as error:
                        st.error(str(error))
        st.caption("Guest mode is active. Sign in to save searches, favourites and reports.")
    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True, type="primary"):
        if st.session_state.user:
            deleted = clear_conversation_history(
                st.session_state.user["id"], st.session_state.thread_id
            )
            logger.info(
                f"conversation_cleared user_id={st.session_state.user['id']} "
                f"messages_deleted={deleted}"
            )
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    with st.expander("Agent network"):
        st.markdown(
            """
            1. Query Understanding
            2. Discovery
            3. Price ML
            4. Locality Intelligence
            5. ROI Prediction
            6. Legal RAG
            7. Recommendation
            8. Report
            """
        )
    with st.expander("Example queries"):
        st.caption("“Find a 2 BHK in Pune under 80 lakh for investment”")
        st.caption("“Find a 3 BHK in Mumbai below 1 crore”")
        st.caption("“What is RERA registration?”")
    if st.session_state.user:
        with st.expander("My favourites"):
            favourites = list_favourites(st.session_state.user["id"])
            if favourites:
                for favourite in favourites[:10]:
                    st.write(
                        f"**{favourite.get('locality')}** · "
                        f"{money(favourite.get('price'))} · {favourite.get('bhk')} BHK"
                    )
            else:
                st.caption("No saved properties yet.")
        with st.expander("My report history"):
            reports = list_reports(st.session_state.user["id"])
            if reports:
                for saved in reports[:10]:
                    st.caption(f"#{saved['id']} · {saved['query']} · {saved['created_at'][:10]}")
            else:
                st.caption("No saved reports yet.")

st.markdown(
    """<div class="hero"><h1>Find property with an AI advisory team</h1>
    <p>One conversation. Four ranked options. Transparent price, locality, ROI and legal intelligence.</p></div>""",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write(
            "Tell me your city, budget, and preferred BHK. You can also mention whether the property "
            "is for investment or self-use. I’ll show the work of every agent and compare up to four options."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("type") == "property":
            render_recommendations(message["data"])
        elif message.get("type") == "legal":
            render_legal(message["data"])

if prompt := st.chat_input("Example: 2 BHK in Pune under 80 lakh for investment"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    if st.session_state.user:
        save_message(
            st.session_state.user["id"], st.session_state.thread_id, "user", prompt
        )
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("LangGraph is routing the agent team and analyzing your request…"):
            outcome = run_conversation(prompt, st.session_state.thread_id)
        response = outcome["message"]
        if outcome["type"] == "property" and outcome.get("data"):
            st.write(response)
            render_recommendations(outcome["data"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "type": "property", "data": outcome["data"]}
            )
            if st.session_state.user:
                user_id = st.session_state.user["id"]
                save_message(
                    user_id, st.session_state.thread_id, "assistant", response,
                    "property", outcome["data"],
                )
                save_search(user_id, prompt, outcome.get("requirements", {}))
                report_id = save_report(
                    user_id, st.session_state.thread_id, prompt, outcome["data"]
                )
                logger.info(f"property_report_created user_id={user_id} report_id={report_id}")
        elif outcome["type"] == "legal" and outcome.get("data"):
            st.write(response)
            render_legal(outcome["data"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "type": "legal", "data": outcome["data"]}
            )
            if st.session_state.user:
                save_message(
                    st.session_state.user["id"], st.session_state.thread_id,
                    "assistant", response, "legal", outcome["data"],
                )
        else:
            st.info(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            if st.session_state.user:
                save_message(
                    st.session_state.user["id"], st.session_state.thread_id,
                    "assistant", response,
                )
