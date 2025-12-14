"""
Stock Analysis Page - Multi-Agent Debate Interface

This is the main interface for running multi-agent stock analysis debates.
Users can input natural language queries and receive investment recommendations.
"""
import streamlit as st
import time
from datetime import datetime
from typing import Optional

st.set_page_config(
    page_title="Stock Analysis - RoundtableAI",
    page_icon="💬",
    layout="wide"
)

# Import components and agents
from streamlit_components import (
    render_agent_card,
    render_debate_timeline,
    render_recommendation_panel,
    render_vote_summary
)
from streamlit_components.agent_card import render_agent_panel
from streamlit_components.debate_timeline import render_round_summary
from streamlit_components.recommendation_panel import render_consensus_gauge, render_export_options
from utils.config import get_streamlit_config, get_debate_config


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if "debate_state" not in st.session_state:
        st.session_state.debate_state = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "is_debating" not in st.session_state:
        st.session_state.is_debating = False
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    if "agent_statuses" not in st.session_state:
        st.session_state.agent_statuses = {
            "fundamental": {"status": "idle", "recommendation": None, "confidence": None},
            "sentiment": {"status": "idle", "recommendation": None, "confidence": None},
            "valuation": {"status": "idle", "recommendation": None, "confidence": None}
        }
    if "final_recommendation" not in st.session_state:
        st.session_state.final_recommendation = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "debate_history" not in st.session_state:
        st.session_state.debate_history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


# =============================================================================
# Callback Functions
# =============================================================================

def on_message_callback(message):
    """Callback when a new debate message is created."""
    msg_dict = message.to_dict() if hasattr(message, 'to_dict') else message

    # Deduplication: Check if this exact message already exists
    # A message is duplicate if it has same agent_type, round_number, and is_response status
    # (each agent speaks exactly once per round, either as initial analysis or response)
    is_duplicate = any(
        existing.get("agent_type") == msg_dict.get("agent_type") and
        existing.get("round_number") == msg_dict.get("round_number") and
        existing.get("is_response") == msg_dict.get("is_response")
        for existing in st.session_state.messages
    )

    if not is_duplicate:
        st.session_state.messages.append(msg_dict)

    # Update agent status
    agent_type = msg_dict.get("agent_type")
    if agent_type in st.session_state.agent_statuses:
        st.session_state.agent_statuses[agent_type] = {
            "status": "complete",
            "recommendation": msg_dict.get("recommendation"),
            "confidence": msg_dict.get("confidence"),
            "is_active": False,
            "is_thinking": False
        }


def on_round_complete_callback(round_result):
    """Callback when a debate round completes."""
    result_dict = round_result.to_dict() if hasattr(round_result, 'to_dict') else round_result
    st.session_state.debate_history.append({
        "type": "round_complete",
        "round_number": result_dict.get("round_number"),
        "consensus": result_dict.get("consensus_percentage"),
        "timestamp": datetime.now()
    })


# =============================================================================
# Main Page
# =============================================================================

def main():
    """Main page entry point."""
    init_session_state()

    # Custom CSS
    st.markdown("""
        <style>
        .stButton > button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .query-box {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="color: #4CAF50;">💬 Stock Analysis</h1>
            <p style="color: #888; font-size: 18px;">
                Multi-Agent Debate System for Investment Recommendations
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar settings
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")

        debate_config = get_debate_config()

        max_rounds = st.slider(
            "Maximum Rounds",
            min_value=1,
            max_value=10,
            value=debate_config["max_rounds"],
            help="Maximum number of debate rounds before forcing a decision"
        )

        consensus_threshold = st.slider(
            "Consensus Threshold",
            min_value=0.5,
            max_value=1.0,
            value=debate_config["consensus_threshold"],
            step=0.05,
            help="Required agreement level to end debate early"
        )

        st.markdown("---")

        # Quick select companies
        st.markdown("### 🏢 Quick Select")
        popular_companies = ["Maybank", "CIMB", "Public Bank", "Tenaga Nasional", "Petronas Chemicals"]

        for company in popular_companies:
            if st.button(company, key=f"quick_{company}", use_container_width=True):
                st.session_state.user_query = f"Should I invest in {company}?"
                st.rerun()

        st.markdown("---")

        # Reset button
        if st.button("🔄 Reset Analysis", disabled=not st.session_state.messages, use_container_width=True):
            st.session_state.debate_state = None
            st.session_state.messages = []
            st.session_state.is_debating = False
            st.session_state.final_recommendation = None
            st.session_state.last_result = None
            st.session_state.user_query = ""
            st.session_state.agent_statuses = {
                agent: {"status": "idle", "recommendation": None, "confidence": None}
                for agent in ["fundamental", "sentiment", "valuation"]
            }
            st.rerun()

    # Main content
    st.markdown("---")

    # Query input section
    st.markdown("### 💭 Ask About a Stock")

    st.markdown("""
    <div style="color: #888; font-size: 14px; margin-bottom: 15px;">
        Enter your investment question. The system will automatically detect the company
        and infer your risk tolerance from the query.
    </div>
    """, unsafe_allow_html=True)

    # Example queries
    with st.expander("📝 Example Queries"):
        st.markdown("""
        **Conservative investor:**
        - "I'm a retiree looking for safe dividend stocks. Should I invest in Maybank?"
        - "Is Public Bank suitable for capital preservation?"

        **Moderate investor:**
        - "Should I invest in CIMB? I'm looking for balanced growth."
        - "What's your recommendation on Tenaga Nasional?"

        **Aggressive investor:**
        - "I want maximum growth and don't mind volatility. Should I buy Petronas Chemicals?"
        - "Looking for high-risk high-reward stocks. Is Top Glove a good bet?"

        **Single agent queries (no debate):**
        - "What is the recent news sentiment for CIMB?"
        - "What's the P/E ratio and ROE of Maybank?"
        - "What is the volatility and Sharpe ratio for Public Bank?"
        """)

    # Query input
    col1, col2 = st.columns([4, 1])

    with col1:
        user_query = st.text_input(
            "Your Question",
            value=st.session_state.user_query,
            placeholder="e.g., Should I invest in Maybank? I'm looking for stable dividends.",
            label_visibility="collapsed"
        )

    with col2:
        analyze_button = st.button(
            "🚀 Analyze",
            type="primary",
            disabled=st.session_state.is_debating or not user_query,
            use_container_width=True
        )

    # Run analysis
    if analyze_button and user_query:
        st.session_state.user_query = user_query
        st.session_state.is_debating = True
        st.session_state.messages = []
        st.session_state.final_recommendation = None
        st.session_state.last_result = None
        st.session_state.agent_statuses = {
            agent: {"status": "idle", "recommendation": None, "confidence": None}
            for agent in ["fundamental", "sentiment", "valuation"]
        }
        run_analysis(user_query, max_rounds, consensus_threshold)

    # Display results
    if st.session_state.last_result:
        render_results(st.session_state.last_result)
    elif not st.session_state.is_debating:
        render_welcome()


def run_analysis(query: str, max_rounds: int, consensus_threshold: float):
    """Run the analysis using the orchestrator with real-time streaming."""
    st.markdown("---")
    st.markdown(f"### 🔄 Processing Query")
    st.markdown(f"> *{query}*")

    # Agent display info
    agent_info = {
        "fundamental": {"icon": "📊", "name": "Fundamental Analyst", "color": "#4CAF50"},
        "sentiment": {"icon": "📰", "name": "Sentiment Analyst", "color": "#2196F3"},
        "valuation": {"icon": "📈", "name": "Valuation Analyst", "color": "#FF9800"}
    }

    try:
        # Import orchestrator (lazy load)
        from agents import create_debate_orchestrator

        # Create orchestrator if not exists
        if st.session_state.orchestrator is None:
            with st.spinner("Loading AI models... (this may take a moment on first run)"):
                st.session_state.orchestrator = create_debate_orchestrator(
                    max_rounds=max_rounds,
                    consensus_threshold=consensus_threshold,
                    on_message_callback=on_message_callback
                )

        orchestrator = st.session_state.orchestrator

        # Use st.status for real-time progress updates
        with st.status("🔍 Analyzing your query...", expanded=True) as status:
            # Step 1: Classify the query
            st.write("**Step 1:** Classifying query and inferring risk tolerance...")
            classification = orchestrator.classify_query(query)

            company = classification.get("company")
            risk_tolerance = classification.get("risk_tolerance", "moderate")
            needs_debate = classification.get("needs_debate", True)

            risk_emoji = {"conservative": "🛡️", "moderate": "⚖️", "aggressive": "🚀"}.get(risk_tolerance, "⚖️")
            st.write(f"• Company detected: **{company or 'Not specified'}**")
            st.write(f"• Risk tolerance: {risk_emoji} **{risk_tolerance.title()}**")
            st.write(f"• Route: **{'Multi-Agent Debate' if needs_debate else 'Single Agent'}**")

            if needs_debate:
                if not company:
                    st.session_state.last_result = {
                        "response": "Please specify which company or stock you'd like me to analyze.",
                        "route_type": "error",
                        "agent_used": None,
                        "recommendation": None,
                        "risk_tolerance": risk_tolerance,
                        "classification": classification
                    }
                    st.session_state.is_debating = False
                    status.update(label="❌ Analysis failed - no company specified", state="error")
                    st.rerun()
                    return

                # Step 2: Run multi-agent debate with streaming
                st.write("")
                st.write("**Step 2:** Starting multi-agent debate...")

                # Resolve company to ticker
                resolution = orchestrator.resolve_company(company)
                if not resolution.get("success"):
                    raise ValueError(f"Could not resolve company: {company}. {resolution.get('error', '')}")

                ticker = resolution["ticker"]
                company_name = resolution.get("company_name", company)
                st.write(f"• Resolved: **{company_name}** ({ticker})")

                # Start the debate generator
                st.write("")
                st.write("**Step 3:** Agents analyzing...")

                # Track current round for display
                current_display_round = 0
                message_count = 0

                # Use the generator directly for streaming
                debate_gen = orchestrator.start_debate(company_name, risk_tolerance=risk_tolerance)

                try:
                    while True:
                        message = next(debate_gen)
                        message_count += 1

                        # Call the callback to store the message
                        on_message_callback(message)

                        # Display round header if new round
                        if message.round_number != current_display_round:
                            current_display_round = message.round_number
                            round_label = "Initial Analysis" if current_display_round == 1 else f"Round {current_display_round} - Response"
                            st.write(f"")
                            st.write(f"**{round_label}:**")

                        # Display agent message summary
                        info = agent_info.get(message.agent_type, {"icon": "🤖", "name": "Agent"})
                        rec_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(message.recommendation.value, "⚪")
                        st.write(f"  {info['icon']} **{info['name']}**: {rec_emoji} {message.recommendation.value} ({message.confidence:.0%} confidence)")

                except StopIteration as e:
                    # Debate complete, get final recommendation
                    final_rec = e.value

                # Step 4: Synthesis
                st.write("")
                st.write("**Step 4:** Synthesizing final recommendation...")

                # Build result
                result = {
                    "response": final_rec.summary,
                    "route_type": "debate",
                    "agent_used": ["fundamental", "sentiment", "valuation"],
                    "recommendation": final_rec.recommendation.value,
                    "confidence": final_rec.confidence,
                    "consensus": final_rec.consensus_level,
                    "risk_tolerance": risk_tolerance,
                    "full_result": final_rec,
                    "classification": classification
                }

                # Store result
                st.session_state.last_result = result
                st.session_state.final_recommendation = final_rec.to_dict() if hasattr(final_rec, 'to_dict') else final_rec

                # Final status
                rec_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}.get(final_rec.recommendation.value, "⚪")
                status.update(
                    label=f"✅ Analysis complete - {rec_emoji} {final_rec.recommendation.value} ({final_rec.confidence:.0%} confidence)",
                    state="complete",
                    expanded=False
                )

            else:
                # Single agent query with streaming
                st.write("")
                st.write("**Step 2:** Routing to single agent...")

                agent_type = classification.get("agent_type") or "fundamental"
                info = agent_info.get(agent_type, {"icon": "🤖", "name": "Agent"})
                st.write(f"  {info['icon']} **{info['name']}** is analyzing...")
                st.write("")

                # Use streaming for single agent response
                stream_gen = orchestrator._handle_single_agent_query_stream(
                    query, agent_type, company, risk_tolerance
                )

                # Stream the response with st.write_stream
                response_container = st.empty()
                full_response = []

                try:
                    while True:
                        chunk = next(stream_gen)
                        full_response.append(chunk)
                        # Update the display with accumulated response
                        response_container.markdown("".join(full_response))
                except StopIteration as e:
                    # Get the final result from the generator
                    result = e.value if e.value else {
                        "response": "".join(full_response),
                        "route_type": "single_agent",
                        "agent_used": agent_type,
                        "risk_tolerance": risk_tolerance,
                        "recommendation": None
                    }

                result["classification"] = classification

                # Store result
                st.session_state.last_result = result

                status.update(label="✅ Analysis complete", state="complete", expanded=False)

        st.session_state.is_debating = False
        st.rerun()

    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        st.session_state.is_debating = False

        with st.expander("Error Details"):
            st.exception(e)


def render_results(result: dict):
    """Render the analysis results."""
    st.markdown("---")

    route_type = result.get('route_type')
    classification = result.get('classification', {})

    # Show classification info
    st.markdown("### 📋 Query Classification")

    info_cols = st.columns(4)

    with info_cols[0]:
        st.metric("Route Type", route_type.upper() if route_type else "N/A")

    with info_cols[1]:
        risk_tol = result.get('risk_tolerance', 'moderate')
        risk_emoji = {"conservative": "🛡️", "moderate": "⚖️", "aggressive": "🚀"}.get(risk_tol, "⚖️")
        st.metric("Risk Tolerance", f"{risk_emoji} {risk_tol.title()}")

    with info_cols[2]:
        company = classification.get('company', 'N/A')
        st.metric("Company", company)

    with info_cols[3]:
        if route_type == 'debate':
            st.metric("Agents Used", "3 (All)")
        else:
            st.metric("Agent Used", result.get('agent_used', 'N/A').title())

    st.markdown("---")

    # Render based on route type
    if route_type == 'debate':
        render_debate_results(result)
    elif route_type == 'single_agent':
        render_single_agent_results(result)
    elif route_type == 'error':
        st.error(result.get('response', 'An error occurred'))


def render_debate_results(result: dict):
    """Render multi-agent debate results."""
    st.markdown("### 🎯 Final Recommendation")

    rec = result.get('recommendation', 'HOLD')
    confidence = result.get('confidence', 0.5)
    consensus = result.get('consensus', 0.5)

    # Recommendation color
    rec_colors = {"BUY": "#4CAF50", "HOLD": "#FF9800", "SELL": "#f44336"}
    rec_color = rec_colors.get(rec, "#888")

    # Main recommendation display
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #1E1E1E 0%, #2a2a2a 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border-left: 5px solid {rec_color};
    ">
        <h1 style="color: {rec_color}; font-size: 3em; margin: 0;">{rec}</h1>
        <p style="color: #888; margin-top: 10px;">
            Confidence: <strong>{confidence:.0%}</strong> | Consensus: <strong>{consensus:.0%}</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Full result details
    if 'full_result' in result:
        full_rec = result['full_result']
        if hasattr(full_rec, 'to_dict'):
            full_rec = full_rec.to_dict()

        # Summary
        if full_rec.get('summary'):
            st.markdown("### 📝 Analysis Summary")
            st.markdown(full_rec['summary'])

        # Key points and risks
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ✅ Key Points")
            for point in full_rec.get('key_points', []):
                st.markdown(f"- {point}")

        with col2:
            st.markdown("### ⚠️ Key Risks")
            for risk in full_rec.get('risks', []):
                st.markdown(f"- {risk}")

        # Agent breakdown (vote summary only - full analysis available in debate timeline)
        if full_rec.get('agent_breakdown'):
            st.markdown("### 🤖 Agent Breakdown")
            render_vote_summary(full_rec['agent_breakdown'], show_reasoning=False)

        # Consensus gauge
        render_consensus_gauge(consensus, 0.75)

    # Debate timeline
    if st.session_state.messages:
        st.markdown("### 💬 Debate Timeline")
        render_debate_timeline(st.session_state.messages)

    # Export options
    if st.session_state.orchestrator:
        debate_data = st.session_state.orchestrator.export_debate()
        transcript = st.session_state.orchestrator.get_debate_transcript() if hasattr(st.session_state.orchestrator, 'get_debate_transcript') else ""
        render_export_options(debate_data, transcript)


def render_single_agent_results(result: dict):
    """Render single agent results."""
    st.markdown("### 💡 Agent Response")

    agent_used = result.get('agent_used', 'unknown')
    agent_colors = {
        "fundamental": "#4CAF50",
        "sentiment": "#2196F3",
        "valuation": "#FF9800"
    }
    color = agent_colors.get(agent_used, "#888")

    st.markdown(f"""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid {color};
    ">
        <p style="color: {color}; font-weight: bold; margin-bottom: 10px;">
            {agent_used.upper()} AGENT
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(result.get('response', 'No response available'))


def render_welcome():
    """Render welcome message when no analysis is running."""
    st.markdown("""
    <div style="
        text-align: center;
        padding: 50px;
        background-color: #1E1E1E;
        border-radius: 15px;
        margin: 20px 0;
    ">
        <h2>Ready to Analyze</h2>
        <p style="color: #888; font-size: 16px;">
            Enter a question about a Malaysian stock above to get started.
        </p>
        <div style="margin-top: 30px;">
            <p style="color: #666;">The system will:</p>
            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 20px;">
                <div style="text-align: center;">
                    <div style="font-size: 36px;">🔍</div>
                    <div style="color: #4CAF50;">Classify Query</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 36px;">🎯</div>
                    <div style="color: #2196F3;">Infer Risk Profile</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 36px;">🤖</div>
                    <div style="color: #FF9800;">Route to Agent(s)</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
