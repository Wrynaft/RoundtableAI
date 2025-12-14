"""
Recommendation panel components for displaying final analysis results.
"""
import streamlit as st
from typing import Dict, List, Optional, Any


RECOMMENDATION_CONFIG = {
    "BUY": {
        "color": "#4CAF50",
        "bg_color": "#1B5E20",
        "icon": "📈",
        "description": "Strong fundamentals, positive sentiment, favorable risk-return"
    },
    "HOLD": {
        "color": "#FFC107",
        "bg_color": "#F57F17",
        "icon": "⏸️",
        "description": "Mixed signals, monitor for changes"
    },
    "SELL": {
        "color": "#F44336",
        "bg_color": "#B71C1C",
        "icon": "📉",
        "description": "Concerns identified across multiple factors"
    }
}

AGENT_ICONS = {
    "fundamental": "📊",
    "sentiment": "📰",
    "valuation": "📈"
}


def render_recommendation_panel(
    recommendation: str,
    confidence: float,
    consensus: float,
    summary: str,
    key_points: List[str] = None,
    risks: List[str] = None,
    company: str = "",
    ticker: str = ""
) -> None:
    """
    Render the final recommendation panel.

    Args:
        recommendation: Final recommendation (BUY/HOLD/SELL)
        confidence: Overall confidence level (0.0-1.0)
        consensus: Consensus level achieved (0.0-1.0)
        summary: Summary text of the analysis
        key_points: List of key investment points
        risks: List of identified risks
        company: Company name
        ticker: Stock ticker symbol
    """
    config = RECOMMENDATION_CONFIG.get(recommendation.upper(), RECOMMENDATION_CONFIG["HOLD"])

    # Header with recommendation
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {config['bg_color']} 0%, #1E1E1E 100%);
            border-radius: 15px;
            padding: 25px;
            margin: 20px 0;
            border: 2px solid {config['color']};
        ">
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="font-size: 48px; margin-bottom: 10px;">{config['icon']}</div>
                <div style="
                    font-size: 36px;
                    font-weight: bold;
                    color: {config['color']};
                    letter-spacing: 2px;
                ">{recommendation.upper()}</div>
                <div style="font-size: 14px; color: #AAAAAA; margin-top: 5px;">
                    {company} ({ticker})
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Metrics row
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Confidence",
            value=f"{confidence:.0%}",
            delta=None
        )

    with col2:
        st.metric(
            label="Agent Consensus",
            value=f"{consensus:.0%}",
            delta="Unanimous" if consensus >= 1.0 else None
        )

    with col3:
        consensus_text = "High" if consensus >= 0.75 else "Medium" if consensus >= 0.5 else "Low"
        st.metric(
            label="Agreement Level",
            value=consensus_text
        )

    # Summary section
    st.markdown("### Analysis Summary")
    st.markdown(summary)

    # Key points and risks in columns
    if key_points or risks:
        col1, col2 = st.columns(2)

        with col1:
            if key_points:
                st.markdown("### 🎯 Key Investment Points")
                for point in key_points:
                    st.markdown(f"- {point}")

        with col2:
            if risks:
                st.markdown("### ⚠️ Key Risks")
                for risk in risks:
                    st.markdown(f"- {risk}")


def render_vote_summary(
    votes: Dict[str, Dict[str, Any]],
    show_reasoning: bool = True
) -> None:
    """
    Render a summary of all agent votes.

    Args:
        votes: Dictionary mapping agent_type to vote info
               Each vote should have: recommendation, confidence, reasoning, key_factors
        show_reasoning: Whether to show detailed reasoning
    """
    st.markdown("### Agent Votes")

    # Vote count summary
    vote_counts = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for vote in votes.values():
        rec = vote.get("recommendation", "HOLD").upper()
        if rec in vote_counts:
            vote_counts[rec] += 1

    # Visual vote summary
    cols = st.columns(3)
    for i, (rec, count) in enumerate(vote_counts.items()):
        config = RECOMMENDATION_CONFIG.get(rec, {})
        with cols[i]:
            st.markdown(f"""
                <div style="
                    text-align: center;
                    padding: 15px;
                    background-color: {'#2D2D2D' if count > 0 else '#1E1E1E'};
                    border-radius: 10px;
                    border: 2px solid {config.get('color', '#444') if count > 0 else '#333'};
                ">
                    <div style="font-size: 32px;">{config.get('icon', '❓')}</div>
                    <div style="
                        font-size: 24px;
                        font-weight: bold;
                        color: {config.get('color', '#888')};
                    ">{rec}</div>
                    <div style="font-size: 18px; color: #888;">{count} vote{'s' if count != 1 else ''}</div>
                </div>
            """, unsafe_allow_html=True)

    # Detailed vote breakdown
    if show_reasoning:
        st.markdown("---")
        st.markdown("### Detailed Agent Analysis")

        for agent_type, vote in votes.items():
            icon = AGENT_ICONS.get(agent_type, "🤖")
            rec = vote.get("recommendation", "HOLD")
            conf = vote.get("confidence", 0.5)
            reasoning = vote.get("reasoning", "No reasoning provided")
            factors = vote.get("key_factors", [])

            rec_config = RECOMMENDATION_CONFIG.get(rec.upper(), {})

            with st.expander(f"{icon} {agent_type.title()} Analyst - {rec} ({conf:.0%} confidence)"):
                st.markdown(f"**Recommendation:** :{'green' if rec.upper() == 'BUY' else 'orange' if rec.upper() == 'HOLD' else 'red'}[{rec}]")
                st.markdown(f"**Confidence:** {conf:.0%}")

                st.markdown("**Reasoning:**")
                st.markdown(reasoning[:1000] + "..." if len(reasoning) > 1000 else reasoning)

                if factors:
                    st.markdown("**Key Factors:**")
                    for factor in factors:
                        st.markdown(f"- {factor}")


def render_consensus_gauge(
    consensus: float,
    threshold: float = 0.75
) -> None:
    """
    Render a visual consensus gauge.

    Args:
        consensus: Current consensus level (0.0-1.0)
        threshold: Consensus threshold for decision
    """
    # Determine color based on consensus level
    if consensus >= threshold:
        color = "#4CAF50"
        status = "Consensus Reached"
    elif consensus >= 0.5:
        color = "#FFC107"
        status = "Partial Agreement"
    else:
        color = "#F44336"
        status = "No Consensus"

    st.markdown(f"""
        <div style="
            padding: 15px;
            background-color: #2D2D2D;
            border-radius: 10px;
            margin: 10px 0;
        ">
            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                <span style="font-weight: bold;">Consensus Level</span>
                <span style="color: {color};">{status}</span>
            </div>
            <div style="
                background-color: #1E1E1E;
                border-radius: 5px;
                height: 20px;
                overflow: hidden;
                position: relative;
            ">
                <div style="
                    background-color: {color};
                    width: {consensus * 100}%;
                    height: 100%;
                    border-radius: 5px;
                    transition: width 0.5s ease;
                "></div>
                <div style="
                    position: absolute;
                    left: {threshold * 100}%;
                    top: 0;
                    height: 100%;
                    width: 2px;
                    background-color: white;
                "></div>
            </div>
            <div style="
                display: flex;
                justify-content: space-between;
                font-size: 11px;
                color: #888;
                margin-top: 5px;
            ">
                <span>0%</span>
                <span>Threshold: {threshold:.0%}</span>
                <span>100%</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_export_options(
    debate_data: dict,
    transcript: str
) -> None:
    """
    Render export options for the debate results.

    Args:
        debate_data: Full debate data dictionary
        transcript: Formatted transcript string
    """
    st.markdown("### Export Options")

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="📄 Download Transcript",
            data=transcript,
            file_name=f"debate_transcript_{debate_data.get('ticker', 'unknown')}.txt",
            mime="text/plain"
        )

    with col2:
        import json
        st.download_button(
            label="📊 Download Full Data (JSON)",
            data=json.dumps(debate_data, indent=2, default=str),
            file_name=f"debate_data_{debate_data.get('ticker', 'unknown')}.json",
            mime="application/json"
        )
