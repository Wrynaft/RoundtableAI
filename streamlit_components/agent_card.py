"""
Agent card components for displaying agent information and status.
"""
import streamlit as st
from typing import Optional, Dict, Any


# Agent configuration with colors and icons
AGENT_CONFIG = {
    "fundamental": {
        "name": "Fundamental Analyst",
        "icon": "📊",
        "color": "#4CAF50",  # Green
        "description": "Analyzes financial statements and company fundamentals"
    },
    "sentiment": {
        "name": "Sentiment Analyst",
        "icon": "📰",
        "color": "#2196F3",  # Blue
        "description": "Analyzes news sentiment and market perception"
    },
    "valuation": {
        "name": "Valuation Analyst",
        "icon": "📈",
        "color": "#FF9800",  # Orange
        "description": "Analyzes risk-return metrics and valuations"
    }
}


def render_agent_card(
    agent_type: str,
    recommendation: Optional[str] = None,
    confidence: Optional[float] = None,
    is_active: bool = False,
    is_thinking: bool = False
) -> None:
    """
    Render an agent card with status information.

    Args:
        agent_type: Type of agent (fundamental, sentiment, valuation)
        recommendation: Current recommendation (BUY/HOLD/SELL)
        confidence: Confidence level (0.0-1.0)
        is_active: Whether agent is currently speaking
        is_thinking: Whether agent is processing
    """
    config = AGENT_CONFIG.get(agent_type, {
        "name": agent_type.title(),
        "icon": "🤖",
        "color": "#9E9E9E",
        "description": "Analysis agent"
    })

    # Card styling based on state
    border_color = config["color"] if is_active else "#444444"
    bg_color = "#1E1E1E" if not is_active else "#2D2D2D"

    # Render card container
    st.markdown(f"""
        <div style="
            border: 2px solid {border_color};
            border-radius: 10px;
            padding: 15px;
            background-color: {bg_color};
            margin-bottom: 10px;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 24px; margin-right: 10px;">{config['icon']}</span>
                <span style="font-size: 16px; font-weight: bold; color: {config['color']};">
                    {config['name']}
                </span>
            </div>
            <p style="font-size: 12px; color: #888888; margin-bottom: 10px;">
                {config['description']}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Status indicator
    if is_thinking:
        st.markdown(f"""
            <div style="
                display: flex;
                align-items: center;
                padding: 5px 10px;
                background-color: #333333;
                border-radius: 5px;
                margin-top: -5px;
            ">
                <span style="color: {config['color']};">⏳ Analyzing...</span>
            </div>
        """, unsafe_allow_html=True)

    # Recommendation badge
    if recommendation:
        rec_color = {
            "BUY": "#4CAF50",
            "HOLD": "#FFC107",
            "SELL": "#F44336"
        }.get(recommendation.upper(), "#9E9E9E")

        st.markdown(f"""
            <div style="
                display: inline-block;
                padding: 5px 15px;
                background-color: {rec_color};
                color: white;
                border-radius: 15px;
                font-weight: bold;
                font-size: 14px;
            ">
                {recommendation.upper()}
            </div>
        """, unsafe_allow_html=True)

    # Confidence bar
    if confidence is not None:
        st.progress(confidence, text=f"Confidence: {confidence:.0%}")


def render_agent_status(
    agent_type: str,
    status: str = "idle"
) -> None:
    """
    Render a simple agent status indicator.

    Args:
        agent_type: Type of agent
        status: Current status (idle, thinking, complete)
    """
    config = AGENT_CONFIG.get(agent_type, {"icon": "🤖", "name": agent_type, "color": "#9E9E9E"})

    status_icons = {
        "idle": "⚪",
        "thinking": "🔄",
        "complete": "✅",
        "error": "❌"
    }

    status_icon = status_icons.get(status, "⚪")

    st.markdown(f"""
        <div style="
            display: flex;
            align-items: center;
            padding: 8px;
            background-color: #2D2D2D;
            border-radius: 5px;
            margin: 5px 0;
        ">
            <span style="margin-right: 8px;">{config['icon']}</span>
            <span style="color: {config['color']}; flex-grow: 1;">{config['name']}</span>
            <span>{status_icon}</span>
        </div>
    """, unsafe_allow_html=True)


def render_agent_panel(
    agents_status: Dict[str, Dict[str, Any]]
) -> None:
    """
    Render a panel showing all agents and their current status.

    Args:
        agents_status: Dictionary mapping agent_type to status info
                      e.g., {"fundamental": {"status": "complete", "recommendation": "BUY", "confidence": 0.8}}
    """
    cols = st.columns(3)

    for i, (agent_type, status_info) in enumerate(agents_status.items()):
        with cols[i]:
            render_agent_card(
                agent_type=agent_type,
                recommendation=status_info.get("recommendation"),
                confidence=status_info.get("confidence"),
                is_active=status_info.get("is_active", False),
                is_thinking=status_info.get("is_thinking", False)
            )
