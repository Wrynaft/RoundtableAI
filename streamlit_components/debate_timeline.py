"""
Debate timeline components for visualizing the multi-agent conversation.
"""
import streamlit as st
from typing import List, Optional
from datetime import datetime


# Agent colors for consistent styling
AGENT_COLORS = {
    "fundamental": "#4CAF50",
    "sentiment": "#2196F3",
    "valuation": "#FF9800"
}

AGENT_ICONS = {
    "fundamental": "📊",
    "sentiment": "📰",
    "valuation": "📈"
}

RECOMMENDATION_COLORS = {
    "BUY": "#4CAF50",
    "HOLD": "#FFC107",
    "SELL": "#F44336"
}


def render_message_card(
    agent_type: str,
    content: str,
    recommendation: str,
    confidence: float,
    round_number: int,
    timestamp: Optional[datetime] = None,
    is_response: bool = False,
    responding_to: Optional[str] = None,
    expanded: bool = False
) -> None:
    """
    Render a single debate message card.

    Args:
        agent_type: Type of agent (fundamental, sentiment, valuation)
        content: Message content
        recommendation: Agent's recommendation (BUY/HOLD/SELL)
        confidence: Confidence level (0.0-1.0)
        round_number: Debate round number
        timestamp: When the message was created
        is_response: Whether this is a response to another agent
        responding_to: Agent type being responded to
        expanded: Whether to show full content by default
    """
    color = AGENT_COLORS.get(agent_type, "#9E9E9E")
    icon = AGENT_ICONS.get(agent_type, "🤖")
    rec_color = RECOMMENDATION_COLORS.get(recommendation.upper(), "#9E9E9E")

    # Format timestamp
    time_str = ""
    if timestamp:
        if isinstance(timestamp, datetime):
            time_str = timestamp.strftime("%H:%M:%S")
        elif isinstance(timestamp, str):
            # Try to parse the string if it's in ISO format
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                time_str = dt.strftime("%H:%M:%S")
            except (ValueError, AttributeError):
                time_str = timestamp  # Use as-is if parsing fails

    # Response indicator
    response_text = ""
    if is_response and responding_to:
        responding_icon = AGENT_ICONS.get(responding_to, "🤖")
        response_text = f'<span style="font-size: 12px; color: #888;">↳ Responding to {responding_icon} {responding_to.title()}</span>'

    # Create the message card
    st.markdown(f"""
        <div style="
            border-left: 3px solid {color};
            padding: 10px 15px;
            margin: 10px 0;
            background-color: #1E1E1E;
            border-radius: 0 8px 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center;">
                    <span style="font-size: 20px; margin-right: 8px;">{icon}</span>
                    <span style="font-weight: bold; color: {color};">{agent_type.title()} Analyst</span>
                    <span style="
                        margin-left: 10px;
                        padding: 2px 8px;
                        background-color: {rec_color};
                        color: white;
                        border-radius: 10px;
                        font-size: 12px;
                        font-weight: bold;
                    ">{recommendation}</span>
                    <span style="
                        margin-left: 8px;
                        font-size: 12px;
                        color: #888;
                    ">{confidence:.0%} confidence</span>
                </div>
                <div style="font-size: 12px; color: #666;">
                    Round {round_number} • {time_str}
                </div>
            </div>
            {response_text}
        </div>
    """, unsafe_allow_html=True)

    # Content expander
    with st.expander("View Analysis", expanded=expanded):
        st.markdown(content)


def render_debate_timeline(
    messages: List[dict],
    max_messages: int = 50
) -> None:
    """
    Render the full debate timeline.

    Args:
        messages: List of message dictionaries with keys:
                 agent_type, content, recommendation, confidence,
                 round_number, timestamp, is_response, responding_to
        max_messages: Maximum messages to display
    """
    if not messages:
        st.info("No messages yet. Start a debate to see the conversation.")
        return

    # Group messages by round
    rounds = {}
    for msg in messages[-max_messages:]:
        round_num = msg.get("round_number", 1)
        if round_num not in rounds:
            rounds[round_num] = []
        rounds[round_num].append(msg)

    # Render each round
    for round_num in sorted(rounds.keys()):
        st.markdown(f"""
            <div style="
                padding: 5px 15px;
                background-color: #333;
                border-radius: 5px;
                margin: 15px 0 10px 0;
                display: inline-block;
            ">
                <span style="font-weight: bold; color: #FFC107;">Round {round_num}</span>
            </div>
        """, unsafe_allow_html=True)

        for msg in rounds[round_num]:
            render_message_card(
                agent_type=msg.get("agent_type", "unknown"),
                content=msg.get("content", ""),
                recommendation=msg.get("recommendation", "HOLD"),
                confidence=msg.get("confidence", 0.5),
                round_number=round_num,
                timestamp=msg.get("timestamp"),
                is_response=msg.get("is_response", False),
                responding_to=msg.get("responding_to"),
                expanded=False
            )


def render_round_summary(
    round_number: int,
    votes: dict,
    consensus: float
) -> None:
    """
    Render a summary of a completed round.

    Args:
        round_number: The round number
        votes: Dictionary of agent_type -> vote info
        consensus: Consensus percentage achieved
    """
    # Determine consensus status
    if consensus >= 0.75:
        consensus_status = "🟢 Consensus Reached"
        status_color = "#4CAF50"
    elif consensus >= 0.5:
        consensus_status = "🟡 Partial Agreement"
        status_color = "#FFC107"
    else:
        consensus_status = "🔴 No Consensus"
        status_color = "#F44336"

    st.markdown(f"""
        <div style="
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            background-color: #2D2D2D;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight: bold;">Round {round_number} Summary</span>
                <span style="color: {status_color};">{consensus_status} ({consensus:.0%})</span>
            </div>
            <div style="display: flex; gap: 20px;">
    """, unsafe_allow_html=True)

    # Vote summary
    cols = st.columns(3)
    for i, (agent_type, vote) in enumerate(votes.items()):
        with cols[i]:
            icon = AGENT_ICONS.get(agent_type, "🤖")
            rec = vote.get("recommendation", "HOLD")
            conf = vote.get("confidence", 0.5)
            rec_color = RECOMMENDATION_COLORS.get(rec.upper(), "#9E9E9E")

            st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 24px;">{icon}</div>
                    <div style="
                        padding: 3px 10px;
                        background-color: {rec_color};
                        color: white;
                        border-radius: 10px;
                        font-size: 12px;
                        margin: 5px 0;
                        display: inline-block;
                    ">{rec}</div>
                    <div style="font-size: 11px; color: #888;">{conf:.0%}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)
