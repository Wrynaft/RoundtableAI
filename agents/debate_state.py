"""
Debate state management for multi-agent discussions.

This module provides data structures for tracking:
- Individual debate messages from agents
- Agent votes and recommendations
- Overall debate state and consensus
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict
from enum import Enum


class Recommendation(Enum):
    """Investment recommendation options."""
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class DebateMessage:
    """
    Represents a single message in the debate.

    Attributes:
        agent_type: Type of agent (fundamental, sentiment, valuation)
        content: The message content/analysis
        confidence: Confidence level (0.0 to 1.0)
        recommendation: Investment recommendation (BUY/HOLD/SELL)
        round_number: Which debate round this message belongs to
        timestamp: When the message was created
        is_response: Whether this is a response to another agent
        responding_to: Agent type being responded to (if is_response=True)
    """
    agent_type: str
    content: str
    confidence: float
    recommendation: Recommendation
    round_number: int
    timestamp: datetime = field(default_factory=datetime.now)
    is_response: bool = False
    responding_to: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert message to dictionary for serialization."""
        return {
            "agent_type": self.agent_type,
            "content": self.content,
            "confidence": self.confidence,
            "recommendation": self.recommendation.value,
            "round_number": self.round_number,
            "timestamp": self.timestamp.isoformat(),
            "is_response": self.is_response,
            "responding_to": self.responding_to
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebateMessage":
        """Create message from dictionary."""
        return cls(
            agent_type=data["agent_type"],
            content=data["content"],
            confidence=data["confidence"],
            recommendation=Recommendation(data["recommendation"]),
            round_number=data["round_number"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            is_response=data.get("is_response", False),
            responding_to=data.get("responding_to")
        )


@dataclass
class AgentVote:
    """
    Represents an agent's vote/recommendation.

    Attributes:
        agent_type: Type of agent casting the vote
        recommendation: Investment recommendation (BUY/HOLD/SELL)
        confidence: Confidence level (0.0 to 1.0)
        reasoning: Brief explanation for the vote
        key_factors: List of key factors supporting the recommendation
    """
    agent_type: str
    recommendation: Recommendation
    confidence: float
    reasoning: str
    key_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert vote to dictionary for serialization."""
        return {
            "agent_type": self.agent_type,
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "key_factors": self.key_factors
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentVote":
        """Create vote from dictionary."""
        return cls(
            agent_type=data["agent_type"],
            recommendation=Recommendation(data["recommendation"]),
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            key_factors=data.get("key_factors", [])
        )


@dataclass
class RoundResult:
    """
    Results from a single debate round.

    Attributes:
        round_number: The round number
        messages: Messages from this round
        votes: Current votes after this round
        consensus_percentage: Consensus level achieved
        leading_recommendation: Most popular recommendation
    """
    round_number: int
    messages: List[DebateMessage]
    votes: List[AgentVote]
    consensus_percentage: float
    leading_recommendation: Recommendation

    def to_dict(self) -> dict:
        """Convert round result to dictionary."""
        return {
            "round_number": self.round_number,
            "messages": [m.to_dict() for m in self.messages],
            "votes": [v.to_dict() for v in self.votes],
            "consensus_percentage": self.consensus_percentage,
            "leading_recommendation": self.leading_recommendation.value
        }


@dataclass
class DebateState:
    """
    Tracks the overall state of a multi-agent debate.

    Attributes:
        company: Company name being analyzed
        ticker: Stock ticker symbol
        risk_tolerance: User's risk tolerance (conservative/moderate/aggressive)
        messages: All messages in the debate
        current_round: Current round number
        max_rounds: Maximum rounds allowed
        consensus_threshold: Required consensus percentage (0.0 to 1.0)
        consensus_reached: Whether consensus has been achieved
        final_recommendation: Final synthesized recommendation
        current_votes: Latest votes from each agent
        round_results: Results from each completed round
        started_at: When the debate started
        ended_at: When the debate ended (if finished)
    """
    company: str
    ticker: str
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    messages: List[DebateMessage] = field(default_factory=list)
    current_round: int = 0
    max_rounds: int = 5
    consensus_threshold: float = 0.75
    consensus_reached: bool = False
    final_recommendation: Optional[Recommendation] = None
    current_votes: Dict[str, AgentVote] = field(default_factory=dict)
    round_results: List[RoundResult] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None

    def add_message(self, message: DebateMessage) -> None:
        """Add a message to the debate history."""
        self.messages.append(message)

    def update_vote(self, vote: AgentVote) -> None:
        """Update an agent's vote."""
        self.current_votes[vote.agent_type] = vote

    def get_messages_by_agent(self, agent_type: str) -> List[DebateMessage]:
        """Get all messages from a specific agent."""
        return [m for m in self.messages if m.agent_type == agent_type]

    def get_messages_by_round(self, round_number: int) -> List[DebateMessage]:
        """Get all messages from a specific round."""
        return [m for m in self.messages if m.round_number == round_number]

    def get_latest_message_by_agent(self, agent_type: str) -> Optional[DebateMessage]:
        """Get the most recent message from a specific agent."""
        agent_messages = self.get_messages_by_agent(agent_type)
        return agent_messages[-1] if agent_messages else None

    def calculate_consensus(self) -> float:
        """
        Calculate current consensus percentage.

        Returns:
            Float between 0.0 and 1.0 representing consensus level.
            1.0 = all agents agree, 0.33 = evenly split (3 agents)
        """
        if not self.current_votes:
            return 0.0

        # Count votes by recommendation
        vote_counts: Dict[Recommendation, int] = {}
        for vote in self.current_votes.values():
            vote_counts[vote.recommendation] = vote_counts.get(vote.recommendation, 0) + 1

        # Find most common recommendation
        max_votes = max(vote_counts.values())
        total_votes = len(self.current_votes)

        return max_votes / total_votes if total_votes > 0 else 0.0

    def get_leading_recommendation(self) -> Optional[Recommendation]:
        """Get the recommendation with most votes."""
        if not self.current_votes:
            return None

        vote_counts: Dict[Recommendation, float] = {}
        for vote in self.current_votes.values():
            # Weight by confidence
            weighted_vote = vote.confidence
            vote_counts[vote.recommendation] = vote_counts.get(vote.recommendation, 0) + weighted_vote

        return max(vote_counts.keys(), key=lambda r: vote_counts[r])

    def check_consensus(self) -> bool:
        """Check if consensus threshold has been reached."""
        consensus = self.calculate_consensus()
        return consensus >= self.consensus_threshold

    def is_finished(self) -> bool:
        """Check if debate should end."""
        return self.consensus_reached or self.current_round >= self.max_rounds

    def finish_debate(self, final_rec: Recommendation) -> None:
        """Mark the debate as finished."""
        self.consensus_reached = self.check_consensus()
        self.final_recommendation = final_rec
        self.ended_at = datetime.now()

    def get_debate_summary(self) -> str:
        """Generate a summary of the debate for synthesis."""
        summary_parts = []
        summary_parts.append(f"Debate Summary for {self.company} ({self.ticker})")
        summary_parts.append(f"Investor Risk Tolerance: {self.risk_tolerance.upper()}")
        summary_parts.append(f"Rounds completed: {self.current_round}")
        summary_parts.append(f"Consensus: {self.calculate_consensus():.1%}")
        summary_parts.append("")

        # Summarize each agent's position
        for agent_type, vote in self.current_votes.items():
            summary_parts.append(f"=== {agent_type.upper()} AGENT ===")
            summary_parts.append(f"Recommendation: {vote.recommendation.value}")
            summary_parts.append(f"Confidence: {vote.confidence:.0%}")
            summary_parts.append(f"Reasoning: {vote.reasoning}")
            if vote.key_factors:
                summary_parts.append("Key factors:")
                for factor in vote.key_factors:
                    summary_parts.append(f"  - {factor}")
            summary_parts.append("")

        return "\n".join(summary_parts)

    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization."""
        return {
            "company": self.company,
            "ticker": self.ticker,
            "risk_tolerance": self.risk_tolerance,
            "messages": [m.to_dict() for m in self.messages],
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "consensus_threshold": self.consensus_threshold,
            "consensus_reached": self.consensus_reached,
            "final_recommendation": self.final_recommendation.value if self.final_recommendation else None,
            "current_votes": {k: v.to_dict() for k, v in self.current_votes.items()},
            "round_results": [r.to_dict() for r in self.round_results],
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DebateState":
        """Create state from dictionary."""
        state = cls(
            company=data["company"],
            ticker=data["ticker"],
            risk_tolerance=data.get("risk_tolerance", "moderate"),
            current_round=data["current_round"],
            max_rounds=data["max_rounds"],
            consensus_threshold=data["consensus_threshold"],
            consensus_reached=data["consensus_reached"],
            started_at=datetime.fromisoformat(data["started_at"])
        )
        state.messages = [DebateMessage.from_dict(m) for m in data["messages"]]
        state.final_recommendation = Recommendation(data["final_recommendation"]) if data["final_recommendation"] else None
        state.current_votes = {k: AgentVote.from_dict(v) for k, v in data["current_votes"].items()}
        state.ended_at = datetime.fromisoformat(data["ended_at"]) if data["ended_at"] else None
        return state


@dataclass
class FinalRecommendation:
    """
    Final synthesized recommendation from the debate.

    Attributes:
        recommendation: The final BUY/HOLD/SELL recommendation
        confidence: Overall confidence in the recommendation
        consensus_level: Level of agreement among agents
        summary: Synthesized summary of all perspectives
        key_points: Key points from the analysis
        risks: Identified risks
        agent_breakdown: How each agent voted
        debate_state: Reference to the full debate state
    """
    recommendation: Recommendation
    confidence: float
    consensus_level: float
    summary: str
    key_points: List[str]
    risks: List[str]
    agent_breakdown: Dict[str, AgentVote]
    debate_state: Optional[DebateState] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for display/storage."""
        return {
            "recommendation": self.recommendation.value,
            "confidence": self.confidence,
            "consensus_level": self.consensus_level,
            "summary": self.summary,
            "key_points": self.key_points,
            "risks": self.risks,
            "agent_breakdown": {k: v.to_dict() for k, v in self.agent_breakdown.items()}
        }
