"""
Tests for debate state management.
"""
import pytest
from datetime import datetime

from agents.debate_state import (
    DebateMessage,
    DebateState,
    AgentVote,
    RoundResult,
    Recommendation,
    FinalRecommendation
)


class TestRecommendation:
    """Tests for Recommendation enum."""

    def test_recommendation_values(self):
        """Test that all recommendation values exist."""
        assert Recommendation.BUY.value == "BUY"
        assert Recommendation.HOLD.value == "HOLD"
        assert Recommendation.SELL.value == "SELL"

    def test_recommendation_from_string(self):
        """Test creating recommendation from string."""
        assert Recommendation("BUY") == Recommendation.BUY
        assert Recommendation("HOLD") == Recommendation.HOLD
        assert Recommendation("SELL") == Recommendation.SELL


class TestDebateMessage:
    """Tests for DebateMessage dataclass."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = DebateMessage(
            agent_type="fundamental",
            content="Test analysis content",
            confidence=0.8,
            recommendation=Recommendation.BUY,
            round_number=1
        )

        assert msg.agent_type == "fundamental"
        assert msg.confidence == 0.8
        assert msg.recommendation == Recommendation.BUY
        assert msg.round_number == 1
        assert not msg.is_response
        assert msg.responding_to is None

    def test_message_with_response(self):
        """Test message that is a response."""
        msg = DebateMessage(
            agent_type="sentiment",
            content="Response to fundamental",
            confidence=0.7,
            recommendation=Recommendation.HOLD,
            round_number=2,
            is_response=True,
            responding_to="fundamental"
        )

        assert msg.is_response
        assert msg.responding_to == "fundamental"

    def test_message_to_dict(self):
        """Test message serialization."""
        msg = DebateMessage(
            agent_type="valuation",
            content="Test",
            confidence=0.6,
            recommendation=Recommendation.SELL,
            round_number=1
        )

        d = msg.to_dict()

        assert d["agent_type"] == "valuation"
        assert d["confidence"] == 0.6
        assert d["recommendation"] == "SELL"
        assert "timestamp" in d

    def test_message_from_dict(self):
        """Test message deserialization."""
        d = {
            "agent_type": "fundamental",
            "content": "Test",
            "confidence": 0.75,
            "recommendation": "BUY",
            "round_number": 1,
            "timestamp": datetime.now().isoformat(),
            "is_response": False,
            "responding_to": None
        }

        msg = DebateMessage.from_dict(d)

        assert msg.agent_type == "fundamental"
        assert msg.confidence == 0.75
        assert msg.recommendation == Recommendation.BUY


class TestAgentVote:
    """Tests for AgentVote dataclass."""

    def test_vote_creation(self):
        """Test basic vote creation."""
        vote = AgentVote(
            agent_type="fundamental",
            recommendation=Recommendation.BUY,
            confidence=0.85,
            reasoning="Strong financials",
            key_factors=["High ROE", "Low debt"]
        )

        assert vote.agent_type == "fundamental"
        assert vote.recommendation == Recommendation.BUY
        assert vote.confidence == 0.85
        assert len(vote.key_factors) == 2

    def test_vote_to_dict(self):
        """Test vote serialization."""
        vote = AgentVote(
            agent_type="sentiment",
            recommendation=Recommendation.HOLD,
            confidence=0.6,
            reasoning="Mixed signals"
        )

        d = vote.to_dict()

        assert d["agent_type"] == "sentiment"
        assert d["recommendation"] == "HOLD"


class TestDebateState:
    """Tests for DebateState dataclass."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = DebateState(
            company="Maybank",
            ticker="1155.KL"
        )

        assert state.company == "Maybank"
        assert state.ticker == "1155.KL"
        assert state.current_round == 0
        assert len(state.messages) == 0
        assert not state.consensus_reached

    def test_add_message(self):
        """Test adding messages to state."""
        state = DebateState(company="Test", ticker="TEST.KL")

        msg = DebateMessage(
            agent_type="fundamental",
            content="Test",
            confidence=0.8,
            recommendation=Recommendation.BUY,
            round_number=1
        )

        state.add_message(msg)

        assert len(state.messages) == 1
        assert state.messages[0].agent_type == "fundamental"

    def test_update_vote(self):
        """Test updating agent votes."""
        state = DebateState(company="Test", ticker="TEST.KL")

        vote = AgentVote(
            agent_type="fundamental",
            recommendation=Recommendation.BUY,
            confidence=0.8,
            reasoning="Test"
        )

        state.update_vote(vote)

        assert "fundamental" in state.current_votes
        assert state.current_votes["fundamental"].recommendation == Recommendation.BUY

    def test_calculate_consensus(self):
        """Test consensus calculation."""
        state = DebateState(company="Test", ticker="TEST.KL")

        # Add unanimous votes
        for agent in ["fundamental", "sentiment", "valuation"]:
            vote = AgentVote(
                agent_type=agent,
                recommendation=Recommendation.BUY,
                confidence=0.8,
                reasoning="Test"
            )
            state.update_vote(vote)

        consensus = state.calculate_consensus()
        assert consensus == 1.0  # 100% agreement

    def test_calculate_consensus_partial(self):
        """Test partial consensus calculation."""
        state = DebateState(company="Test", ticker="TEST.KL")

        # 2 BUY, 1 SELL
        state.update_vote(AgentVote("fundamental", Recommendation.BUY, 0.8, "Test"))
        state.update_vote(AgentVote("sentiment", Recommendation.BUY, 0.7, "Test"))
        state.update_vote(AgentVote("valuation", Recommendation.SELL, 0.6, "Test"))

        consensus = state.calculate_consensus()
        assert abs(consensus - 0.667) < 0.01  # ~66.7% agreement

    def test_check_consensus(self):
        """Test consensus threshold checking."""
        state = DebateState(company="Test", ticker="TEST.KL", consensus_threshold=0.75)

        # Add unanimous votes
        for agent in ["fundamental", "sentiment", "valuation"]:
            state.update_vote(AgentVote(agent, Recommendation.BUY, 0.8, "Test"))

        assert state.check_consensus()  # 100% >= 75%

    def test_get_messages_by_agent(self):
        """Test filtering messages by agent."""
        state = DebateState(company="Test", ticker="TEST.KL")

        state.add_message(DebateMessage("fundamental", "Msg 1", 0.8, Recommendation.BUY, 1))
        state.add_message(DebateMessage("sentiment", "Msg 2", 0.7, Recommendation.HOLD, 1))
        state.add_message(DebateMessage("fundamental", "Msg 3", 0.85, Recommendation.BUY, 2))

        fundamental_msgs = state.get_messages_by_agent("fundamental")

        assert len(fundamental_msgs) == 2
        assert all(m.agent_type == "fundamental" for m in fundamental_msgs)

    def test_get_messages_by_round(self):
        """Test filtering messages by round."""
        state = DebateState(company="Test", ticker="TEST.KL")

        state.add_message(DebateMessage("fundamental", "Msg 1", 0.8, Recommendation.BUY, 1))
        state.add_message(DebateMessage("sentiment", "Msg 2", 0.7, Recommendation.HOLD, 1))
        state.add_message(DebateMessage("valuation", "Msg 3", 0.6, Recommendation.BUY, 2))

        round1_msgs = state.get_messages_by_round(1)

        assert len(round1_msgs) == 2
        assert all(m.round_number == 1 for m in round1_msgs)

    def test_is_finished_by_consensus(self):
        """Test debate finishing by consensus."""
        state = DebateState(company="Test", ticker="TEST.KL", consensus_threshold=0.75)

        # Add unanimous votes
        for agent in ["fundamental", "sentiment", "valuation"]:
            state.update_vote(AgentVote(agent, Recommendation.BUY, 0.8, "Test"))

        state.consensus_reached = True
        assert state.is_finished()

    def test_is_finished_by_max_rounds(self):
        """Test debate finishing by max rounds."""
        state = DebateState(company="Test", ticker="TEST.KL", max_rounds=5)
        state.current_round = 5

        assert state.is_finished()

    def test_state_serialization(self):
        """Test state to_dict and from_dict."""
        state = DebateState(company="Test", ticker="TEST.KL")
        state.add_message(DebateMessage("fundamental", "Test", 0.8, Recommendation.BUY, 1))
        state.update_vote(AgentVote("fundamental", Recommendation.BUY, 0.8, "Test"))

        d = state.to_dict()
        restored = DebateState.from_dict(d)

        assert restored.company == state.company
        assert restored.ticker == state.ticker
        assert len(restored.messages) == 1
        assert "fundamental" in restored.current_votes


class TestFinalRecommendation:
    """Tests for FinalRecommendation dataclass."""

    def test_final_recommendation_creation(self):
        """Test creating final recommendation."""
        final = FinalRecommendation(
            recommendation=Recommendation.BUY,
            confidence=0.85,
            consensus_level=0.75,
            summary="Strong fundamentals and positive sentiment",
            key_points=["High ROE", "Positive news"],
            risks=["Market volatility"],
            agent_breakdown={
                "fundamental": AgentVote("fundamental", Recommendation.BUY, 0.9, "Test"),
                "sentiment": AgentVote("sentiment", Recommendation.BUY, 0.8, "Test"),
                "valuation": AgentVote("valuation", Recommendation.HOLD, 0.7, "Test")
            }
        )

        assert final.recommendation == Recommendation.BUY
        assert final.confidence == 0.85
        assert len(final.key_points) == 2
        assert len(final.risks) == 1

    def test_final_recommendation_to_dict(self):
        """Test final recommendation serialization."""
        final = FinalRecommendation(
            recommendation=Recommendation.SELL,
            confidence=0.7,
            consensus_level=0.67,
            summary="Test",
            key_points=["Point 1"],
            risks=["Risk 1"],
            agent_breakdown={}
        )

        d = final.to_dict()

        assert d["recommendation"] == "SELL"
        assert d["confidence"] == 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
