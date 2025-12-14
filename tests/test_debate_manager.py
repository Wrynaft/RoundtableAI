"""
Tests for debate manager logic.
"""
import pytest

from agents.debate_manager import DebateManager
from agents.debate_state import (
    DebateState,
    DebateMessage,
    AgentVote,
    Recommendation
)


class TestDebateManager:
    """Tests for DebateManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DebateManager(
            max_rounds=5,
            min_turns_per_agent=2,
            consensus_threshold=0.75
        )

    def test_manager_initialization(self):
        """Test manager initializes with correct defaults."""
        assert self.manager.max_rounds == 5
        assert self.manager.min_turns_per_agent == 2
        assert self.manager.consensus_threshold == 0.75
        assert self.manager.agent_order == ["fundamental", "sentiment", "valuation"]

    def test_custom_agent_order(self):
        """Test manager with custom agent order."""
        manager = DebateManager(agent_order=["valuation", "fundamental", "sentiment"])
        assert manager.agent_order == ["valuation", "fundamental", "sentiment"]

    def test_create_debate_state(self):
        """Test creating a new debate state."""
        state = self.manager.create_debate_state("Maybank", "1155.KL")

        assert state.company == "Maybank"
        assert state.ticker == "1155.KL"
        assert state.max_rounds == 5
        assert state.consensus_threshold == 0.75

    def test_get_next_speaker_initial(self):
        """Test getting first speaker."""
        state = self.manager.create_debate_state("Test", "TEST.KL")

        next_speaker = self.manager.get_next_speaker(state)

        assert next_speaker == "fundamental"  # First in default order

    def test_get_next_speaker_rotation(self):
        """Test speaker rotation."""
        state = self.manager.create_debate_state("Test", "TEST.KL")

        # Add messages to simulate progression
        state.add_message(DebateMessage("fundamental", "Msg 1", 0.8, Recommendation.BUY, 1))
        next_speaker = self.manager.get_next_speaker(state)
        assert next_speaker == "sentiment"

        state.add_message(DebateMessage("sentiment", "Msg 2", 0.7, Recommendation.HOLD, 1))
        next_speaker = self.manager.get_next_speaker(state)
        assert next_speaker == "valuation"

        state.add_message(DebateMessage("valuation", "Msg 3", 0.6, Recommendation.BUY, 1))
        next_speaker = self.manager.get_next_speaker(state)
        assert next_speaker == "fundamental"  # Back to start

    def test_get_speakers_for_round(self):
        """Test getting all speakers for a round."""
        speakers = self.manager.get_speakers_for_round(1)

        assert speakers == ["fundamental", "sentiment", "valuation"]

    def test_should_continue_debate_initial(self):
        """Test debate should continue at start."""
        state = self.manager.create_debate_state("Test", "TEST.KL")

        assert self.manager.should_continue_debate(state)

    def test_should_continue_debate_max_rounds(self):
        """Test debate stops at max rounds."""
        state = self.manager.create_debate_state("Test", "TEST.KL")
        state.current_round = 5

        assert not self.manager.should_continue_debate(state)

    def test_should_continue_debate_consensus(self):
        """Test debate stops when consensus reached."""
        state = self.manager.create_debate_state("Test", "TEST.KL")

        # Add minimum messages
        for i in range(6):  # min_turns_per_agent * 3 agents
            agent = self.manager.agent_order[i % 3]
            state.add_message(DebateMessage(agent, f"Msg {i}", 0.8, Recommendation.BUY, i // 3 + 1))

        # Add unanimous votes
        for agent in self.manager.agent_order:
            state.update_vote(AgentVote(agent, Recommendation.BUY, 0.8, "Test"))

        assert not self.manager.should_continue_debate(state)

    def test_calculate_consensus_unanimous(self):
        """Test consensus calculation with unanimous votes."""
        votes = {
            "fundamental": AgentVote("fundamental", Recommendation.BUY, 0.8, "Test"),
            "sentiment": AgentVote("sentiment", Recommendation.BUY, 0.7, "Test"),
            "valuation": AgentVote("valuation", Recommendation.BUY, 0.9, "Test")
        }

        consensus = self.manager.calculate_consensus(votes)

        assert consensus == 1.0

    def test_calculate_consensus_split(self):
        """Test consensus calculation with split votes."""
        votes = {
            "fundamental": AgentVote("fundamental", Recommendation.BUY, 0.8, "Test"),
            "sentiment": AgentVote("sentiment", Recommendation.SELL, 0.7, "Test"),
            "valuation": AgentVote("valuation", Recommendation.HOLD, 0.9, "Test")
        }

        consensus = self.manager.calculate_consensus(votes)

        assert abs(consensus - 0.333) < 0.01  # Each has 1/3

    def test_calculate_weighted_consensus(self):
        """Test weighted consensus calculation."""
        votes = {
            "fundamental": AgentVote("fundamental", Recommendation.BUY, 0.9, "Test"),  # High confidence BUY
            "sentiment": AgentVote("sentiment", Recommendation.BUY, 0.5, "Test"),  # Low confidence BUY
            "valuation": AgentVote("valuation", Recommendation.SELL, 0.6, "Test")  # Medium confidence SELL
        }

        weighted = self.manager.calculate_weighted_consensus(votes)

        # BUY: 0.9 + 0.5 = 1.4, SELL: 0.6, Total: 2.0
        # BUY should win with 1.4/2.0 = 0.7
        assert abs(weighted - 0.7) < 0.01

    def test_get_majority_recommendation(self):
        """Test getting majority recommendation."""
        votes = {
            "fundamental": AgentVote("fundamental", Recommendation.BUY, 0.9, "Test"),
            "sentiment": AgentVote("sentiment", Recommendation.BUY, 0.7, "Test"),
            "valuation": AgentVote("valuation", Recommendation.SELL, 0.8, "Test")
        }

        majority = self.manager.get_majority_recommendation(votes)

        assert majority == Recommendation.BUY


class TestDebatePrompts:
    """Tests for debate prompt generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DebateManager()

    def test_initial_analysis_prompt_fundamental(self):
        """Test fundamental agent initial prompt."""
        prompt = self.manager.get_initial_analysis_prompt("Maybank", "1155.KL", "fundamental")

        assert "Maybank" in prompt
        assert "1155.KL" in prompt
        assert "FUNDAMENTAL" in prompt
        assert "financial statements" in prompt.lower()
        assert "BUY" in prompt
        assert "HOLD" in prompt
        assert "SELL" in prompt

    def test_initial_analysis_prompt_sentiment(self):
        """Test sentiment agent initial prompt."""
        prompt = self.manager.get_initial_analysis_prompt("Maybank", "1155.KL", "sentiment")

        assert "SENTIMENT" in prompt
        assert "news" in prompt.lower()
        assert "FinBERT" in prompt

    def test_initial_analysis_prompt_valuation(self):
        """Test valuation agent initial prompt."""
        prompt = self.manager.get_initial_analysis_prompt("Maybank", "1155.KL", "valuation")

        assert "VALUATION" in prompt
        assert "risk" in prompt.lower()
        assert "Sharpe" in prompt

    def test_response_prompt(self):
        """Test response prompt generation."""
        state = self.manager.create_debate_state("Maybank", "1155.KL")

        prompt = self.manager.get_response_prompt(
            company="Maybank",
            ticker="1155.KL",
            agent_type="sentiment",
            other_agent="fundamental",
            other_analysis="The company shows strong fundamentals with ROE of 15%.",
            state=state
        )

        assert "sentiment" in prompt.lower()
        assert "fundamental" in prompt.lower()
        assert "ROE of 15%" in prompt
        assert "agree" in prompt.lower() or "disagree" in prompt.lower()

    def test_synthesis_prompt(self):
        """Test synthesis prompt generation."""
        state = self.manager.create_debate_state("Maybank", "1155.KL")

        # Add votes
        state.update_vote(AgentVote("fundamental", Recommendation.BUY, 0.8, "Strong financials"))
        state.update_vote(AgentVote("sentiment", Recommendation.BUY, 0.7, "Positive news"))
        state.update_vote(AgentVote("valuation", Recommendation.HOLD, 0.6, "Fair valuation"))

        prompt = self.manager.get_synthesis_prompt(state)

        assert "Maybank" in prompt
        assert "synthesize" in prompt.lower()
        assert "BUY" in prompt
        assert "HOLD" in prompt


class TestResponseParsing:
    """Tests for parsing agent responses."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = DebateManager()

    def test_parse_recommendation_explicit(self):
        """Test parsing explicit recommendation."""
        response = "Based on my analysis, my Recommendation: BUY with high confidence."
        rec = self.manager.parse_recommendation_from_response(response)
        assert rec == Recommendation.BUY

    def test_parse_recommendation_implicit(self):
        """Test parsing implicit recommendation."""
        response = "I strongly recommend investors BUY this stock. The fundamentals are solid."
        rec = self.manager.parse_recommendation_from_response(response)
        assert rec == Recommendation.BUY

    def test_parse_recommendation_sell(self):
        """Test parsing SELL recommendation."""
        response = "Recommendation: SELL. The risks outweigh potential returns."
        rec = self.manager.parse_recommendation_from_response(response)
        assert rec == Recommendation.SELL

    def test_parse_recommendation_hold(self):
        """Test parsing HOLD recommendation."""
        response = "I recommend to HOLD for now until we see clearer signals."
        rec = self.manager.parse_recommendation_from_response(response)
        assert rec == Recommendation.HOLD

    def test_parse_confidence_percentage(self):
        """Test parsing confidence as percentage."""
        response = "My confidence level: 85%"
        conf = self.manager.parse_confidence_from_response(response)
        assert abs(conf - 0.85) < 0.01

    def test_parse_confidence_decimal(self):
        """Test parsing confidence as decimal."""
        response = "Confidence: 0.75"
        conf = self.manager.parse_confidence_from_response(response)
        assert abs(conf - 0.75) < 0.01

    def test_parse_confidence_default(self):
        """Test default confidence when not found."""
        response = "This is a great company."
        conf = self.manager.parse_confidence_from_response(response)
        assert conf == 0.5  # Default

    def test_create_vote_from_response(self):
        """Test creating vote from full response."""
        response = """
        Based on my analysis:

        Recommendation: BUY
        Confidence: 80%

        Key Supporting Evidence:
        - Strong revenue growth of 15% YoY
        - Healthy profit margins
        - Low debt levels
        """

        vote = self.manager.create_vote_from_response("fundamental", response)

        assert vote.agent_type == "fundamental"
        assert vote.recommendation == Recommendation.BUY
        assert abs(vote.confidence - 0.8) < 0.01

    def test_create_message_from_response(self):
        """Test creating message from response."""
        response = "Recommendation: HOLD. Confidence: 70%"

        msg = self.manager.create_message_from_response(
            agent_type="sentiment",
            response=response,
            round_number=2,
            is_response=True,
            responding_to="fundamental"
        )

        assert msg.agent_type == "sentiment"
        assert msg.recommendation == Recommendation.HOLD
        assert abs(msg.confidence - 0.7) < 0.01
        assert msg.round_number == 2
        assert msg.is_response
        assert msg.responding_to == "fundamental"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
