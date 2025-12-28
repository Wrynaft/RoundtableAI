"""
Debate management logic for multi-agent discussions.

This module provides:
- Round-robin agent selection
- Consensus calculation
- Debate prompts for initial analysis and responses
- Debate flow control
"""
from typing import List, Dict, Optional, Callable
from datetime import datetime
import re

from .debate_state import (
    DebateState,
    DebateMessage,
    AgentVote,
    RoundResult,
    Recommendation,
    FinalRecommendation
)


# Default agent order for round-robin selection
DEFAULT_AGENT_ORDER = ["fundamental", "sentiment", "valuation"]


class DebateManager:
    """
    Manages the flow and logic of multi-agent debates.

    Handles turn selection, consensus checking, and debate prompts.
    """

    def __init__(
        self,
        agent_order: List[str] = None,
        max_rounds: int = 5,
        min_turns_per_agent: int = 2,
        consensus_threshold: float = 0.75
    ):
        """
        Initialize the debate manager.

        Args:
            agent_order: Order of agents for round-robin turns
            max_rounds: Maximum number of debate rounds
            min_turns_per_agent: Minimum turns each agent must take
            consensus_threshold: Required consensus level to end debate early
        """
        self.agent_order = agent_order or DEFAULT_AGENT_ORDER
        self.max_rounds = max_rounds
        self.min_turns_per_agent = min_turns_per_agent
        self.consensus_threshold = consensus_threshold

    def create_debate_state(
        self,
        company: str,
        ticker: str,
        risk_tolerance: str = "moderate"
    ) -> DebateState:
        """
        Create a new debate state for a company.

        Args:
            company: Company name
            ticker: Stock ticker symbol
            risk_tolerance: User's risk tolerance (conservative/moderate/aggressive)

        Returns:
            Initialized DebateState
        """
        return DebateState(
            company=company,
            ticker=ticker,
            risk_tolerance=risk_tolerance,
            max_rounds=self.max_rounds,
            consensus_threshold=self.consensus_threshold
        )

    def get_next_speaker(self, state: DebateState) -> str:
        """
        Get the next agent to speak in round-robin order.

        Args:
            state: Current debate state

        Returns:
            Agent type string for the next speaker
        """
        # Count total messages to determine position in order
        total_messages = len(state.messages)
        agent_index = total_messages % len(self.agent_order)
        return self.agent_order[agent_index]

    def get_speakers_for_round(self, round_number: int) -> List[str]:
        """
        Get the ordered list of speakers for a given round.

        Args:
            round_number: The round number (1-indexed)

        Returns:
            List of agent types in speaking order
        """
        return self.agent_order.copy()

    def should_continue_debate(self, state: DebateState) -> bool:
        """
        Determine if the debate should continue.

        Args:
            state: Current debate state

        Returns:
            True if debate should continue, False otherwise
        """
        # Check if max rounds reached
        if state.current_round >= self.max_rounds:
            return False

        # Check if consensus reached (only after minimum turns)
        min_messages = len(self.agent_order) * self.min_turns_per_agent
        if len(state.messages) >= min_messages:
            if state.check_consensus():
                return False

        return True

    def calculate_consensus(self, votes: Dict[str, AgentVote]) -> float:
        """
        Calculate consensus percentage from votes.

        Args:
            votes: Dictionary of agent votes

        Returns:
            Consensus percentage (0.0 to 1.0)
        """
        if not votes:
            return 0.0

        # Count votes by recommendation
        vote_counts: Dict[Recommendation, int] = {}
        for vote in votes.values():
            rec = vote.recommendation
            vote_counts[rec] = vote_counts.get(rec, 0) + 1

        # Calculate consensus as max votes / total votes
        max_votes = max(vote_counts.values())
        total_votes = len(votes)

        return max_votes / total_votes

    def calculate_weighted_consensus(self, votes: Dict[str, AgentVote]) -> float:
        """
        Calculate confidence-weighted consensus.

        Higher confidence votes count more toward consensus.

        Args:
            votes: Dictionary of agent votes

        Returns:
            Weighted consensus percentage (0.0 to 1.0)
        """
        if not votes:
            return 0.0

        # Sum confidence-weighted votes by recommendation
        weighted_counts: Dict[Recommendation, float] = {}
        total_weight = 0.0

        for vote in votes.values():
            rec = vote.recommendation
            weight = vote.confidence
            weighted_counts[rec] = weighted_counts.get(rec, 0) + weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        max_weight = max(weighted_counts.values())
        return max_weight / total_weight

    def get_majority_recommendation(self, votes: Dict[str, AgentVote]) -> Recommendation:
        """
        Get the recommendation with most votes (weighted by confidence).

        Args:
            votes: Dictionary of agent votes

        Returns:
            The majority recommendation
        """
        if not votes:
            return Recommendation.HOLD  # Default to HOLD if no votes

        weighted_counts: Dict[Recommendation, float] = {}
        for vote in votes.values():
            weight = vote.confidence
            weighted_counts[vote.recommendation] = weighted_counts.get(vote.recommendation, 0) + weight

        return max(weighted_counts.keys(), key=lambda r: weighted_counts[r])

    def create_round_result(self, state: DebateState, round_number: int) -> RoundResult:
        """
        Create a result summary for a completed round.

        Args:
            state: Current debate state
            round_number: The round that just completed

        Returns:
            RoundResult with summary information
        """
        round_messages = state.get_messages_by_round(round_number)
        votes = list(state.current_votes.values())
        consensus = self.calculate_consensus(state.current_votes)
        leading = self.get_majority_recommendation(state.current_votes)

        return RoundResult(
            round_number=round_number,
            messages=round_messages,
            votes=votes,
            consensus_percentage=consensus,
            leading_recommendation=leading
        )

    def get_initial_analysis_prompt(
        self,
        company: str,
        ticker: str,
        agent_type: str,
        risk_tolerance: str = "moderate"
    ) -> str:
        """
        Generate the initial analysis prompt for an agent.

        Args:
            company: Company name
            ticker: Stock ticker symbol
            agent_type: Type of agent (fundamental, sentiment, valuation)
            risk_tolerance: User's risk tolerance (conservative/moderate/aggressive)

        Returns:
            Formatted prompt string
        """
        # Risk tolerance context for each profile
        risk_context = {
            "conservative": "The investor is CONSERVATIVE - they prioritize capital preservation, stable income, and low volatility over high returns. They prefer established companies with consistent dividends and minimal downside risk.",
            "moderate": "The investor is MODERATE - they seek a balance between growth and stability. They accept reasonable risk for better returns but want to avoid extreme volatility.",
            "aggressive": "The investor is AGGRESSIVE - they prioritize maximum growth potential and are willing to accept high volatility and significant short-term losses for potentially higher long-term returns."
        }

        risk_guidance = risk_context.get(risk_tolerance, risk_context["moderate"])

        # Structured header instruction
        structured_header = """**CRITICAL: You MUST start your response with this EXACT structured header format:**
```
[DECISION]
RECOMMENDATION: <BUY or HOLD or SELL>
CONFIDENCE: <number from 0 to 100>%
[/DECISION]
```

After the structured header, provide your detailed analysis."""

        prompts = {
            "fundamental": f"""Analyze {company} ({ticker}) from a FUNDAMENTAL ANALYSIS perspective.

**INVESTOR PROFILE**: {risk_guidance}

You are the Fundamental Analysis Agent. Your role is to evaluate the company's financial health, profitability, and intrinsic value based on financial statements.

{structured_header}

In your analysis, cover:

1. **Key Findings**: What do the financial statements reveal about this company?
   - Revenue and profit trends
   - Financial ratios (P/E, ROE, debt ratios, etc.)
   - Cash flow quality
   - Balance sheet strength

2. **Key Supporting Evidence**: List 3-5 specific data points that support your recommendation.

3. **Key Risks**: What fundamental risks should this {risk_tolerance} investor be particularly aware of?

Be specific with numbers and data. Tailor your recommendation to the investor's risk profile.""",

            "sentiment": f"""Analyze {company} ({ticker}) from a MARKET SENTIMENT perspective.

**INVESTOR PROFILE**: {risk_guidance}

You are the Sentiment Analysis Agent. Your role is to evaluate market perception, news sentiment, and investor mood based on recent news articles and FinBERT sentiment scores.

{structured_header}

In your analysis, cover:

1. **Key Findings**: What does recent news sentiment reveal?
   - Overall sentiment distribution (positive/negative/neutral percentages)
   - Key themes in recent coverage
   - Notable news events
   - Sentiment trend (improving/stable/deteriorating)

2. **Key Supporting Evidence**: List 3-5 specific sentiment indicators that support your recommendation.

3. **Sentiment Risks**: What sentiment-related risks should this {risk_tolerance} investor particularly monitor?

Reference specific articles or sentiment patterns. Tailor your recommendation to the investor's risk profile.""",

            "valuation": f"""Analyze {company} ({ticker}) from a RISK-RETURN VALUATION perspective.

**INVESTOR PROFILE**: {risk_guidance}

You are the Valuation Analysis Agent. Your role is to evaluate the stock's risk characteristics, historical performance, and risk-adjusted returns.

{structured_header}

In your analysis, cover:

1. **Key Findings**: What do the risk-return metrics reveal?
   - Historical returns (annualized, cumulative)
   - Volatility measures (daily, annualized)
   - Risk metrics (Sharpe ratio, VaR, max drawdown)
   - Volume patterns

2. **Key Supporting Evidence**: List 3-5 specific risk-return metrics that support your recommendation.

3. **Risk Assessment**: What are the key risks that this {risk_tolerance} investor should focus on based on historical patterns?

Provide specific numbers for all metrics. Tailor your recommendation to the investor's risk profile."""
        }

        return prompts.get(agent_type, prompts["fundamental"])

    def _format_own_history(self, state: DebateState, agent_type: str) -> str:
        """
        Format an agent's own previous messages for context.

        Args:
            state: Current debate state
            agent_type: The agent requesting their history

        Returns:
            Formatted string of agent's previous analyses
        """
        own_messages = state.get_messages_by_agent(agent_type)
        if not own_messages:
            return ""

        history_parts = ["## YOUR PREVIOUS ANALYSIS"]
        for msg in own_messages:
            history_parts.append(f"**Round {msg.round_number}** [{msg.recommendation.value}, {msg.confidence:.0%}]:")
            # Truncate long content to save tokens
            content_preview = msg.content[:800] + "..." if len(msg.content) > 800 else msg.content
            history_parts.append(content_preview)
            history_parts.append("")

        return "\n".join(history_parts)

    def _format_current_positions(self, state: DebateState, current_agent: str) -> str:
        """
        Format current voting positions of all agents.

        Args:
            state: Current debate state
            current_agent: The agent receiving this context

        Returns:
            Formatted string of agent positions
        """
        if not state.current_votes:
            return ""

        positions = ["## CURRENT AGENT POSITIONS"]
        for agent, vote in state.current_votes.items():
            marker = "(You)" if agent == current_agent else ""
            positions.append(f"- **{agent.upper()}** {marker}: {vote.recommendation.value} ({vote.confidence:.0%})")

        return "\n".join(positions)

    def get_response_prompt(
        self,
        company: str,
        ticker: str,
        agent_type: str,
        other_agent: str,
        other_analysis: str,
        state: DebateState
    ) -> str:
        """
        Generate a response prompt for an agent to critique another's analysis.

        Args:
            company: Company name
            ticker: Stock ticker symbol
            agent_type: Type of agent responding
            other_agent: Type of agent being responded to
            other_analysis: The analysis being responded to
            state: Current debate state

        Returns:
            Formatted prompt string
        """
        risk_tolerance = state.risk_tolerance

        # Risk tolerance context
        risk_context = {
            "conservative": "CONSERVATIVE (prioritizes capital preservation and low volatility)",
            "moderate": "MODERATE (seeks balance between growth and stability)",
            "aggressive": "AGGRESSIVE (prioritizes growth, accepts high volatility)"
        }
        risk_desc = risk_context.get(risk_tolerance, risk_context["moderate"])

        # Get agent's own previous messages for continuity
        own_history = self._format_own_history(state, agent_type)

        # Get current positions of all agents
        current_positions = self._format_current_positions(state, agent_type)

        # Structured header for response
        structured_header = """**CRITICAL: You MUST start your response with this EXACT structured header format:**
```
[DECISION]
RECOMMENDATION: <BUY or HOLD or SELL>
CONFIDENCE: <number from 0 to 100>%
[/DECISION]
```

After the structured header, provide your response."""

        return f"""You are the {agent_type.upper()} Analysis Agent in a multi-agent debate about {company} ({ticker}).

**INVESTOR PROFILE**: {risk_desc}
**CURRENT ROUND**: {state.current_round}

{current_positions}

{own_history}
---

The {other_agent.upper()} Agent just provided this analysis:

---
{other_analysis}
---

{structured_header}

From your {agent_type} analysis perspective, respond to their assessment:

1. **Agreement/Disagreement**: Do you agree with their recommendation? Why or why not?

2. **Additional Insights**: What does your analysis add that they may have missed?

3. **Updated Assessment**: Considering all perspectives and your previous analysis, do you maintain or update your recommendation? If changing, explain why.

Be constructive in your critique. Acknowledge valid points while highlighting different perspectives. Reference your previous analysis if relevant."""

    def get_synthesis_prompt(self, state: DebateState) -> str:
        """
        Generate the final synthesis prompt to combine all perspectives.

        Args:
            state: Final debate state

        Returns:
            Formatted prompt string
        """
        debate_summary = state.get_debate_summary()
        risk_tolerance = state.risk_tolerance

        # Get majority vote from agents
        majority_rec = self.get_majority_recommendation(state.current_votes)
        consensus = self.calculate_consensus(state.current_votes)

        # Risk tolerance context for synthesis
        risk_context = {
            "conservative": "a CONSERVATIVE investor who prioritizes capital preservation and minimal volatility",
            "moderate": "a MODERATE investor who seeks a balance between growth and stability",
            "aggressive": "an AGGRESSIVE investor who prioritizes maximum growth potential"
        }
        risk_guidance = risk_context.get(risk_tolerance, risk_context["moderate"])

        return f"""Based on the following multi-agent debate about {state.company} ({state.ticker}), synthesize a final investment recommendation.

**INVESTOR PROFILE**: This recommendation is for {risk_guidance}.

{debate_summary}

**CRITICAL INSTRUCTION**: The agents have voted and reached a consensus. The majority recommendation is **{majority_rec.value}** with {consensus:.0%} agreement. Your final recommendation MUST reflect this majority vote.

**You MUST start your response with this EXACT structured header:**
```
[DECISION]
RECOMMENDATION: {majority_rec.value}
CONFIDENCE: <your confidence 0-100>%
[/DECISION]
```

After the structured header, provide:

1. **Synthesis Summary**: Combine the key insights from all three perspectives (Fundamental, Sentiment, Risk-return)

2. **Key Investment Points**: The 3-5 most important points for this {risk_tolerance} investor

3. **Key Risks**: The most significant risks to be aware of

4. **Risk-Adjusted Advice**: Specific guidance for this {risk_tolerance} investor (position sizing, entry timing, etc.)"""

    def strip_decision_block(self, response: str) -> str:
        """
        Remove the [DECISION]...[/DECISION] block from response for clean display.

        Args:
            response: Agent's response text

        Returns:
            Response with decision block removed
        """
        # Pattern 1: Triple backticks on their own line, then decision block, then closing backticks
        # This handles: ```\n[DECISION]...[/DECISION]\n```
        cleaned = re.sub(
            r'```\s*\n\[DECISION\].*?\[/DECISION\]\s*\n```',
            '',
            response,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Pattern 2: Triple backticks with optional language identifier, then decision block
        # This handles: ```text\n[DECISION]... or ```markdown\n[DECISION]...
        cleaned = re.sub(
            r'```\w*\s*\n?\[DECISION\].*?\[/DECISION\]\s*\n?```',
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Pattern 3: Decision block with single backticks
        cleaned = re.sub(
            r'`\[DECISION\].*?\[/DECISION\]`',
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Pattern 4: Plain decision block (no code fence)
        cleaned = re.sub(
            r'\[DECISION\].*?\[/DECISION\]',
            '',
            cleaned,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Clean up orphaned triple backticks ANYWHERE in the text (not just start/end)
        # Pattern: line that is ONLY triple backticks (with optional language identifier)
        cleaned = re.sub(r'^\s*```\w*\s*$', '', cleaned, flags=re.MULTILINE)

        # Also clean up paired empty code blocks that might remain: ```\n```
        cleaned = re.sub(r'```\s*\n\s*```', '', cleaned)

        # Clean up multiple consecutive newlines (more than 2)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Clean up extra whitespace at the start
        cleaned = cleaned.strip()
        return cleaned

    def parse_recommendation_from_response(self, response: str) -> Recommendation:
        """
        Extract recommendation (BUY/HOLD/SELL) from agent response.

        Args:
            response: Agent's response text

        Returns:
            Parsed Recommendation enum
        """
        response_upper = response.upper()

        # Method 1 (PRIORITY): Look for structured [DECISION] block
        decision_match = re.search(
            r'\[DECISION\].*?RECOMMENDATION:\s*(BUY|HOLD|SELL)',
            response_upper,
            re.DOTALL
        )
        if decision_match:
            return Recommendation(decision_match.group(1))

        # Method 2: Look for explicit recommendation patterns (same line)
        patterns = [
            r"RECOMMENDATION[:\s]+(\bBUY\b|\bHOLD\b|\bSELL\b)",
            r"RECOMMEND[:\s]+(\bBUY\b|\bHOLD\b|\bSELL\b)",
            r"I RECOMMEND[:\s]+(\bBUY\b|\bHOLD\b|\bSELL\b)",
            r"(\bBUY\b|\bHOLD\b|\bSELL\b)\s+RECOMMENDATION",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                rec_str = match.group(1)
                return Recommendation(rec_str)

        # Method 3: Find "Recommendation" section and get FIRST recommendation word after it
        # This handles multi-line formats like "4. Recommendation:\n\nHOLD. ..."
        rec_section_match = re.search(
            r"(?:\d+\.\s*)?RECOMMENDATION[:\s]*",
            response_upper
        )
        if rec_section_match:
            # Get text after the "Recommendation" header (next 200 chars)
            start_pos = rec_section_match.end()
            section_text = response_upper[start_pos:start_pos + 200]
            # Find the first BUY/HOLD/SELL in this section
            first_rec_match = re.search(r"\b(BUY|HOLD|SELL)\b", section_text)
            if first_rec_match:
                return Recommendation(first_rec_match.group(1))

        # Method 4: Look for bold recommendation patterns like **HOLD** or **BUY**
        bold_match = re.search(r"\*\*(BUY|HOLD|SELL)\*\*", response_upper)
        if bold_match:
            return Recommendation(bold_match.group(1))

        # Method 5: Look for recommendation at the start of a sentence
        sentence_patterns = [
            r"(?:^|\.\s+)(BUY|HOLD|SELL)\.",
            r"(?:^|\n)\s*(BUY|HOLD|SELL)\s*[-–:]",
        ]
        for pattern in sentence_patterns:
            match = re.search(pattern, response_upper)
            if match:
                return Recommendation(match.group(1))

        # Fallback: count occurrences (last resort)
        buy_count = response_upper.count("BUY")
        sell_count = response_upper.count("SELL")
        hold_count = response_upper.count("HOLD")

        if buy_count > sell_count and buy_count > hold_count:
            return Recommendation.BUY
        elif sell_count > buy_count and sell_count > hold_count:
            return Recommendation.SELL
        else:
            return Recommendation.HOLD

    def parse_confidence_from_response(self, response: str) -> float:
        """
        Extract confidence level from agent response.

        Args:
            response: Agent's response text

        Returns:
            Confidence as float (0.0 to 1.0)
        """
        response_upper = response.upper()

        # Method 1 (PRIORITY): Look for structured [DECISION] block
        decision_match = re.search(
            r'\[DECISION\].*?CONFIDENCE:\s*(\d+)',
            response_upper,
            re.DOTALL
        )
        if decision_match:
            value = float(decision_match.group(1))
            # Convert to 0-1 range if given as percentage
            if value > 1:
                value = value / 100
            return min(1.0, max(0.0, value))

        # Method 2: Look for percentage patterns
        patterns = [
            r"CONFIDENCE[:\s]+(\d+)%",
            r"CONFIDENCE LEVEL[:\s]+(\d+)%",
            r"(\d+)%\s+CONFIDEN",
            r"CONFIDENCE[:\s]+(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response_upper)
            if match:
                value = float(match.group(1))
                # Convert to 0-1 range if given as percentage
                if value > 1:
                    value = value / 100
                return min(1.0, max(0.0, value))

        # Default confidence
        return 0.5

    def parse_key_factors_from_response(self, response: str) -> List[str]:
        """
        Extract key factors/evidence from agent response.

        Args:
            response: Agent's response text

        Returns:
            List of key factor strings
        """
        factors = []

        # Look for bullet points or numbered lists
        lines = response.split('\n')
        in_evidence_section = False

        # Keywords that indicate start of key points section
        start_keywords = [
            'key investment points',
            'key points',
            'investment points',
            'important points',
            'evidence',
            'supporting',
            'factors'
        ]

        # Keywords that indicate end of section (start of next section)
        end_keywords = [
            'key risks',
            'risks',
            'confidence',
            'risk-adjusted',
            'recommendation'
        ]

        for line in lines:
            line = line.strip()
            line_lower = line.lower()

            # Check if we're entering a key points section
            if any(keyword in line_lower for keyword in start_keywords) and ('risk' not in line_lower):
                in_evidence_section = True
                continue

            # Check for new section start (end of key points)
            if in_evidence_section:
                if any(keyword in line_lower for keyword in end_keywords):
                    in_evidence_section = False
                    continue
                if line.startswith('#') or (line.endswith(':') and len(line) < 50 and any(kw in line_lower for kw in end_keywords)):
                    in_evidence_section = False
                    continue

            # Extract bullet points
            if in_evidence_section and (line.startswith('-') or line.startswith('*') or re.match(r'^\d+\.', line)):
                # Clean up the line
                factor = re.sub(r'^[-*\d.]+\s*', '', line).strip()
                # Remove markdown bold markers
                factor = re.sub(r'\*\*([^*]+)\*\*', r'\1', factor)
                if factor and len(factor) > 10:
                    factors.append(factor)

        return factors[:5]  # Limit to 5 factors

    def parse_reasoning_from_response(self, response: str) -> str:
        """
        Extract the actual reasoning/analysis text from agent response,
        skipping structured headers like Recommendation:, Confidence:, etc.

        Args:
            response: Agent's response text

        Returns:
            Extracted reasoning text
        """
        # Try to find the reasoning section
        lines = response.split('\n')
        reasoning_lines = []
        in_reasoning = False
        skip_headers = ['recommendation:', 'confidence:', 'reasoning:']

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Skip empty lines at the start
            if not reasoning_lines and not line_stripped:
                continue

            # Skip header lines like "Recommendation: BUY", "Confidence: 75%"
            if any(line_lower.startswith(header) for header in skip_headers):
                # If it's "Reasoning:", start capturing after this
                if line_lower.startswith('reasoning:'):
                    in_reasoning = True
                    # Check if there's content after "Reasoning:"
                    after_header = line_stripped[10:].strip()
                    if after_header:
                        reasoning_lines.append(after_header)
                continue

            # If we haven't seen "Reasoning:" yet, check if this looks like analysis content
            if not in_reasoning:
                # Skip numbered sections at the very start (like "1. Agreement/Disagreement:")
                if re.match(r'^\d+\.\s+\w', line_stripped) and len(reasoning_lines) == 0:
                    in_reasoning = True

            # Once we're in reasoning mode, collect content
            if in_reasoning or len(reasoning_lines) > 0 or (
                line_stripped and not any(line_lower.startswith(h) for h in skip_headers)
            ):
                reasoning_lines.append(line_stripped)
                in_reasoning = True

        reasoning = '\n'.join(reasoning_lines).strip()

        # If we didn't find structured reasoning, fall back to the response
        # but try to skip the first few header lines
        if not reasoning:
            # Skip first few lines if they're headers
            skip_count = 0
            for line in lines[:5]:
                if any(h in line.lower() for h in skip_headers) or not line.strip():
                    skip_count += 1
                else:
                    break
            reasoning = '\n'.join(lines[skip_count:]).strip()

        return reasoning[:500] if reasoning else response[:500]

    def create_vote_from_response(
        self,
        agent_type: str,
        response: str
    ) -> AgentVote:
        """
        Parse an agent response into an AgentVote.

        Args:
            agent_type: Type of agent
            response: Agent's response text

        Returns:
            Parsed AgentVote
        """
        return AgentVote(
            agent_type=agent_type,
            recommendation=self.parse_recommendation_from_response(response),
            confidence=self.parse_confidence_from_response(response),
            reasoning=self.parse_reasoning_from_response(response),
            key_factors=self.parse_key_factors_from_response(response)
        )

    def create_message_from_response(
        self,
        agent_type: str,
        response: str,
        round_number: int,
        is_response: bool = False,
        responding_to: str = None
    ) -> DebateMessage:
        """
        Create a DebateMessage from an agent's response.

        Args:
            agent_type: Type of agent
            response: Agent's response text
            round_number: Current round number
            is_response: Whether this is responding to another agent
            responding_to: Agent being responded to

        Returns:
            Created DebateMessage
        """
        return DebateMessage(
            agent_type=agent_type,
            content=self.strip_decision_block(response),
            confidence=self.parse_confidence_from_response(response),
            recommendation=self.parse_recommendation_from_response(response),
            round_number=round_number,
            is_response=is_response,
            responding_to=responding_to
        )

    def create_final_recommendation(
        self,
        state: DebateState,
        synthesis_response: str
    ) -> FinalRecommendation:
        """
        Create the final recommendation from synthesis.

        Args:
            state: Final debate state
            synthesis_response: Response from synthesis prompt

        Returns:
            FinalRecommendation with all details
        """
        recommendation = self.parse_recommendation_from_response(synthesis_response)
        confidence = self.parse_confidence_from_response(synthesis_response)
        consensus = self.calculate_weighted_consensus(state.current_votes)

        # Extract key points and risks from synthesis
        key_points = self.parse_key_factors_from_response(synthesis_response)

        # Improved risk extraction
        risks = []
        lines = synthesis_response.split('\n')
        in_risk_section = False

        # Keywords that indicate start of risks section
        risk_start_keywords = ['key risks', 'risks:', 'significant risks']
        # Keywords that indicate end of risks section
        risk_end_keywords = ['confidence', 'risk-adjusted advice', 'guidance', 'position sizing']

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Check if entering risk section
            if any(keyword in line_lower for keyword in risk_start_keywords):
                in_risk_section = True
                continue

            # Check if leaving risk section
            if in_risk_section:
                if any(keyword in line_lower for keyword in risk_end_keywords):
                    in_risk_section = False
                    continue
                # Also end on numbered sections that aren't risks
                if re.match(r'^\d+\.', line_stripped) and 'risk' not in line_lower:
                    if any(kw in line_lower for kw in risk_end_keywords):
                        in_risk_section = False
                        continue

            # Extract bullet points in risk section
            if in_risk_section and (line_stripped.startswith('-') or line_stripped.startswith('*') or re.match(r'^\d+\.', line_stripped)):
                risk = re.sub(r'^[-*\d.]+\s*', '', line_stripped).strip()
                # Remove markdown bold markers
                risk = re.sub(r'\*\*([^*]+)\*\*', r'\1', risk)
                if risk and len(risk) > 10:
                    risks.append(risk)

        return FinalRecommendation(
            recommendation=recommendation,
            confidence=confidence,
            consensus_level=consensus,
            summary=self.strip_decision_block(synthesis_response),
            key_points=key_points,
            risks=risks[:5],
            agent_breakdown=state.current_votes.copy(),
            debate_state=state
        )
