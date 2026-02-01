"""
Debate Orchestrator for multi-agent stock analysis.

This module provides the central orchestrator that:
- Routes user queries to appropriate agent(s)
- Manages the multi-agent debate process
- Coordinates between Fundamental, Sentiment, and Valuation agents
- Handles round-robin turn taking
- Synthesizes final recommendations
"""
from typing import Dict, Optional, Callable, Generator, Any, Tuple, List
from datetime import datetime
import re
import time

from .base import get_llm, BaseAgentMixin
from .fundamental_agent import FundamentalAgent, create_fundamental_agent
from .sentiment_agent import SentimentAgent, create_sentiment_agent
from .valuation_agent import ValuationAgent, create_valuation_agent
from .debate_state import (
    DebateState,
    DebateMessage,
    AgentVote,
    RoundResult,
    Recommendation,
    FinalRecommendation
)
from .debate_manager import DebateManager
from .debate_metrics import DebateMetrics
from utils.ticker_resolver import resolve_ticker_symbol


class DebateOrchestrator:
    """
    Central orchestrator for multi-agent stock analysis debates.

    Manages the debate flow between specialist agents and synthesizes
    their analyses into a final investment recommendation.
    """

    def __init__(
        self,
        llm=None,
        model_name: str = None,
        max_rounds: int = 5,
        min_turns_per_agent: int = 2,
        consensus_threshold: float = 0.75,
        agent_order: list = None,
        on_message_callback: Callable[[DebateMessage], None] = None,
        on_round_complete_callback: Callable[[RoundResult], None] = None,
        track_metrics: bool = True
    ):
        """
        Initialize the debate orchestrator.

        Args:
            llm: Language model instance (shared across all agents)
            model_name: Name of the model to use (supports Gemini, Groq, etc.)
                       If provided, overrides the llm parameter
            max_rounds: Maximum number of debate rounds
            min_turns_per_agent: Minimum turns each agent must take
            consensus_threshold: Required consensus level (0.0-1.0)
            agent_order: Custom order for round-robin turns
            on_message_callback: Called when an agent posts a message
            on_round_complete_callback: Called when a round completes
            track_metrics: Whether to track efficiency metrics (default: True)
        """
        # Initialize LLM
        if model_name:
            self.llm = get_llm(model_name=model_name)
            self.model_name = model_name
        else:
            self.llm = llm or get_llm()
            self.model_name = "gemini-2.0-flash"  # Default

        self.track_metrics = track_metrics

        self.max_rounds = max_rounds
        self.min_turns_per_agent = min_turns_per_agent
        self.consensus_threshold = consensus_threshold

        # Initialize debate manager
        self.debate_manager = DebateManager(
            agent_order=agent_order,
            max_rounds=max_rounds,
            min_turns_per_agent=min_turns_per_agent,
            consensus_threshold=consensus_threshold
        )

        # Initialize specialist agents
        self.agents: Dict[str, BaseAgentMixin] = {
            "fundamental": create_fundamental_agent(llm=self.llm),
            "sentiment": create_sentiment_agent(llm=self.llm),
            "valuation": create_valuation_agent(llm=self.llm)
        }

        # Callbacks for UI updates
        self.on_message_callback = on_message_callback
        self.on_round_complete_callback = on_round_complete_callback

        # Current debate state
        self.current_debate: Optional[DebateState] = None

        # Current metrics tracking
        self.current_metrics: Optional[DebateMetrics] = None

    def classify_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to classify whether a query requires multi-agent debate or single agent,
        and infer the user's risk tolerance from the query language.

        Args:
            query: User's query string

        Returns:
            Dictionary with:
            - is_investment_related: bool (whether query is about stocks/investments)
            - needs_debate: bool
            - agent_type: str or None (if single agent needed)
            - company: str or None (extracted company/ticker)
            - risk_tolerance: str (conservative/moderate/aggressive)
            - reasoning: str (explanation of classification)
        """
        classification_prompt = f"""You are a query router for a stock analysis system. Analyze the user query and determine:

1. Whether the query is INVESTMENT-RELATED (about stocks, companies, financial analysis, or investment decisions)
2. Whether it requires a MULTI-AGENT DEBATE (multiple specialist perspectives) or a SINGLE AGENT response
3. If single agent, which specialist should handle it
4. What company/stock is being asked about (if any)
5. The user's implied RISK TOLERANCE based on their language and intent

INVESTMENT-RELATED queries include:
- Questions about specific stocks, companies, or financial instruments
- Requests for investment advice, recommendations, or analysis
- Questions about financial metrics, valuations, or market conditions
- Queries about portfolio management or investment strategies
- Questions about company fundamentals, sentiment, or valuation

NOT INVESTMENT-RELATED queries include:
- General knowledge questions (history, science, weather, etc.)
- Personal advice unrelated to finance (relationships, health, etc.)
- Technical questions about non-financial topics
- Casual conversation or greetings
- Questions about unrelated industries without investment context

MULTI-AGENT DEBATE is needed when:
- Query asks for investment decisions or recommendations (buy/sell/hold advice)
- Query requires weighing conflicting factors or multiple perspectives
- Query asks for comprehensive analysis that would benefit from specialist disagreement/consensus
- Query involves risk assessment or strategic financial decisions
- Query asks for opinions, advice, or evaluations with multiple valid perspectives
- Query involves portfolio management decisions
- Query asks to "analyze," "evaluate," "assess," or "recommend" regarding investments

SINGLE AGENT is sufficient when:
- Query asks for simple factual information regarding price or market data → use "valuation" agent
- Query is about specific technical/valuation analysis, price history, or stock charts → use "valuation" agent
- Query is about news sentiment only → use "sentiment" agent
- Query is about fundamental/financial data only (revenue, earnings, balance sheet) → use "fundamental" agent
- Query asks for definitions or explanations
- Query is a simple informational request that doesn't require decision-making

RISK TOLERANCE inference guidelines:
- CONSERVATIVE: User mentions safety, stability, capital preservation, low risk, dividend income, retirement, risk-averse, secure, defensive, blue-chip, or expresses worry about losses
- MODERATE: User seeks balance, reasonable returns, some growth, diversification, or doesn't express strong risk preferences (this is the default)
- AGGRESSIVE: User mentions high growth, maximum returns, willing to take risks, speculative, momentum, high volatility is acceptable, or seeks rapid gains

USER QUERY: {query}

Respond in this exact format:
IS_INVESTMENT_RELATED: [true/false]
NEEDS_DEBATE: [true/false]
AGENT: [fundamental/sentiment/valuation/none]
COMPANY: [comma-separated list of companies, e.g. "Maybank, CIMB", or "none" if not specified]
RISK_TOLERANCE: [conservative/moderate/aggressive]
REASONING: [brief explanation including why you chose this classification]"""

        # Use LLM to classify
        response = self.llm.invoke(classification_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        # Handle different response formats (Gemini 2.5 Pro returns list of content blocks)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif hasattr(block, 'text'):
                    text_parts.append(block.text)
            response_text = '\n'.join(text_parts)
        else:
            response_text = content

        # Parse the response
        result = {
            "is_investment_related": True,  # Default to investment-related
            "needs_debate": True,  # Default to debate
            "agent_type": None,
            "company": None,
            "risk_tolerance": "moderate",  # Default to moderate
            "reasoning": ""
        }

        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("IS_INVESTMENT_RELATED:"):
                value = line.split(":", 1)[1].strip().lower()
                # Strip brackets if present
                value = value.strip("[]")
                result["is_investment_related"] = value == "true"
            elif line.startswith("NEEDS_DEBATE:"):
                value = line.split(":", 1)[1].strip().lower()
                # Strip brackets if present (LLM sometimes returns [true] instead of true)
                value = value.strip("[]")
                result["needs_debate"] = value == "true"
            elif line.startswith("AGENT:"):
                value = line.split(":", 1)[1].strip().lower()
                # Strip brackets if present (LLM sometimes returns [sentiment] instead of sentiment)
                value = value.strip("[]")
                if value != "none" and value in ["fundamental", "sentiment", "valuation"]:
                    result["agent_type"] = value
            elif line.startswith("COMPANY:"):
                value = line.split(":", 1)[1].strip()
                # Strip brackets if present
                value = value.strip("[]")
                if value.lower() != "none":
                    # Check for comma-separated list
                    if "," in value:
                        # Keep as list of strings
                        result["company"] = [c.strip() for c in value.split(",")]
                    else:
                        result["company"] = value
            elif line.startswith("RISK_TOLERANCE:"):
                value = line.split(":", 1)[1].strip().lower()
                # Strip brackets if present
                value = value.strip("[]")
                if value in ["conservative", "moderate", "aggressive"]:
                    result["risk_tolerance"] = value
            elif line.startswith("REASONING:"):
                try:
                    result["reasoning"] = line.split(":", 1)[1].strip()
                except IndexError:
                    pass

        return result

    def route_query(self, query: str, company: str = None, risk_tolerance: str = None) -> Dict[str, Any]:
        """
        Main entry point - routes user query to appropriate agent(s).

        Args:
            query: User's query string
            company: Company name/ticker (optional, will try to extract from query)
            risk_tolerance: Override for risk tolerance (optional, otherwise inferred from query)

        Returns:
            Dictionary with:
            - response: The analysis response
            - route_type: "debate" or "single_agent"
            - agent_used: Agent type(s) used
            - recommendation: Final recommendation (if applicable)
            - risk_tolerance: The risk tolerance used for analysis
            - classification: The query classification details
        """
        # Classify the query using LLM
        classification = self.classify_query(query)

        # Use provided company or extracted one
        target_company = company or classification.get("company")

        # Use provided risk tolerance or inferred one
        effective_risk_tolerance = risk_tolerance or classification.get("risk_tolerance", "moderate")

        if classification["needs_debate"]:
            if not target_company:
                return {
                    "response": "Please specify which company or stock you'd like me to analyze.",
                    "route_type": "error",
                    "agent_used": None,
                    "recommendation": None,
                    "risk_tolerance": effective_risk_tolerance,
                    "classification": classification
                }

            # Check for multi-company comparison
            companies = []
            if isinstance(target_company, list):
                companies = target_company
            elif isinstance(target_company, str) and "," in target_company:
                companies = [c.strip() for c in target_company.split(",")]
            
            if len(companies) > 1:
                # Run comparative debate
                try:
                    final_rec = self.run_comparative_debate(companies, risk_tolerance=effective_risk_tolerance)
                    return {
                        "response": final_rec.summary,
                        "route_type": "comparison",
                        "agent_used": ["fundamental", "sentiment", "valuation"],
                        "recommendation": final_rec.recommendation.value,
                        "confidence": final_rec.confidence,
                        "consensus": final_rec.consensus_level,
                        "risk_tolerance": effective_risk_tolerance,
                        "full_result": final_rec,
                        "classification": classification,
                        "companies": companies
                    }
                except Exception as e:
                    return {
                        "response": f"Error during comparative analysis: {str(e)}",
                        "route_type": "error",
                        "agent_used": None,
                        "recommendation": None,
                        "risk_tolerance": effective_risk_tolerance,
                        "classification": classification
                    }

            # Run standard single-company debate
            try:
                target_company_str = target_company[0] if isinstance(target_company, list) else target_company
                final_rec = self.run_debate(target_company_str, risk_tolerance=effective_risk_tolerance)
                return {
                    "response": final_rec.summary,
                    "route_type": "debate",
                    "agent_used": ["fundamental", "sentiment", "valuation"],
                    "recommendation": final_rec.recommendation.value,
                    "confidence": final_rec.confidence,
                    "consensus": final_rec.consensus_level,
                    "risk_tolerance": effective_risk_tolerance,
                    "full_result": final_rec,
                    "classification": classification
                }
            except Exception as e:
                return {
                    "response": f"Error during analysis: {str(e)}",
                    "route_type": "error",
                    "agent_used": None,
                    "recommendation": None,
                    "risk_tolerance": effective_risk_tolerance,
                    "classification": classification
                }
        else:
            # Route to single agent
            agent_type = classification.get("agent_type") or "fundamental"
            result = self._handle_single_agent_query(query, agent_type, target_company, effective_risk_tolerance)
            result["classification"] = classification
            return result

    def _handle_single_agent_query(
        self,
        query: str,
        agent_type: str,
        company: str = None,
        risk_tolerance: str = "moderate"
    ) -> Dict[str, Any]:
        """
        Handle a query that only requires a single agent.

        Args:
            query: User's query string
            agent_type: Type of agent to use
            company: Company name/ticker (optional)
            risk_tolerance: User's risk tolerance level

        Returns:
            Response dictionary
        """
        agent = self.agents.get(agent_type)
        if not agent:
            # Fallback to fundamental agent
            agent = self.agents["fundamental"]
            agent_type = "fundamental"

        # Build enhanced query with context
        context_parts = []
        if company:
            context_parts.append(f"Company: {company}")
        context_parts.append(f"Investor Risk Tolerance: {risk_tolerance}")

        if context_parts:
            enhanced_query = f"[Context: {', '.join(context_parts)}]\n\n{query}"
        else:
            enhanced_query = query

        thread_id = f"single-{agent_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        response = agent.chat(enhanced_query, thread_id=thread_id)

        return {
            "response": response,
            "route_type": "single_agent",
            "agent_used": agent_type,
            "risk_tolerance": risk_tolerance,
            "recommendation": None  # Single agent doesn't give formal recommendation
        }

    def _handle_single_agent_query_stream(
        self,
        query: str,
        agent_type: str,
        company: str = None,
        risk_tolerance: str = "moderate"
    ) -> Generator[str, None, Dict[str, Any]]:
        """
        Handle a single agent query with streaming response.

        Args:
            query: User's query string
            agent_type: Type of agent to use
            company: Company name/ticker (optional)
            risk_tolerance: User's risk tolerance level

        Yields:
            Response text chunks as they are generated

        Returns:
            Final response dictionary (via StopIteration.value)
        """
        agent = self.agents.get(agent_type)
        if not agent:
            agent = self.agents["fundamental"]
            agent_type = "fundamental"

        # Build enhanced query with context
        context_parts = []
        if company:
            context_parts.append(f"Company: {company}")
        context_parts.append(f"Investor Risk Tolerance: {risk_tolerance}")

        if context_parts:
            enhanced_query = f"[Context: {', '.join(context_parts)}]\n\n{query}"
        else:
            enhanced_query = query

        thread_id = f"single-{agent_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Collect full response while streaming
        full_response = []

        # Use streaming if available
        if hasattr(agent, 'chat_stream'):
            for chunk in agent.chat_stream(enhanced_query, thread_id=thread_id):
                full_response.append(chunk)
                yield chunk
        else:
            # Fallback to non-streaming
            response = agent.chat(enhanced_query, thread_id=thread_id)
            full_response.append(response)
            yield response

        # Return final result
        return {
            "response": "".join(full_response),
            "route_type": "single_agent",
            "agent_used": agent_type,
            "risk_tolerance": risk_tolerance,
            "recommendation": None
        }

    def resolve_company(self, query: str) -> dict:
        """
        Resolve a company name to ticker symbol.

        Args:
            query: Company name or ticker symbol

        Returns:
            Resolution result dictionary
        """
        return resolve_ticker_symbol.invoke({"company_name": query})

    def start_debate(
        self,
        company: str,
        risk_tolerance: str = "moderate"
    ) -> Generator[DebateMessage, None, FinalRecommendation]:
        """
        Start a multi-agent debate about a company.

        This is a generator that yields each DebateMessage as it's created,
        allowing real-time updates to the UI.

        Args:
            company: Company name or ticker symbol
            risk_tolerance: User's risk tolerance (conservative/moderate/aggressive)

        Yields:
            DebateMessage objects as they are created

        Returns:
            FinalRecommendation when debate completes
        """
        # Resolve company to ticker
        resolution = self.resolve_company(company)
        if not resolution.get("success"):
            raise ValueError(f"Could not resolve company: {company}. {resolution.get('error', '')}")

        ticker = resolution["ticker"]
        company_name = resolution.get("company_name", company)

        # Update metrics with resolved ticker
        if self.track_metrics and self.current_metrics:
            self.current_metrics.ticker = ticker
            self.current_metrics.company = company_name

        # Store risk tolerance for use in prompts
        self._current_risk_tolerance = risk_tolerance

        # Create debate state
        self.current_debate = self.debate_manager.create_debate_state(
            company=company_name,
            ticker=ticker,
            risk_tolerance=risk_tolerance
        )

        # Run initial round (each agent does independent analysis)
        yield from self._run_initial_round()

        # Run subsequent rounds (agents respond to each other)
        while self.debate_manager.should_continue_debate(self.current_debate):
            self.current_debate.current_round += 1
            yield from self._run_response_round()

            # Create round result
            round_result = self.debate_manager.create_round_result(
                self.current_debate,
                self.current_debate.current_round
            )
            self.current_debate.round_results.append(round_result)

            if self.on_round_complete_callback:
                self.on_round_complete_callback(round_result)

        # Synthesize final recommendation
        final_rec = self._synthesize_recommendation()
        self.current_debate.finish_debate(final_rec.recommendation)

        return final_rec

    def run_comparative_debate(self, companies: List[str], risk_tolerance: str = "moderate") -> FinalRecommendation:
        """
        Run sequential debates for multiple companies and synthesize a comparison.

        Args:
            companies: List of company names/tickers
            risk_tolerance: User's risk tolerance

        Returns:
            FinalRecommendation object containing the comparative analysis
        """
        results = {}
        
        # Run debates sequentially to avoid rate limits
        for company in companies:
            print(f"Running debate for {company}...")
            try:
                # Using existing run_debate method
                # This handles resolving ticker, running rounds, etc.
                rec = self.run_debate(company, risk_tolerance=risk_tolerance)
                results[company] = rec
            except Exception as e:
                print(f"Error analyzing {company}: {e}")
                # Continue with other companies if one fails
                continue
        
        if not results:
            raise ValueError("All debates failed.")
            
        return self._synthesize_comparison(results, risk_tolerance)

    def _synthesize_comparison(self, results: Dict[str, FinalRecommendation], risk_tolerance: str) -> FinalRecommendation:
        """
        Synthesize a comparative analysis from multiple debate results.
        """
        # Get comparison prompt
        prompt = self.debate_manager.get_comparison_synthesis_prompt(results, risk_tolerance)
        
        # Invoke LLM
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Handle list response format
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif hasattr(block, 'text'):
                    text_parts.append(block.text)
            comparison_text = '\n'.join(text_parts)
        else:
            comparison_text = content
            
        # Determine "winner" based on confidence/recommendation logic
        # For simplicity, we create a specialized FinalRecommendation
        # logic could be expanded to parse the "Winner" from text
        
        best_company = None
        highest_score = -1
        
        for company, rec in results.items():
            score = 0
            if rec.recommendation == Recommendation.BUY:
                score += 2
            elif rec.recommendation == Recommendation.HOLD:
                score += 1
            
            score += rec.confidence
            
            if score > highest_score:
                highest_score = score
                best_company = company
                
        # Create a "Comparative" recommendation object
        # We misuse the standard object slightly to fit the UI contract
        return FinalRecommendation(
            recommendation=Recommendation.BUY if best_company else Recommendation.HOLD, # Placeholder
            confidence=0.0, # Placeholder
            consensus_level=0.0,
            summary=comparison_text, # The full comparative report
            key_points=[],
            risks=[],
            agent_breakdown={}, # Not applicable for comparison top-level
            debate_state=None 
        )

    def run_debate(self, company: str, risk_tolerance: str = "moderate") -> FinalRecommendation:
        """
        Run a complete debate and return the final recommendation.

        Non-generator version for simpler usage.

        Args:
            company: Company name or ticker symbol
            risk_tolerance: User's risk tolerance (conservative/moderate/aggressive)

        Returns:
            FinalRecommendation with analysis results and metrics
        """
        # Initialize metrics tracking if enabled
        if self.track_metrics:
            # We'll get ticker after resolution, for now use company name
            self.current_metrics = DebateMetrics(
                company=company,
                ticker="",  # Will be updated after resolution
                model_name=self.model_name
            )

        # Consume the generator to get final result
        debate_gen = self.start_debate(company, risk_tolerance=risk_tolerance)
        try:
            while True:
                message = next(debate_gen)
                if self.on_message_callback:
                    self.on_message_callback(message)
        except StopIteration as e:
            final_result = e.value

            # Finalize metrics
            if self.track_metrics and self.current_metrics:
                self.current_metrics.finalize()
                # Attach metrics to final recommendation
                final_result.metrics = self.current_metrics

            return final_result

    def _run_initial_round(self) -> Generator[DebateMessage, None, None]:
        """
        Run the initial round where each agent performs independent analysis.

        Yields:
            DebateMessage from each agent
        """
        self.current_debate.current_round = 1
        company = self.current_debate.company
        ticker = self.current_debate.ticker
        risk_tolerance = self.current_debate.risk_tolerance

        # Track round start
        if self.track_metrics and self.current_metrics:
            self.current_metrics.start_round()

        for agent_type in self.debate_manager.agent_order:
            # Get initial analysis prompt with risk tolerance
            prompt = self.debate_manager.get_initial_analysis_prompt(
                company, ticker, agent_type, risk_tolerance
            )

            # Get agent response with timing
            thread_id = f"debate-{ticker}-{agent_type}-r1"
            agent = self.agents[agent_type]

            agent_start = time.time()

            # Always use chat() to avoid double-invoke bug
            response = agent.chat(prompt, thread_id=thread_id)

            agent_duration = time.time() - agent_start

            # Track agent metrics (time-based only)
            if self.track_metrics and self.current_metrics:
                self.current_metrics.record_agent_response(agent_type, agent_duration, tokens=0)

            # Create message and vote
            message = self.debate_manager.create_message_from_response(
                agent_type=agent_type,
                response=response,
                round_number=1,
                is_response=False
            )
            vote = self.debate_manager.create_vote_from_response(agent_type, response)

            # Update state
            self.current_debate.add_message(message)
            self.current_debate.update_vote(vote)

            # Note: Callback is handled in run_debate() to avoid duplicates
            yield message

        # Track round end
        if self.track_metrics and self.current_metrics:
            self.current_metrics.end_round()

    def _run_response_round(self) -> Generator[DebateMessage, None, None]:
        """
        Run a response round where agents critique each other's analyses.

        Yields:
            DebateMessage from each agent
        """
        round_num = self.current_debate.current_round
        company = self.current_debate.company
        ticker = self.current_debate.ticker

        # Track round start
        if self.track_metrics and self.current_metrics:
            self.current_metrics.start_round()

        for i, agent_type in enumerate(self.debate_manager.agent_order):
            # Get the previous agent's analysis to respond to
            prev_agent_idx = (i - 1) % len(self.debate_manager.agent_order)
            prev_agent_type = self.debate_manager.agent_order[prev_agent_idx]
            prev_message = self.current_debate.get_latest_message_by_agent(prev_agent_type)

            if not prev_message:
                continue

            # Get response prompt
            prompt = self.debate_manager.get_response_prompt(
                company=company,
                ticker=ticker,
                agent_type=agent_type,
                other_agent=prev_agent_type,
                other_analysis=prev_message.content,
                state=self.current_debate
            )

            # Get agent response with timing
            thread_id = f"debate-{ticker}-{agent_type}-r{round_num}"
            agent = self.agents[agent_type]

            agent_start = time.time()

            # Always use chat() to avoid double-invoke bug
            response = agent.chat(prompt, thread_id=thread_id)

            agent_duration = time.time() - agent_start

            # Track agent metrics (time-based only)
            if self.track_metrics and self.current_metrics:
                self.current_metrics.record_agent_response(agent_type, agent_duration, tokens=0)

            # Create message and vote
            message = self.debate_manager.create_message_from_response(
                agent_type=agent_type,
                response=response,
                round_number=round_num,
                is_response=True,
                responding_to=prev_agent_type
            )
            vote = self.debate_manager.create_vote_from_response(agent_type, response)

            # Update state
            self.current_debate.add_message(message)
            self.current_debate.update_vote(vote)

            # Check if consensus reached
            if self.track_metrics and self.current_metrics:
                consensus = self.current_debate.calculate_consensus()
                if consensus >= self.consensus_threshold and not self.current_metrics.consensus_reached:
                    self.current_metrics.mark_consensus()

            # Note: Callback is handled in run_debate() to avoid duplicates
            yield message

        # Track round end
        if self.track_metrics and self.current_metrics:
            self.current_metrics.end_round()

    def _synthesize_recommendation(self) -> FinalRecommendation:
        """
        Synthesize final recommendation from all agent analyses.

        Returns:
            FinalRecommendation with combined analysis
        """
        # Get synthesis prompt
        prompt = self.debate_manager.get_synthesis_prompt(self.current_debate)

        # Use Orchestrator LLM for neutral synthesis
        response = self.llm.invoke(prompt)
        
        # Extract content from response
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Handle different response formats (Gemini 2.5 Pro returns list of content blocks)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif hasattr(block, 'text'):
                    text_parts.append(block.text)
            synthesis_response = '\n'.join(text_parts)
        else:
            synthesis_response = content

        # Create final recommendation
        final_rec = self.debate_manager.create_final_recommendation(
            self.current_debate,
            synthesis_response
        )

        return final_rec

    def get_debate_transcript(self) -> str:
        """
        Get a formatted transcript of the current debate.

        Returns:
            Formatted string transcript
        """
        if not self.current_debate:
            return "No debate in progress"

        lines = []
        lines.append("=" * 60)
        lines.append(f"DEBATE TRANSCRIPT: {self.current_debate.company} ({self.current_debate.ticker})")
        lines.append(f"Started: {self.current_debate.started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 60)
        lines.append("")

        current_round = 0
        for message in self.current_debate.messages:
            if message.round_number != current_round:
                current_round = message.round_number
                lines.append(f"\n--- ROUND {current_round} ---\n")

            lines.append(f"[{message.agent_type.upper()}] ({message.recommendation.value}, {message.confidence:.0%} confidence)")
            if message.is_response:
                lines.append(f"  Responding to: {message.responding_to}")
            lines.append("-" * 40)
            lines.append(message.content)
            lines.append("")

        if self.current_debate.final_recommendation:
            lines.append("=" * 60)
            lines.append("FINAL RECOMMENDATION")
            lines.append("=" * 60)
            lines.append(f"Recommendation: {self.current_debate.final_recommendation.value}")
            lines.append(f"Consensus: {self.current_debate.calculate_consensus():.0%}")

        return "\n".join(lines)

    def get_vote_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current votes.

        Returns:
            Dictionary with vote summary
        """
        if not self.current_debate:
            return {}

        votes = self.current_debate.current_votes
        consensus = self.debate_manager.calculate_consensus(votes)
        weighted_consensus = self.debate_manager.calculate_weighted_consensus(votes)
        leading = self.debate_manager.get_majority_recommendation(votes)

        return {
            "votes": {k: v.to_dict() for k, v in votes.items()},
            "consensus": consensus,
            "weighted_consensus": weighted_consensus,
            "leading_recommendation": leading.value,
            "agents_agree": consensus >= self.consensus_threshold
        }

    def export_debate(self) -> dict:
        """
        Export the complete debate data for storage/analysis.

        Returns:
            Dictionary containing full debate data
        """
        if not self.current_debate:
            return {}

        return self.current_debate.to_dict()

def create_debate_orchestrator(
    llm=None,
    model_name: str = None,
    max_rounds: int = 5,
    consensus_threshold: float = 0.75,
    on_message_callback: Callable = None
) -> DebateOrchestrator:
    """
    Factory function to create a DebateOrchestrator.

    Args:
        llm: Language model instance (optional)
        model_name: Name of the Gemini model to use (e.g., "gemini-2.0-flash", "gemini-2.5-pro")
        max_rounds: Maximum debate rounds
        consensus_threshold: Required consensus level
        on_message_callback: Callback for new messages

    Returns:
        Configured DebateOrchestrator instance
    """
    return DebateOrchestrator(
        llm=llm,
        model_name=model_name,
        max_rounds=max_rounds,
        consensus_threshold=consensus_threshold,
        on_message_callback=on_message_callback
    )
