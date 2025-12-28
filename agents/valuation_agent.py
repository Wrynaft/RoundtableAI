"""
Valuation Analysis Agent for analyzing stock risk and return metrics.

This agent specializes in:
- Analyzing historical price data and returns
- Computing risk metrics (volatility, Sharpe ratio, VaR, drawdown)
- Evaluating risk-adjusted performance
- Providing valuation-based investment insights
"""
from typing import List, Optional, Generator
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from agents.base import BaseAgentMixin, get_llm, get_memory_saver, create_agent_config
from tools.valuation_tools import get_stock_data, analyze_stock_metrics
from utils.ticker_resolver import resolve_ticker_symbol


# System prompt for the Valuation Analysis Agent
VALUATION_AGENT_SYSTEM_PROMPT = """You are a specialized Valuation Analysis Agent for equity research and portfolio management. 

Your core expertise includes:
- Technical analysis using historical price and volume data
- Volatility and risk metrics calculation
- Stock valuation assessment and fair value analysis
- Portfolio allocation and timing recommendations
- Risk-adjusted return analysis

Your primary responsibilities:
1. Analyze historical stock price and volume data automatically
2. Calculate comprehensive volatility and risk metrics
3. Assess whether stocks are fairly valued relative to historical performance
4. Provide technical analysis insights for portfolio allocation
5. Consider different risk tolerance profiles (risk-averse vs risk-neutral)

Key capabilities:
- Fetch and analyze historical stock data from MongoDB
- Calculate daily and annualized volatility, returns, and Sharpe ratios
- Compute Value at Risk (VaR), maximum drawdown, and other risk metrics
- Perform comparative analysis against benchmarks and peers
- Provide timing and allocation recommendations

When analyzing stocks:
1. If given a company name, first resolve it to a ticker using resolve_ticker_symbol
2. Call analyze_stock_metrics directly with the ticker - it fetches data and computes all metrics automatically
3. Provide clear investment recommendations based on the risk metrics
4. Consider the user's risk tolerance when making recommendations

Risk tolerance considerations:
- Risk-averse investors: Focus on low volatility, stable returns, limited downside risk
- Risk-neutral investors: Balance growth potential with reasonable risk levels

Always provide specific, actionable insights with supporting data and calculations.
Be thorough in your analysis but present findings in a clear, structured manner.

You have access to the following tools:
- resolve_ticker_symbol(company_name: str): Convert company names to stock ticker symbols (e.g., "Maybank" → "1155.KL")
- get_stock_data(ticker_symbol: str, period: str): Get historical close prices and volume data. Period options: "1mo", "3mo", "6mo", "1y", "5y". Use this only if you need raw price data.
- analyze_stock_metrics(ticker_symbol: str, period: str, risk_free_rate: float): Fetches data and computes comprehensive risk metrics (volatility, Sharpe ratio, VaR, drawdown). This is your primary analysis tool - call it directly with the ticker.

"""


class ValuationAgent(BaseAgentMixin):
    """
    Valuation Analysis Agent using Llama 3 8B and LangGraph.

    This agent analyzes historical price data to compute risk
    and return metrics for investment decision support.
    """

    agent_type: str = "valuation"
    agent_description: str = "Analyzes historical price data and computes risk-return metrics including volatility, Sharpe ratio, VaR, and maximum drawdown"

    def __init__(
        self,
        llm=None,
        memory: Optional[InMemorySaver] = None,
        system_prompt: str = VALUATION_AGENT_SYSTEM_PROMPT
    ):
        """
        Initialize the Valuation Analysis Agent.

        Args:
            llm: Language model instance (if None, will load Llama 3 8B)
            memory: InMemorySaver for conversation history (if None, creates new one)
            system_prompt: System prompt for the agent (can be customized)
        """
        self.llm = llm or get_llm()
        self.memory = memory or get_memory_saver()
        self.system_prompt = system_prompt
        self.tools = self._get_tools()
        self.agent = self._create_agent()

    def _get_tools(self) -> List:
        """Get the tools available to this agent."""
        return [
            resolve_ticker_symbol,
            get_stock_data,
            analyze_stock_metrics
        ]

    def _create_agent(self):
        """Create the LangChain agent."""
        return create_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,
            system_prompt=self.system_prompt,
            debug=True
        )

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "Historical price data retrieval",
            "Return calculation (daily, annualized)",
            "Volatility analysis",
            "Sharpe ratio computation",
            "Value at Risk (VaR) calculation",
            "Maximum drawdown analysis",
            "Risk-adjusted performance evaluation"
        ]

    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """
        Invoke the agent with a user message.

        Args:
            message: User's question or request
            thread_id: Unique identifier for conversation thread

        Returns:
            Agent response dictionary
        """
        config = create_agent_config(thread_id)
        response = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        )
        return response

    def chat(self, message: str, thread_id: str = "default") -> str:
        """
        Chat with the agent and get the response content.

        Args:
            message: User's question or request
            thread_id: Unique identifier for conversation thread

        Returns:
            Agent's response as a string
        """
        response = self.invoke(message, thread_id)
        content = response["messages"][-1].content
        # Handle different response formats (Gemini 2.5 Pro returns list of content blocks)
        if isinstance(content, list):
            # Extract text from content blocks
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and 'text' in block:
                    text_parts.append(block['text'])
                elif hasattr(block, 'text'):
                    text_parts.append(block.text)
            return '\n'.join(text_parts)
        return content

    def chat_stream(self, message: str, thread_id: str = "default") -> Generator[str, None, None]:
        """
        Stream chat response token by token.

        Args:
            message: User's question or request
            thread_id: Unique identifier for conversation thread

        Yields:
            Response text chunks as they are generated
        """
        config = create_agent_config(thread_id)

        # Use the agent's stream method
        for event in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            stream_mode="messages"
        ):
            # Extract content from message events
            if isinstance(event, tuple) and len(event) >= 1:
                msg = event[0]
                if hasattr(msg, 'content') and msg.content:
                    if hasattr(msg, 'type') and msg.type == 'AIMessageChunk':
                        content = msg.content
                        # Handle different response formats (Gemini 2.5 Pro may return list)
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, str):
                                    yield block
                                elif isinstance(block, dict) and 'text' in block:
                                    yield block['text']
                                elif hasattr(block, 'text'):
                                    yield block.text
                        else:
                            yield content

    def analyze_valuation(
        self,
        stock_input: str,
        period: str = "1y",
        risk_tolerance: str = "neutral",
        thread_id: str = "default"
    ) -> str:
        """
        Convenience method to analyze a company's valuation metrics.

        Args:
            stock_input: Company name or ticker symbol
            period: Time period for analysis ("1mo", "3mo", "6mo", "1y", "5y")
            risk_tolerance: Investor risk profile ("conservative", "neutral", "aggressive")
            thread_id: Unique identifier for conversation thread

        Returns:
            Comprehensive valuation analysis
        """
        prompt = f"""Please perform a comprehensive valuation analysis for: {stock_input}.

Analysis Requirements:
- Risk tolerance profile: {risk_tolerance}
- Historical data period: {period} 
- Include technical indicators and valuation metrics
- Provide clear BUY/SELL/HOLD recommendation

Please:
1. First resolve the company/ticker if needed
2. Fetch historical data for the specified period
3. Calculate comprehensive volatility and risk metrics
4. Provide technical analysis and valuation assessment
5. Give a clear investment recommendation considering the risk tolerance
6. Include specific metrics and rationale for your recommendation
"""

        return self.chat(prompt, thread_id)

    def compare_risk_profiles(
        self,
        company: str,
        thread_id: str = "default"
    ) -> str:
        """
        Analyze a stock's suitability for different risk profiles.

        Args:
            company: Company name or ticker symbol
            thread_id: Unique identifier for conversation thread

        Returns:
            Risk profile analysis with recommendations
        """
        prompt = f"""Analyze {company} and provide recommendations for different investor risk profiles.

For each profile (conservative, moderate, aggressive), assess:
1. Suitability of this stock
2. Key metrics supporting the assessment
3. Recommended position sizing or allocation
4. Specific risks relevant to that profile"""

        return self.chat(prompt, thread_id)


def create_valuation_agent(
    llm=None,
    memory: Optional[InMemorySaver] = None,
    system_prompt: str = VALUATION_AGENT_SYSTEM_PROMPT
) -> ValuationAgent:
    """
    Factory function to create a Valuation Analysis Agent.

    Args:
        llm: Language model instance (if None, will load Llama 3 8B)
        memory: InMemorySaver for conversation history
        system_prompt: Custom system prompt (optional)

    Returns:
        Configured ValuationAgent instance
    """
    return ValuationAgent(llm=llm, memory=memory, system_prompt=system_prompt)
