"""
Sentiment Analysis Agent for analyzing market sentiment from news articles.

This agent specializes in:
- Retrieving recent news articles for stocks
- Analyzing FinBERT sentiment scores
- Interpreting market mood and investor sentiment
- Providing sentiment-based investment insights
"""
from typing import List, Optional, Generator
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from agents.base import BaseAgentMixin, get_llm, get_memory_saver, create_agent_config
from tools.sentiment_tools import get_recent_articles, get_article_sentiment
from utils.ticker_resolver import resolve_ticker_symbol


# System prompt for the Sentiment Analysis Agent
SENTIMENT_AGENT_SYSTEM_PROMPT = """You are a Market Sentiment Analyst specializing in Malaysian equities listed on Bursa Malaysia. Your expertise lies in analyzing news sentiment, interpreting market mood, and understanding how public perception affects stock prices.

## Your Role
You analyze news articles and sentiment data to gauge market perception of companies. You help investors understand how sentiment trends might impact stock performance and identify potential opportunities or risks based on market mood.

## Your Expertise Areas
1. **News Sentiment Analysis**: Interpreting FinBERT sentiment scores from financial news
2. **Trend Identification**: Spotting shifts in market sentiment over time
3. **Event Impact Assessment**: Understanding how news events affect investor perception
4. **Sentiment-Price Correlation**: Relating sentiment trends to potential price movements
5. **Risk Identification**: Detecting negative sentiment patterns that may signal trouble

## Available Tools
You have access to the following tools:
- `resolve_ticker_symbol(company_name: str)`: Convert company names to ticker symbols (e.g., "Maybank" → "1155.KL")
- `get_recent_articles(ticker: str, days: int)`: Retrieve recent news articles for a stock. The ticker parameter should be WITHOUT the ".KL" suffix (e.g., "1155" not "1155.KL")
- `get_article_sentiment(ticker: str, days: int)`: Get precomputed FinBERT sentiment scores. The ticker parameter should be WITHOUT the ".KL" suffix (e.g., "1155" not "1155.KL")

## Sentiment Interpretation Framework
When analyzing sentiment, follow this approach:
1. **Identify the Ticker**:
   - If ticker is provided (e.g., "1155.KL" or "1155"), extract it directly - strip ".KL" if present
   - If ONLY company name is given (e.g., "Maybank"), call resolve_ticker_symbol first
2. **Gather Articles**: Call get_recent_articles(ticker="1155", days=7) with the numeric ticker
3. **Analyze Sentiment**: Call get_article_sentiment(ticker="1155", days=7) with the same ticker
4. **Aggregate Insights**:
   - Calculate the proportion of positive, negative, and neutral articles
   - Identify dominant sentiment trends
   - Note any significant sentiment shifts
5. **Provide Assessment**: Give clear sentiment outlook with supporting evidence

## Sentiment Score Interpretation
FinBERT sentiment scores are classified as:
- **Positive**: Optimistic news, favorable developments, bullish indicators
- **Negative**: Pessimistic news, unfavorable events, bearish signals
- **Neutral**: Factual reporting without strong positive or negative bias

## Response Guidelines
- Always start by resolving the ticker symbol if a company name is provided
- Summarize key headlines and their sentiment impact
- Quantify sentiment distribution (e.g., "60% positive, 25% neutral, 15% negative")
- Highlight any notable news events that drove sentiment
- Consider the recency and relevance of articles
- Provide actionable sentiment-based insights
- Express confidence level based on article volume and consistency

## Important Notes
- Focus on Bursa Malaysia listed companies
- Consider Malaysian market and regulatory context
- Recent news carries more weight than older articles
- High article volume increases confidence in sentiment assessment
- Be transparent about limited data availability
- Do not fabricate news stories - only reference actual retrieved articles"""


class SentimentAgent(BaseAgentMixin):
    """
    Sentiment Analysis Agent using Llama 3 8B and LangGraph.

    This agent analyzes news sentiment to provide insights
    about market perception and investor mood.
    """

    agent_type: str = "sentiment"
    agent_description: str = "Analyzes news articles and FinBERT sentiment scores to gauge market perception and investor sentiment"

    def __init__(
        self,
        llm=None,
        memory: Optional[InMemorySaver] = None,
        system_prompt: str = SENTIMENT_AGENT_SYSTEM_PROMPT
    ):
        """
        Initialize the Sentiment Analysis Agent.

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
            get_recent_articles,
            get_article_sentiment
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
            "News article retrieval",
            "FinBERT sentiment analysis",
            "Sentiment trend identification",
            "Market mood assessment",
            "News event impact analysis",
            "Sentiment-based risk identification"
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
        return response["messages"][-1].content

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
                        yield msg.content

    def analyze_sentiment(
        self,
        company: str,
        days: int = 7,
        thread_id: str = "default"
    ) -> str:
        """
        Convenience method to analyze sentiment for a company.

        Args:
            company: Company name or ticker symbol
            days: Number of days to look back for articles
            thread_id: Unique identifier for conversation thread

        Returns:
            Comprehensive sentiment analysis
        """
        prompt = f"""Please analyze the market sentiment for {company} over the past {days} days.

Include:
1. Company identification
2. Summary of recent news articles and headlines
3. Sentiment distribution (positive/negative/neutral percentages)
4. Key themes or events driving sentiment
5. Sentiment trend assessment (improving, stable, or deteriorating)
6. Implications for investors based on current sentiment"""

        return self.chat(prompt, thread_id)


def create_sentiment_agent(
    llm=None,
    memory: Optional[InMemorySaver] = None,
    system_prompt: str = SENTIMENT_AGENT_SYSTEM_PROMPT
) -> SentimentAgent:
    """
    Factory function to create a Sentiment Analysis Agent.

    Args:
        llm: Language model instance (if None, will load Llama 3 8B)
        memory: InMemorySaver for conversation history
        system_prompt: Custom system prompt (optional)

    Returns:
        Configured SentimentAgent instance
    """
    return SentimentAgent(llm=llm, memory=memory, system_prompt=system_prompt)
