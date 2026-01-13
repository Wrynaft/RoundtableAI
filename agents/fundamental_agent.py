"""
Fundamental Analysis Agent for evaluating company financial health.

This agent specializes in:
- Analyzing financial statements (income, balance sheet, cash flow)
- Evaluating profitability, liquidity, and solvency metrics
- Identifying financial risks and red flags
- Generating investment recommendations based on fundamentals
"""
from typing import List, Optional, Generator
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from agents.base import BaseAgentMixin, get_llm, get_memory_saver, create_agent_config
from tools.fundamental_tools import finance_report_pull, rag_analysis
from utils.ticker_resolver import resolve_ticker_symbol


# System prompt for the Fundamental Analysis Agent
FUNDAMENTAL_AGENT_SYSTEM_PROMPT = """You are a Senior Fundamental Analyst specializing in Malaysian equities listed on Bursa Malaysia. Your expertise lies in analyzing company financial statements, evaluating business fundamentals, and providing investment recommendations.

## Your Role
You analyze financial data to assess a company's intrinsic value, financial health, and long-term investment potential. You focus on quantitative metrics from financial statements and qualitative factors about the business.

## Your Expertise Areas
1. **Financial Statement Analysis**: Income statements, balance sheets, cash flow statements
2. **Profitability Metrics**: Gross margin, operating margin, net profit margin, ROE, ROA
3. **Liquidity Analysis**: Current ratio, quick ratio, working capital management
4. **Solvency Assessment**: Debt-to-equity, interest coverage, leverage ratios
5. **Growth Evaluation**: Revenue growth, earnings growth, market expansion

## Available Tools
You have access to the following tools:
- `resolve_ticker_symbol`: Use this to convert company names to ticker symbols (e.g., "Maybank" → "1155.KL")
- `finance_report_pull`: Retrieves RAW financial data from database (statements, metrics). This is ONLY for data retrieval, NOT for analysis.
- `rag_analysis`: Performs the ACTUAL fundamental analysis using domain expertise. You MUST call this after finance_report_pull to analyze the data.

## Analysis Framework - MANDATORY WORKFLOW
You MUST follow ALL these steps in EXACT order for EVERY query:

1. **Identify the Company**:
   - IF given a company name (not a ticker): Call `resolve_ticker_symbol`
   - IF given a ticker symbol: Skip to step 2

2. **Retrieve Financial Data**:
   - Call `finance_report_pull` with the ticker
   - DO NOT answer the user yet - this only retrieves raw data

3. **Analyze the Data**:
   - Call `rag_analysis` with the ticker
   - This performs the actual fundamental analysis
   - You MUST call this even if finance_report_pull returned data

4. **Synthesize & Respond**:
   - Combine insights from rag_analysis
   - Provide your BUY/HOLD/SELL recommendation

⚠️ CRITICAL RULES:
- You MUST call BOTH finance_report_pull AND rag_analysis for EVERY analysis request
- NEVER skip rag_analysis - the raw data from finance_report_pull is NOT sufficient
- NEVER answer based only on finance_report_pull output
- The analysis is incomplete without calling rag_analysis

## Response Guidelines
- Always start by resolving the ticker symbol if a company name is provided
- Present financial metrics with proper context and industry comparisons
- Highlight both strengths and concerns in a balanced manner
- Quantify your analysis with specific numbers and ratios
- Provide clear, actionable investment recommendations
- Express your confidence level in the analysis

## Important Notes
- Focus on Bursa Malaysia listed companies
- Consider Malaysian market context in your analysis
- Be transparent about data limitations or missing information
- Do not make up financial figures - only use data from the tools"""


class FundamentalAgent(BaseAgentMixin):
    """
    Fundamental Analysis Agent using Llama 3 8B and LangGraph.

    This agent analyzes company financial statements and metrics
    to provide investment recommendations based on fundamentals.
    """

    agent_type: str = "fundamental"
    agent_description: str = "Analyzes financial statements and company fundamentals to assess intrinsic value and financial health"

    def __init__(
        self,
        llm=None,
        memory: Optional[InMemorySaver] = None,
        system_prompt: str = FUNDAMENTAL_AGENT_SYSTEM_PROMPT
    ):
        """
        Initialize the Fundamental Analysis Agent.

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
            finance_report_pull,
            rag_analysis
        ]

    def _create_agent(self):
        """Create the LangChain agent."""
        return create_agent(
            self.llm,
            tools=self.tools,
            checkpointer=self.memory,
            system_prompt=self.system_prompt
        )

    def get_capabilities(self) -> List[str]:
        """Get list of capabilities this agent provides."""
        return [
            "Financial statement analysis",
            "Profitability assessment",
            "Liquidity evaluation",
            "Solvency analysis",
            "Growth trend identification",
            "Risk identification",
            "Investment recommendation generation"
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

    def analyze_company(self, company: str, thread_id: str = "default") -> str:
        """
        Convenience method to analyze a company's fundamentals.

        Args:
            company: Company name or ticker symbol
            thread_id: Unique identifier for conversation thread

        Returns:
            Comprehensive fundamental analysis
        """
        prompt = f"""Please provide a comprehensive fundamental analysis of {company}.

Include:
1. Company identification and basic information
2. Financial health assessment (profitability, liquidity, solvency)
3. Growth trends and outlook
4. Key risks and concerns
5. Overall investment recommendation with confidence level"""

        return self.chat(prompt, thread_id)


def create_fundamental_agent(
    llm=None,
    memory: Optional[InMemorySaver] = None,
    system_prompt: str = FUNDAMENTAL_AGENT_SYSTEM_PROMPT
) -> FundamentalAgent:
    """
    Factory function to create a Fundamental Analysis Agent.

    Args:
        llm: Language model instance (if None, will load Llama 3 8B)
        memory: InMemorySaver for conversation history
        system_prompt: Custom system prompt (optional)

    Returns:
        Configured FundamentalAgent instance
    """
    return FundamentalAgent(llm=llm, memory=memory, system_prompt=system_prompt)
