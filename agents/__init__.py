"""
Agent modules for RoundtableAI multi-agent system.

This package contains specialized analysis agents:
- FundamentalAgent: Financial statement analysis and company fundamentals
- SentimentAgent: News sentiment analysis using FinBERT scores
- ValuationAgent: Risk-return metrics and valuation analysis
- DebateOrchestrator: Multi-agent debate coordination

Each agent uses Llama 3 8B via HuggingFace Transformers with LangChain's
create_agent for tool orchestration and conversation management.
"""
from .base import (
    BaseAgentMixin,
    get_llm,
    get_memory_saver,
    create_agent_config
)
from .fundamental_agent import (
    FundamentalAgent,
    create_fundamental_agent,
    FUNDAMENTAL_AGENT_SYSTEM_PROMPT
)
from .sentiment_agent import (
    SentimentAgent,
    create_sentiment_agent,
    SENTIMENT_AGENT_SYSTEM_PROMPT
)
from .valuation_agent import (
    ValuationAgent,
    create_valuation_agent,
    VALUATION_AGENT_SYSTEM_PROMPT
)
from .debate_state import (
    DebateMessage,
    DebateState,
    AgentVote,
    RoundResult,
    Recommendation,
    FinalRecommendation
)
from .debate_manager import DebateManager
from .orchestrator import (
    DebateOrchestrator,
    create_debate_orchestrator
)

__all__ = [
    # Base utilities
    'BaseAgentMixin',
    'get_llm',
    'get_memory_saver',
    'create_agent_config',
    # Fundamental Agent
    'FundamentalAgent',
    'create_fundamental_agent',
    'FUNDAMENTAL_AGENT_SYSTEM_PROMPT',
    # Sentiment Agent
    'SentimentAgent',
    'create_sentiment_agent',
    'SENTIMENT_AGENT_SYSTEM_PROMPT',
    # Valuation Agent
    'ValuationAgent',
    'create_valuation_agent',
    'VALUATION_AGENT_SYSTEM_PROMPT',
    # Debate State
    'DebateMessage',
    'DebateState',
    'AgentVote',
    'RoundResult',
    'Recommendation',
    'FinalRecommendation',
    # Debate Manager
    'DebateManager',
    # Orchestrator
    'DebateOrchestrator',
    'create_debate_orchestrator',
]
