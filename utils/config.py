"""
Configuration module for loading environment variables and application settings.

This module provides centralized configuration for:
- Database connections
- API keys
- Debate orchestration settings
- Streamlit UI settings
- Phoenix evaluation settings
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# =============================================================================
# API Keys and Credentials
# =============================================================================

def get_gemini_api_key() -> str:
    """Get Google Gemini API key from environment variables."""
    return os.getenv("GEMINI_API_KEY")


def get_mongo_uri() -> str:
    """Get MongoDB connection URI from environment variables."""
    username = os.getenv("MONGO_USERNAME", "Wrynaft")
    password = os.getenv("MONGO_PASSWORD", "Ryan@120104")
    return f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0"


def get_langsmith_api_key() -> str:
    """Get LangSmith API key from environment variables."""
    return os.getenv("LANGCHAIN_API_KEY")


def get_langsmith_tracing() -> bool:
    """Check if LangSmith tracing is enabled."""
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"


def get_hf_token() -> str:
    """Get HuggingFace token from environment variables."""
    return os.getenv("HF_TOKEN")


# =============================================================================
# Debate Configuration
# =============================================================================

DEBATE_CONFIG = {
    # Maximum number of debate rounds
    "max_rounds": 5,

    # Minimum turns each agent must take before consensus check
    "min_turns_per_agent": 2,

    # Required consensus percentage to end debate early (0.0-1.0)
    "consensus_threshold": 0.75,

    # Order of agents for round-robin turns
    "agent_order": ["fundamental", "sentiment", "valuation"],

    # Ollama model for synthesis
    "synthesis_model": "llama3.1:8b",

    # Default confidence when not parsed from response
    "default_confidence": 0.5,

    # Weights for different agent types in final recommendation
    "agent_weights": {
        "fundamental": 0.4,
        "sentiment": 0.25,
        "valuation": 0.35
    }
}


# =============================================================================
# LLM Configuration (Ollama)
# =============================================================================

LLM_CONFIG = {
    # Ollama model name (run: ollama pull llama3.1:8b)
    "model_name": "llama3.1:8b",

    # Ollama server URL
    "base_url": "http://localhost:11434",

    # Context window size
    "num_ctx": 8192,  # Increased from 4096 for system prompts + tool results

    # Sampling temperature
    "temperature": 0.7,
}


# =============================================================================
# Streamlit Configuration
# =============================================================================

STREAMLIT_CONFIG = {
    # Page settings
    "page_title": "RoundtableAI - Multi-Agent Stock Analysis",
    "page_icon": "📊",
    "layout": "wide",

    # Theme
    "theme": "dark",

    # Update intervals (seconds)
    "message_refresh_interval": 1.0,
    "status_refresh_interval": 0.5,

    # UI limits
    "max_displayed_messages": 50,
    "transcript_max_length": 10000,
}


# =============================================================================
# Phoenix Evaluation Configuration
# =============================================================================

PHOENIX_CONFIG = {
    # Project name for tracing
    "project_name": "roundtable-ai",

    # Phoenix server endpoint
    "endpoint": os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"),

    # Enable/disable tracing
    "enabled": os.getenv("PHOENIX_ENABLED", "false").lower() == "true",

    # Metrics to track
    "track_metrics": [
        "response_latency",
        "token_usage",
        "tool_invocations",
        "recommendation_confidence",
        "consensus_convergence"
    ]
}


# =============================================================================
# Database Configuration
# =============================================================================

DATABASE_CONFIG = {
    # MongoDB database name
    "database_name": "roundtable_ai",

    # Collection names
    "collections": {
        "fundamentals": "fundamentals",
        "stock_prices": "stock_prices",
        "articles": "articles"
    },

    # Query settings
    "default_lookback_days": 7,
    "max_articles_per_query": 100,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_debate_config() -> dict:
    """Get the debate configuration dictionary."""
    return DEBATE_CONFIG.copy()


def get_llm_config() -> dict:
    """Get the LLM configuration dictionary."""
    return LLM_CONFIG.copy()


def get_streamlit_config() -> dict:
    """Get the Streamlit configuration dictionary."""
    return STREAMLIT_CONFIG.copy()


def get_phoenix_config() -> dict:
    """Get the Phoenix evaluation configuration dictionary."""
    return PHOENIX_CONFIG.copy()


def get_database_config() -> dict:
    """Get the database configuration dictionary."""
    return DATABASE_CONFIG.copy()
