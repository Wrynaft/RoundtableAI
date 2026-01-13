"""
Base agent module with shared LLM setup and common utilities.

This module provides:
- Google Gemini API model initialization
- Common agent configuration
- Shared utilities for multi-agent orchestration
"""
import os
from typing import Optional, List, Any, Generator, Dict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # pip install langchain-google-genai
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables
load_dotenv()

# Available models for selection
AVAILABLE_MODELS = {
    "gemini-2.0-flash": {
        "name": "Gemini 2.0 Flash",
        "description": "Fast and efficient, good for quick analysis",
        "model_id": "gemini-2.0-flash"
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "description": "Most capable, best for complex reasoning",
        "model_id": "gemini-2.5-pro"
    }
}

# Cache for model instances (keyed by model_name)
_llm_cache: Dict[str, ChatGoogleGenerativeAI] = {}


def get_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.3,
    force_reload: bool = False
) -> ChatGoogleGenerativeAI:
    """
    Get or create a Google Gemini LLM instance (cached by model name).

    Uses Gemini via Google AI API for inference.

    Prerequisites:
        1. Get API key from: https://aistudio.google.com/apikey
        2. Set GOOGLE_API_KEY in .env file

    Args:
        model_name: Gemini model name (default: gemini-2.0-flash)
                   Options: gemini-2.0-flash, gemini-2.5-pro
        temperature: Sampling temperature (default: 0.3)
        force_reload: Force reload the model even if already cached (default: False)

    Returns:
        ChatGoogleGenerativeAI instance configured for text generation
    """
    global _llm_cache

    # Get the actual model ID (handles aliases)
    model_config = AVAILABLE_MODELS.get(model_name, {})
    actual_model_id = model_config.get("model_id", model_name)

    cache_key = f"{actual_model_id}_{temperature}"
    if cache_key in _llm_cache and not force_reload:
        return _llm_cache[cache_key]

    # Get API key from environment
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables.\n"
            "Please set it in your .env file:\n"
            "GOOGLE_API_KEY=your_api_key_here\n\n"
            "Get your API key from: https://aistudio.google.com/apikey"
        )

    print(f"Initializing Gemini model: {actual_model_id}")

    # Create Gemini LLM instance
    llm_instance = ChatGoogleGenerativeAI(
        model=actual_model_id,
        temperature=temperature,
        google_api_key=api_key,
    )

    # Test connection
    try:
        llm_instance.invoke("Hello")
        print(f"Successfully connected to Gemini model: {actual_model_id}")
    except Exception as e:
        print(f"Warning: Could not connect to Gemini API")
        print(f"Error: {e}")
        print("\nMake sure your API key is valid:")
        print("  1. Get API key: https://aistudio.google.com/apikey")
        print("  2. Add to .env: GOOGLE_API_KEY=your_key_here")

    # Cache the instance
    _llm_cache[cache_key] = llm_instance

    return llm_instance


def get_available_models() -> Dict[str, dict]:
    """
    Get dictionary of available models for UI selection.

    Returns:
        Dictionary mapping model keys to their configuration
    """
    return AVAILABLE_MODELS


def clear_llm_cache():
    """Clear the LLM cache to force reloading models."""
    global _llm_cache
    _llm_cache = {}


def get_memory_saver() -> InMemorySaver:
    """
    Create a new InMemorySaver instance for conversation history.

    Each agent should have its own InMemorySaver to maintain
    separate conversation histories.

    Returns:
        InMemorySaver instance
    """
    return InMemorySaver()


def create_agent_config(thread_id: str = "default") -> dict:
    """
    Create standard agent configuration dictionary.

    Args:
        thread_id: Unique identifier for the conversation thread

    Returns:
        Configuration dictionary for agent invocation
    """
    return {"configurable": {"thread_id": thread_id}}


class BaseAgentMixin:
    """
    Mixin class providing common functionality for all agents.

    This mixin prepares agents for multi-agent orchestration
    by providing:
    - Standard interface for agent invocation
    - Message formatting utilities
    - Metadata for orchestration
    - Debate support methods
    """

    agent_type: str = "base"
    agent_description: str = "Base agent"

    def get_agent_metadata(self) -> dict:
        """
        Get metadata about this agent for orchestration.

        Returns:
            Dictionary containing agent metadata
        """
        return {
            "agent_type": self.agent_type,
            "description": self.agent_description,
            "capabilities": self.get_capabilities()
        }

    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities this agent provides.
        Override in subclasses.

        Returns:
            List of capability descriptions
        """
        return []

    def format_response_for_debate(self, response: str, confidence: float = 0.5) -> dict:
        """
        Format agent response for multi-agent debate.

        Args:
            response: The agent's response text
            confidence: Confidence level (0.0 to 1.0)

        Returns:
            Formatted response dictionary for orchestration
        """
        return {
            "agent_type": self.agent_type,
            "response": response,
            "confidence": confidence,
            "metadata": self.get_agent_metadata()
        }

    def generate_debate_response(
        self,
        prompt: str,
        thread_id: str = "debate"
    ) -> dict:
        """
        Generate a response for the debate context.

        Args:
            prompt: The debate prompt to respond to
            thread_id: Thread ID for conversation memory

        Returns:
            Dictionary with response, confidence, and recommendation
        """
        # This should be overridden in subclasses that have a chat method
        if hasattr(self, 'chat'):
            response = self.chat(prompt, thread_id=thread_id)
            return {
                "agent_type": self.agent_type,
                "response": response,
                "confidence": 0.5,  # Default confidence
                "recommendation": "HOLD"  # Default recommendation
            }
        raise NotImplementedError("Subclass must implement chat method")

    def respond_to_critique(
        self,
        critique: str,
        original_analysis: str,
        thread_id: str = "debate"
    ) -> str:
        """
        Respond to another agent's critique of this agent's analysis.

        Args:
            critique: The critique from another agent
            original_analysis: This agent's original analysis being critiqued
            thread_id: Thread ID for conversation memory

        Returns:
            Response to the critique
        """
        prompt = f"""Another analyst has critiqued your analysis:

Your original analysis:
{original_analysis[:1000]}...

Their critique:
{critique}

Respond to their critique from your {self.agent_type} analysis perspective:
1. Address their specific points
2. Defend or revise your position as appropriate
3. Provide any additional evidence
4. State your updated recommendation (BUY/HOLD/SELL) and confidence level"""

        if hasattr(self, 'chat'):
            return self.chat(prompt, thread_id=thread_id)
        raise NotImplementedError("Subclass must implement chat method")

    def get_analysis_summary(self, thread_id: str = "default") -> str:
        """
        Get a brief summary of the agent's analysis for the current thread.

        Args:
            thread_id: Thread ID to get summary for

        Returns:
            Brief summary of the analysis
        """
        prompt = """Provide a brief 2-3 sentence summary of your analysis so far, including:
1. Your main conclusion
2. Your recommendation (BUY/HOLD/SELL)
3. Your confidence level"""

        if hasattr(self, 'chat'):
            return self.chat(prompt, thread_id=thread_id)
        return "No analysis available"

    def get_recommendation_strength(self) -> str:
        """
        Get a description of how strongly this agent type typically influences recommendations.

        Returns:
            Description of recommendation strength
        """
        strength_descriptions = {
            "fundamental": "Strong influence - fundamentals are core to intrinsic value",
            "sentiment": "Moderate influence - sentiment affects short-term movements",
            "valuation": "Strong influence - risk-return metrics guide position sizing",
            "base": "Unknown influence"
        }
        return strength_descriptions.get(self.agent_type, strength_descriptions["base"])

