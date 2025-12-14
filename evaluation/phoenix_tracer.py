"""
Phoenix tracing integration for LLM observability.

This module provides:
- OpenTelemetry-based tracing setup for Phoenix
- Decorators for tracing agent calls
- Debate-level tracing utilities
"""
import os
import time
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime

from utils.config import get_phoenix_config


# Global tracer instance
_tracer = None
_phoenix_enabled = False


def setup_phoenix_tracing(
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None
) -> bool:
    """
    Set up Phoenix tracing with OpenTelemetry.

    Args:
        project_name: Name of the project for tracing
        endpoint: Phoenix server endpoint

    Returns:
        True if setup successful, False otherwise
    """
    global _tracer, _phoenix_enabled

    config = get_phoenix_config()

    if not config.get("enabled", False):
        print("Phoenix tracing is disabled. Set PHOENIX_ENABLED=true to enable.")
        return False

    project = project_name or config.get("project_name", "roundtable-ai")
    ep = endpoint or config.get("endpoint", "http://localhost:6006")

    try:
        # Import Phoenix and OpenTelemetry
        from phoenix.otel import register
        from phoenix.trace.langchain import LangChainInstrumentor

        # Register tracer provider with Phoenix
        tracer_provider = register(
            project_name=project,
            endpoint=ep
        )

        # Instrument LangChain
        LangChainInstrumentor().instrument()

        _phoenix_enabled = True
        print(f"Phoenix tracing enabled. Project: {project}, Endpoint: {ep}")

        return True

    except ImportError as e:
        print(f"Phoenix not installed. Install with: pip install arize-phoenix arize-phoenix-otel")
        print(f"Error: {e}")
        return False

    except Exception as e:
        print(f"Failed to setup Phoenix tracing: {e}")
        return False


def get_tracer():
    """
    Get the configured tracer instance.

    Returns:
        Tracer instance or None if not configured
    """
    global _tracer

    if not _phoenix_enabled:
        return None

    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer("roundtable-ai")
        except ImportError:
            return None

    return _tracer


def trace_debate(debate_id: str):
    """
    Decorator to trace an entire debate session.

    Args:
        debate_id: Unique identifier for the debate

    Usage:
        @trace_debate("debate-123")
        def run_my_debate():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(f"debate:{debate_id}") as span:
                span.set_attribute("debate.id", debate_id)
                span.set_attribute("debate.start_time", datetime.now().isoformat())

                try:
                    result = func(*args, **kwargs)

                    # Add result attributes
                    if hasattr(result, 'recommendation'):
                        span.set_attribute("debate.recommendation", str(result.recommendation))
                    if hasattr(result, 'consensus_level'):
                        span.set_attribute("debate.consensus", result.consensus_level)

                    span.set_attribute("debate.status", "success")
                    return result

                except Exception as e:
                    span.set_attribute("debate.status", "error")
                    span.set_attribute("debate.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def trace_agent_call(agent_type: str, operation: str = "chat"):
    """
    Decorator to trace individual agent calls.

    Args:
        agent_type: Type of agent (fundamental, sentiment, valuation)
        operation: Type of operation (chat, analyze, etc.)

    Usage:
        @trace_agent_call("fundamental", "analyze")
        def analyze_company(company):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            if tracer is None:
                return func(*args, **kwargs)

            span_name = f"agent:{agent_type}:{operation}"

            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("agent.type", agent_type)
                span.set_attribute("agent.operation", operation)
                span.set_attribute("agent.start_time", datetime.now().isoformat())

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    # Record latency
                    latency = time.time() - start_time
                    span.set_attribute("agent.latency_ms", latency * 1000)

                    # Try to extract response info
                    if isinstance(result, str):
                        span.set_attribute("agent.response_length", len(result))
                    elif isinstance(result, dict):
                        if "recommendation" in result:
                            span.set_attribute("agent.recommendation", result["recommendation"])
                        if "confidence" in result:
                            span.set_attribute("agent.confidence", result["confidence"])

                    span.set_attribute("agent.status", "success")
                    return result

                except Exception as e:
                    span.set_attribute("agent.status", "error")
                    span.set_attribute("agent.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


def trace_tool_call(tool_name: str):
    """
    Decorator to trace tool invocations.

    Args:
        tool_name: Name of the tool being called

    Usage:
        @trace_tool_call("finance_report_pull")
        def get_financial_data(ticker):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            if tracer is None:
                return func(*args, **kwargs)

            with tracer.start_as_current_span(f"tool:{tool_name}") as span:
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("tool.start_time", datetime.now().isoformat())

                start_time = time.time()

                try:
                    result = func(*args, **kwargs)

                    latency = time.time() - start_time
                    span.set_attribute("tool.latency_ms", latency * 1000)
                    span.set_attribute("tool.status", "success")

                    # Record result size
                    if isinstance(result, dict):
                        span.set_attribute("tool.result_keys", list(result.keys()))
                        if "success" in result:
                            span.set_attribute("tool.result_success", result["success"])

                    return result

                except Exception as e:
                    span.set_attribute("tool.status", "error")
                    span.set_attribute("tool.error", str(e))
                    span.record_exception(e)
                    raise

        return wrapper
    return decorator


class DebateTracer:
    """
    Context manager for tracing entire debate sessions with detailed metrics.
    """

    def __init__(self, company: str, ticker: str):
        """
        Initialize debate tracer.

        Args:
            company: Company name being analyzed
            ticker: Stock ticker symbol
        """
        self.company = company
        self.ticker = ticker
        self.debate_id = f"{ticker}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.span = None
        self.start_time = None
        self.round_count = 0
        self.message_count = 0
        self.tool_calls = 0

    def __enter__(self):
        """Enter the debate tracing context."""
        tracer = get_tracer()

        if tracer:
            self.span = tracer.start_span(f"debate:{self.debate_id}")
            self.span.set_attribute("debate.company", self.company)
            self.span.set_attribute("debate.ticker", self.ticker)
            self.span.set_attribute("debate.id", self.debate_id)

        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the debate tracing context."""
        if self.span:
            duration = time.time() - self.start_time

            self.span.set_attribute("debate.duration_seconds", duration)
            self.span.set_attribute("debate.round_count", self.round_count)
            self.span.set_attribute("debate.message_count", self.message_count)
            self.span.set_attribute("debate.tool_calls", self.tool_calls)

            if exc_type:
                self.span.set_attribute("debate.status", "error")
                self.span.record_exception(exc_val)
            else:
                self.span.set_attribute("debate.status", "success")

            self.span.end()

        return False

    def record_round(self, round_number: int, consensus: float):
        """Record a completed debate round."""
        self.round_count = round_number

        if self.span:
            self.span.set_attribute(f"debate.round_{round_number}_consensus", consensus)

    def record_message(self, agent_type: str, recommendation: str, confidence: float):
        """Record a debate message."""
        self.message_count += 1

        if self.span:
            self.span.set_attribute(
                f"debate.message_{self.message_count}",
                f"{agent_type}:{recommendation}:{confidence:.2f}"
            )

    def record_tool_call(self, tool_name: str, success: bool):
        """Record a tool invocation."""
        self.tool_calls += 1

        if self.span:
            self.span.add_event(
                "tool_call",
                {"tool.name": tool_name, "tool.success": success}
            )

    def record_final_recommendation(self, recommendation: str, confidence: float, consensus: float):
        """Record the final debate recommendation."""
        if self.span:
            self.span.set_attribute("debate.final_recommendation", recommendation)
            self.span.set_attribute("debate.final_confidence", confidence)
            self.span.set_attribute("debate.final_consensus", consensus)
