"""
Phoenix tracing integration for LLM observability.

This module provides:
- OpenTelemetry-based tracing setup for Phoenix
- Decorators for tracing agent calls
- Debate-level tracing utilities

Usage (with Docker - RECOMMENDED for Windows):
    # First, run Phoenix in Docker:
    # docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

    from evaluation.phoenix_tracer import setup_docker_phoenix

    # Connect to Docker Phoenix
    setup_docker_phoenix()

    # Now run your LangChain code - traces appear at http://localhost:6006

Usage (in notebook - may have issues on Windows):
    from evaluation.phoenix_tracer import launch_phoenix, setup_tracing

    # Launch Phoenix UI
    session = launch_phoenix()

    # Setup tracing
    setup_tracing()

    # Now run your LangChain code - traces will appear in Phoenix UI
"""
import os
import time
from typing import Optional, Callable, Any
from functools import wraps
from datetime import datetime

from utils.config import get_phoenix_config


# Global state
_tracer = None
_phoenix_enabled = False
_phoenix_session = None


def setup_docker_phoenix(
    project_name: str = "roundtable-ai",
    endpoint: str = "http://localhost:4317"
) -> bool:
    """
    Connect to Phoenix running in Docker.

    Prerequisites:
        Run Phoenix in Docker first:
        docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest

    Args:
        project_name: Name of the project for tracing
        endpoint: Phoenix collector endpoint (default: http://localhost:4317)

    Returns:
        True if setup successful, False otherwise

    Usage:
        setup_docker_phoenix()
        # Then run your LangChain code
        # View traces at http://localhost:6006
    """
    global _phoenix_enabled

    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Register tracer provider pointing to Docker Phoenix
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint
        )

        # Instrument LangChain
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

        _phoenix_enabled = True
        print(f"Connected to Phoenix at {endpoint}")
        print(f"View traces at: http://localhost:6006")
        return True

    except ImportError as e:
        print("Required packages not installed. Install with:")
        print("  pip install arize-phoenix openinference-instrumentation-langchain")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Failed to connect to Phoenix: {e}")
        print("Make sure Phoenix is running in Docker:")
        print("  docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest")
        return False


def launch_phoenix():
    """
    Launch Phoenix UI in notebook or browser.

    Returns:
        Phoenix session object with URL to access the UI

    Usage:
        session = launch_phoenix()
        print(f"Phoenix URL: {session.url}")
    """
    global _phoenix_session

    try:
        import phoenix as px

        # Launch Phoenix app
        _phoenix_session = px.launch_app()
        print(f"Phoenix launched at: {_phoenix_session.url}")
        return _phoenix_session

    except ImportError:
        print("Phoenix not installed. Install with:")
        print("  pip install arize-phoenix")
        return None
    except Exception as e:
        print(f"Failed to launch Phoenix: {e}")
        return None


def setup_tracing(project_name: Optional[str] = None):
    """
    Set up LangChain tracing after Phoenix is launched.

    Args:
        project_name: Name of the project for tracing

    Returns:
        True if setup successful, False otherwise

    Usage:
        # First launch Phoenix
        session = launch_phoenix()

        # Then setup tracing
        setup_tracing()
    """
    global _phoenix_enabled

    config = get_phoenix_config()
    project = project_name or config.get("project_name", "roundtable-ai")

    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Register tracer provider (connects to launched Phoenix instance)
        tracer_provider = register(project_name=project)

        # Instrument LangChain
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

        _phoenix_enabled = True
        print(f"Tracing enabled for project: {project}")
        return True

    except ImportError as e:
        print("Required packages not installed. Install with:")
        print("  pip install arize-phoenix openinference-instrumentation-langchain")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"Failed to setup tracing: {e}")
        return False


def setup_phoenix_tracing(
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None
) -> bool:
    """
    Set up Phoenix tracing with OpenTelemetry (legacy function).

    For best results, use launch_phoenix() + setup_tracing() instead.

    Args:
        project_name: Name of the project for tracing
        endpoint: Phoenix server endpoint (ignored, use launch_phoenix instead)

    Returns:
        True if setup successful, False otherwise
    """
    global _tracer, _phoenix_enabled

    config = get_phoenix_config()

    if not config.get("enabled", False):
        print("Phoenix tracing is disabled. Set PHOENIX_ENABLED=true to enable.")
        return False

    project = project_name or config.get("project_name", "roundtable-ai")

    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Launch Phoenix app (bypasses standalone server issues on Windows)
        session = px.launch_app()
        print(f"Phoenix launched at: {session.url}")

        # Register tracer provider
        tracer_provider = register(project_name=project)

        # Instrument LangChain with the tracer provider
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

        _phoenix_enabled = True
        print(f"Phoenix tracing enabled. Project: {project}")

        return True

    except ImportError as e:
        print(f"Phoenix not installed. Install with:")
        print(f"  pip install arize-phoenix openinference-instrumentation-langchain")
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
