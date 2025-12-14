"""
Evaluation module for RoundtableAI.

This package provides:
- Phoenix tracing integration for LLM observability
- Debate quality metrics
- Backtesting framework for recommendation evaluation
"""
from .phoenix_tracer import (
    setup_phoenix_tracing,
    get_tracer,
    trace_debate,
    trace_agent_call
)
from .metrics import (
    DebateEvaluator,
    calculate_consensus_quality,
    calculate_reasoning_consistency
)

__all__ = [
    'setup_phoenix_tracing',
    'get_tracer',
    'trace_debate',
    'trace_agent_call',
    'DebateEvaluator',
    'calculate_consensus_quality',
    'calculate_reasoning_consistency',
]
