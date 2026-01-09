"""
Debate Metrics Tracking

This module provides classes for tracking efficiency metrics during multi-agent debates.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import time


@dataclass
class DebateMetrics:
    """
    Tracks efficiency metrics for a single debate.

    Metrics tracked:
    - Time/Round: Average time per debate round
    - Time/Agent: Average time per agent response
    - Tokens/Sec: Token throughput during debate
    - Avg Rounds: Number of rounds completed
    - Total Time: End-to-end debate duration
    """

    # Identifiers
    company: str
    ticker: str
    model_name: str

    # Timing metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    consensus_reached_at: Optional[float] = None
    round_times: List[float] = field(default_factory=list)  # Time for each round

    # Round metrics
    rounds_completed: int = 0
    rounds_to_consensus: Optional[int] = None
    consensus_reached: bool = False

    # Token metrics
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    tokens_by_round: List[int] = field(default_factory=list)

    # API metrics
    api_calls_made: int = 0

    # Per-agent metrics (optional detailed tracking)
    agent_times: Dict[str, List[float]] = field(default_factory=dict)
    agent_tokens: Dict[str, int] = field(default_factory=dict)

    def start_round(self):
        """Mark the start of a new round."""
        self.round_times.append(time.time())
        self.tokens_by_round.append(0)

    def end_round(self):
        """Mark the end of the current round."""
        if self.round_times:
            # Calculate round duration
            round_start = self.round_times[-1]
            round_duration = time.time() - round_start
            self.round_times[-1] = round_duration
            self.rounds_completed += 1

    def record_agent_response(self, agent_type: str, duration: float, tokens: int = 0):
        """
        Record metrics for a single agent response.

        Args:
            agent_type: Type of agent (fundamental/sentiment/valuation)
            duration: Time taken for response
            tokens: Tokens used (if available)
        """
        if agent_type not in self.agent_times:
            self.agent_times[agent_type] = []
            self.agent_tokens[agent_type] = 0

        self.agent_times[agent_type].append(duration)
        self.agent_tokens[agent_type] += tokens
        self.api_calls_made += 1

    def record_tokens(self, input_tokens: int, output_tokens: int):
        """
        Record token usage.

        Args:
            input_tokens: Input tokens for this API call
            output_tokens: Output tokens for this API call
        """
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        if self.tokens_by_round:
            self.tokens_by_round[-1] += (input_tokens + output_tokens)

    def mark_consensus(self):
        """Mark when consensus was reached."""
        if not self.consensus_reached:
            self.consensus_reached = True
            self.consensus_reached_at = time.time()
            self.rounds_to_consensus = self.rounds_completed

    def finalize(self):
        """Mark debate as complete and finalize metrics."""
        self.end_time = time.time()

    @property
    def total_time(self) -> float:
        """Total debate duration in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time

    @property
    def time_to_consensus(self) -> Optional[float]:
        """Time taken to reach consensus."""
        if self.consensus_reached_at:
            return self.consensus_reached_at - self.start_time
        return None

    @property
    def time_per_round(self) -> float:
        """Average time per round."""
        if self.rounds_completed > 0:
            return self.total_time / self.rounds_completed
        return 0.0

    @property
    def time_per_agent(self) -> float:
        """Average time per agent response."""
        total_agents = sum(len(times) for times in self.agent_times.values())
        if total_agents > 0:
            return self.total_time / total_agents
        return 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def tokens_per_second(self) -> float:
        """Token throughput (tokens/second)."""
        if self.total_time > 0:
            return self.total_tokens / self.total_time
        return 0.0

    def to_dict(self) -> Dict:
        """Export metrics as dictionary."""
        return {
            # Identifiers
            "company": self.company,
            "ticker": self.ticker,
            "model_name": self.model_name,

            # Summary metrics
            "total_time": round(self.total_time, 2),
            "time_per_round": round(self.time_per_round, 2),
            "time_per_agent": round(self.time_per_agent, 2),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "rounds_completed": self.rounds_completed,
            "rounds_to_consensus": self.rounds_to_consensus,
            "time_to_consensus": round(self.time_to_consensus, 2) if self.time_to_consensus else None,

            # Token metrics
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "tokens_by_round": self.tokens_by_round,

            # API metrics
            "api_calls_made": self.api_calls_made,

            # Status
            "consensus_reached": self.consensus_reached,

            # Detailed breakdowns
            "round_times": [round(t, 2) for t in self.round_times],
            "agent_avg_times": {
                agent: round(sum(times)/len(times), 2) if times else 0
                for agent, times in self.agent_times.items()
            },
            "agent_tokens": self.agent_tokens,
        }

    def print_summary(self):
        """Print a human-readable summary of metrics."""
        print(f"\n{'='*60}")
        print(f"DEBATE METRICS: {self.company} ({self.ticker})")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        print(f"\n⏱️  TIMING METRICS")
        print(f"  Total Time:        {self.total_time:.2f}s")
        print(f"  Time/Round:        {self.time_per_round:.2f}s")
        print(f"  Time/Agent:        {self.time_per_agent:.2f}s")
        if self.time_to_consensus:
            print(f"  Time to Consensus: {self.time_to_consensus:.2f}s")

        print(f"\n🔢 TOKEN METRICS")
        print(f"  Total Tokens:      {self.total_tokens:,}")
        print(f"  Input Tokens:      {self.total_input_tokens:,}")
        print(f"  Output Tokens:     {self.total_output_tokens:,}")
        print(f"  Tokens/Second:     {self.tokens_per_second:.1f}")

        print(f"\n📊 DEBATE METRICS")
        print(f"  Rounds Completed:  {self.rounds_completed}")
        if self.rounds_to_consensus:
            print(f"  Rounds to Consensus: {self.rounds_to_consensus}")
        print(f"  API Calls:         {self.api_calls_made}")
        print(f"  Consensus Reached: {'✓' if self.consensus_reached else '✗'}")

        print(f"\n{'='*60}\n")


@dataclass
class ModelComparisonMetrics:
    """
    Aggregates metrics across multiple debates for model comparison.
    """

    model_name: str
    debates: List[DebateMetrics] = field(default_factory=list)

    def add_debate(self, metrics: DebateMetrics):
        """Add debate metrics to the collection."""
        self.debates.append(metrics)

    @property
    def num_debates(self) -> int:
        """Number of debates tracked."""
        return len(self.debates)

    @property
    def avg_total_time(self) -> float:
        """Average total debate time."""
        if not self.debates:
            return 0.0
        return sum(d.total_time for d in self.debates) / len(self.debates)

    @property
    def avg_time_per_round(self) -> float:
        """Average time per round across all debates."""
        if not self.debates:
            return 0.0
        return sum(d.time_per_round for d in self.debates) / len(self.debates)

    @property
    def avg_time_per_agent(self) -> float:
        """Average time per agent response across all debates."""
        if not self.debates:
            return 0.0
        return sum(d.time_per_agent for d in self.debates) / len(self.debates)

    @property
    def avg_tokens_per_second(self) -> float:
        """Average token throughput across all debates."""
        if not self.debates:
            return 0.0
        return sum(d.tokens_per_second for d in self.debates) / len(self.debates)

    @property
    def avg_rounds(self) -> float:
        """Average rounds completed per debate."""
        if not self.debates:
            return 0.0
        return sum(d.rounds_completed for d in self.debates) / len(self.debates)

    @property
    def avg_rounds_to_consensus(self) -> float:
        """Average rounds to reach consensus."""
        debates_with_consensus = [d for d in self.debates if d.rounds_to_consensus]
        if not debates_with_consensus:
            return 0.0
        return sum(d.rounds_to_consensus for d in debates_with_consensus) / len(debates_with_consensus)

    @property
    def consensus_rate(self) -> float:
        """Percentage of debates that reached consensus."""
        if not self.debates:
            return 0.0
        consensus_count = sum(1 for d in self.debates if d.consensus_reached)
        return consensus_count / len(self.debates)

    @property
    def avg_total_tokens(self) -> float:
        """Average total tokens per debate."""
        if not self.debates:
            return 0.0
        return sum(d.total_tokens for d in self.debates) / len(self.debates)

    def to_dict(self) -> Dict:
        """Export comparison metrics as dictionary."""
        return {
            "model_name": self.model_name,
            "num_debates": self.num_debates,

            # Timing metrics
            "avg_total_time": round(self.avg_total_time, 2),
            "avg_time_per_round": round(self.avg_time_per_round, 2),
            "avg_time_per_agent": round(self.avg_time_per_agent, 2),

            # Token metrics
            "avg_tokens_per_second": round(self.avg_tokens_per_second, 1),
            "avg_total_tokens": round(self.avg_total_tokens, 0),

            # Debate metrics
            "avg_rounds": round(self.avg_rounds, 1),
            "avg_rounds_to_consensus": round(self.avg_rounds_to_consensus, 1),
            "consensus_rate": round(self.consensus_rate, 3),

            # Individual debates
            "debates": [d.to_dict() for d in self.debates]
        }

    def print_summary(self):
        """Print comparison summary."""
        print(f"\n{'='*70}")
        print(f"MODEL COMPARISON: {self.model_name}")
        print(f"{'='*70}")
        print(f"Debates Analyzed: {self.num_debates}")
        print(f"\n⏱️  AVERAGE TIMING")
        print(f"  Total Time:     {self.avg_total_time:.2f}s")
        print(f"  Time/Round:     {self.avg_time_per_round:.2f}s")
        print(f"  Time/Agent:     {self.avg_time_per_agent:.2f}s")

        print(f"\n🔢 AVERAGE TOKENS")
        print(f"  Total Tokens:   {self.avg_total_tokens:.0f}")
        print(f"  Tokens/Second:  {self.avg_tokens_per_second:.1f}")

        print(f"\n📊 DEBATE EFFICIENCY")
        print(f"  Avg Rounds:           {self.avg_rounds:.1f}")
        print(f"  Rounds to Consensus:  {self.avg_rounds_to_consensus:.1f}")
        print(f"  Consensus Rate:       {self.consensus_rate:.1%}")
        print(f"\n{'='*70}\n")
