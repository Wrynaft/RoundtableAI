"""
Debate evaluation metrics for measuring multi-agent system quality.

This module provides:
- Consensus quality metrics
- Reasoning consistency evaluation
- Tool utilization analysis
- Recommendation confidence scoring
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import statistics


@dataclass
class DebateMetrics:
    """Container for debate evaluation metrics."""
    consensus_quality: float
    reasoning_consistency: float
    tool_utilization_score: float
    recommendation_confidence: float
    convergence_rate: float
    rounds_to_consensus: int
    total_messages: int
    total_tool_calls: int
    average_response_time: float
    agent_agreement_matrix: Dict[str, Dict[str, float]]


def calculate_consensus_quality(
    votes: Dict[str, Dict[str, Any]],
    final_recommendation: str
) -> float:
    """
    Calculate the quality of consensus achieved in the debate.

    Higher scores indicate:
    - More agents agreeing on the final recommendation
    - Higher confidence levels among agreeing agents
    - Lower dissent from disagreeing agents

    Args:
        votes: Dictionary of agent_type -> vote info
        final_recommendation: The final synthesized recommendation

    Returns:
        Quality score from 0.0 to 1.0
    """
    if not votes:
        return 0.0

    agreement_score = 0.0
    confidence_sum = 0.0
    total_weight = 0.0

    for agent_type, vote in votes.items():
        rec = vote.get("recommendation", "").upper()
        conf = vote.get("confidence", 0.5)

        # Weight by confidence
        weight = conf
        total_weight += weight

        # Score based on agreement
        if rec == final_recommendation.upper():
            agreement_score += weight * 1.0
        elif rec == "HOLD":
            # HOLD is considered partial agreement with any recommendation
            agreement_score += weight * 0.5
        else:
            # Full disagreement
            agreement_score += weight * 0.0

        confidence_sum += conf

    # Combine agreement and confidence
    if total_weight > 0:
        weighted_agreement = agreement_score / total_weight
    else:
        weighted_agreement = 0.0

    avg_confidence = confidence_sum / len(votes) if votes else 0.0

    # Final quality score (60% agreement, 40% confidence)
    quality = 0.6 * weighted_agreement + 0.4 * avg_confidence

    return min(1.0, max(0.0, quality))


def calculate_reasoning_consistency(
    messages: List[Dict[str, Any]]
) -> float:
    """
    Evaluate consistency of reasoning across debate messages.

    Checks for:
    - Agents maintaining or improving their positions over rounds
    - Logical response to other agents' critiques
    - Recommendation stability within acceptable bounds

    Args:
        messages: List of debate message dictionaries

    Returns:
        Consistency score from 0.0 to 1.0
    """
    if len(messages) < 2:
        return 1.0  # Single message is perfectly consistent

    # Group messages by agent
    agent_messages: Dict[str, List[Dict]] = {}
    for msg in messages:
        agent = msg.get("agent_type", "unknown")
        if agent not in agent_messages:
            agent_messages[agent] = []
        agent_messages[agent].append(msg)

    consistency_scores = []

    for agent, msgs in agent_messages.items():
        if len(msgs) < 2:
            continue

        # Track recommendation changes
        recommendations = [m.get("recommendation", "HOLD") for m in msgs]
        confidences = [m.get("confidence", 0.5) for m in msgs]

        # Penalize flip-flopping
        rec_changes = sum(1 for i in range(1, len(recommendations))
                        if recommendations[i] != recommendations[i-1])

        # Ideal: 0 or 1 change (initial position + possible revision)
        # Penalize more than 1 change
        change_penalty = max(0, (rec_changes - 1) * 0.2)

        # Confidence should generally increase or stay stable
        conf_trend = sum(confidences[i] - confidences[i-1]
                        for i in range(1, len(confidences)))
        conf_score = 0.5 + min(0.5, max(-0.5, conf_trend / len(confidences)))

        # Combine scores
        agent_consistency = max(0, 1.0 - change_penalty) * conf_score
        consistency_scores.append(agent_consistency)

    if not consistency_scores:
        return 1.0

    return statistics.mean(consistency_scores)


def calculate_tool_utilization(
    tool_calls: List[Dict[str, Any]],
    expected_tools: List[str] = None
) -> float:
    """
    Evaluate how effectively tools were utilized.

    Args:
        tool_calls: List of tool call records
        expected_tools: List of tools that should have been called

    Returns:
        Utilization score from 0.0 to 1.0
    """
    if not tool_calls:
        return 0.0

    # Default expected tools for stock analysis
    if expected_tools is None:
        expected_tools = [
            "resolve_ticker_symbol",
            "finance_report_pull",
            "get_recent_articles",
            "get_article_sentiment",
            "get_stock_data",
            "analyze_stock_metrics"
        ]

    # Count unique tools called
    tools_called = set(tc.get("tool_name", "") for tc in tool_calls)

    # Calculate coverage
    coverage = len(tools_called.intersection(set(expected_tools))) / len(expected_tools)

    # Calculate success rate
    successful_calls = sum(1 for tc in tool_calls if tc.get("success", True))
    success_rate = successful_calls / len(tool_calls) if tool_calls else 0.0

    # Combine coverage and success
    utilization = 0.6 * coverage + 0.4 * success_rate

    return utilization


def calculate_convergence_rate(
    round_results: List[Dict[str, Any]]
) -> float:
    """
    Calculate how quickly the debate converged to consensus.

    Faster convergence with high consensus = better score.

    Args:
        round_results: List of round result dictionaries with consensus values

    Returns:
        Convergence rate from 0.0 to 1.0
    """
    if not round_results:
        return 0.0

    # Extract consensus values
    consensus_values = [r.get("consensus_percentage", 0) for r in round_results]

    if not consensus_values:
        return 0.0

    # Calculate improvement rate
    final_consensus = consensus_values[-1]
    initial_consensus = consensus_values[0]

    # Improvement from start to end
    improvement = final_consensus - initial_consensus

    # Speed bonus for early convergence
    rounds_used = len(consensus_values)
    max_rounds = 5  # Assuming max 5 rounds
    speed_factor = 1.0 - (rounds_used - 1) / (max_rounds - 1) if max_rounds > 1 else 1.0

    # Combine final consensus, improvement, and speed
    convergence = 0.5 * final_consensus + 0.3 * (improvement + 1) / 2 + 0.2 * speed_factor

    return min(1.0, max(0.0, convergence))


class DebateEvaluator:
    """
    Comprehensive evaluator for multi-agent debate quality.
    """

    def __init__(self):
        """Initialize the debate evaluator."""
        self.evaluation_history: List[DebateMetrics] = []

    def evaluate_debate(
        self,
        debate_state: Dict[str, Any],
        final_recommendation: str,
        tool_calls: List[Dict[str, Any]] = None,
        response_times: List[float] = None
    ) -> DebateMetrics:
        """
        Perform comprehensive evaluation of a debate.

        Args:
            debate_state: Full debate state dictionary
            final_recommendation: The final recommendation
            tool_calls: List of tool call records
            response_times: List of response times in seconds

        Returns:
            DebateMetrics with all evaluation scores
        """
        messages = debate_state.get("messages", [])
        votes = debate_state.get("current_votes", {})
        round_results = debate_state.get("round_results", [])

        # Calculate individual metrics
        consensus_quality = calculate_consensus_quality(votes, final_recommendation)
        reasoning_consistency = calculate_reasoning_consistency(messages)
        tool_utilization = calculate_tool_utilization(tool_calls or [])
        convergence_rate = calculate_convergence_rate(round_results)

        # Calculate recommendation confidence
        confidences = [v.get("confidence", 0.5) for v in votes.values()]
        recommendation_confidence = statistics.mean(confidences) if confidences else 0.5

        # Calculate rounds to consensus
        rounds_to_consensus = len(round_results)

        # Calculate average response time
        avg_response_time = statistics.mean(response_times) if response_times else 0.0

        # Build agreement matrix
        agent_agreement_matrix = self._build_agreement_matrix(votes)

        metrics = DebateMetrics(
            consensus_quality=consensus_quality,
            reasoning_consistency=reasoning_consistency,
            tool_utilization_score=tool_utilization,
            recommendation_confidence=recommendation_confidence,
            convergence_rate=convergence_rate,
            rounds_to_consensus=rounds_to_consensus,
            total_messages=len(messages),
            total_tool_calls=len(tool_calls) if tool_calls else 0,
            average_response_time=avg_response_time,
            agent_agreement_matrix=agent_agreement_matrix
        )

        self.evaluation_history.append(metrics)

        return metrics

    def _build_agreement_matrix(
        self,
        votes: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Build a matrix showing agreement between agents.

        Args:
            votes: Dictionary of agent votes

        Returns:
            Matrix of agreement scores between agents
        """
        agents = list(votes.keys())
        matrix = {}

        for agent1 in agents:
            matrix[agent1] = {}
            rec1 = votes[agent1].get("recommendation", "HOLD").upper()

            for agent2 in agents:
                if agent1 == agent2:
                    matrix[agent1][agent2] = 1.0
                else:
                    rec2 = votes[agent2].get("recommendation", "HOLD").upper()

                    if rec1 == rec2:
                        matrix[agent1][agent2] = 1.0
                    elif "HOLD" in [rec1, rec2]:
                        matrix[agent1][agent2] = 0.5
                    else:
                        matrix[agent1][agent2] = 0.0

        return matrix

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregate metrics across all evaluated debates.

        Returns:
            Dictionary of average metrics
        """
        if not self.evaluation_history:
            return {}

        return {
            "avg_consensus_quality": statistics.mean(
                m.consensus_quality for m in self.evaluation_history
            ),
            "avg_reasoning_consistency": statistics.mean(
                m.reasoning_consistency for m in self.evaluation_history
            ),
            "avg_tool_utilization": statistics.mean(
                m.tool_utilization_score for m in self.evaluation_history
            ),
            "avg_recommendation_confidence": statistics.mean(
                m.recommendation_confidence for m in self.evaluation_history
            ),
            "avg_convergence_rate": statistics.mean(
                m.convergence_rate for m in self.evaluation_history
            ),
            "avg_rounds_to_consensus": statistics.mean(
                m.rounds_to_consensus for m in self.evaluation_history
            ),
            "total_debates_evaluated": len(self.evaluation_history)
        }

    def generate_evaluation_report(self, metrics: DebateMetrics) -> str:
        """
        Generate a human-readable evaluation report.

        Args:
            metrics: DebateMetrics to report on

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("DEBATE EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")

        # Overall score
        overall_score = (
            metrics.consensus_quality * 0.3 +
            metrics.reasoning_consistency * 0.25 +
            metrics.tool_utilization_score * 0.2 +
            metrics.recommendation_confidence * 0.15 +
            metrics.convergence_rate * 0.1
        )

        report.append(f"Overall Quality Score: {overall_score:.1%}")
        report.append("")

        # Individual metrics
        report.append("DETAILED METRICS:")
        report.append("-" * 40)
        report.append(f"  Consensus Quality:     {metrics.consensus_quality:.1%}")
        report.append(f"  Reasoning Consistency: {metrics.reasoning_consistency:.1%}")
        report.append(f"  Tool Utilization:      {metrics.tool_utilization_score:.1%}")
        report.append(f"  Recommendation Conf.:  {metrics.recommendation_confidence:.1%}")
        report.append(f"  Convergence Rate:      {metrics.convergence_rate:.1%}")
        report.append("")

        # Statistics
        report.append("DEBATE STATISTICS:")
        report.append("-" * 40)
        report.append(f"  Rounds to Consensus:   {metrics.rounds_to_consensus}")
        report.append(f"  Total Messages:        {metrics.total_messages}")
        report.append(f"  Total Tool Calls:      {metrics.total_tool_calls}")
        report.append(f"  Avg Response Time:     {metrics.average_response_time:.2f}s")
        report.append("")

        # Agreement matrix
        report.append("AGENT AGREEMENT MATRIX:")
        report.append("-" * 40)
        agents = list(metrics.agent_agreement_matrix.keys())

        # Header
        header = "         " + "  ".join(f"{a[:5]:>5}" for a in agents)
        report.append(header)

        # Rows
        for agent1 in agents:
            row = f"{agent1[:8]:>8} "
            for agent2 in agents:
                score = metrics.agent_agreement_matrix[agent1].get(agent2, 0)
                row += f" {score:.1f}  "
            report.append(row)

        report.append("")
        report.append("=" * 60)

        return "\n".join(report)
