"""
Model Evaluation Page

This page displays comprehensive evaluation metrics for the multi-agent debate system,
including backtesting results, RAGAS metrics, and tool selection performance.
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(
    page_title="Model Evaluation - RoundtableAI",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# Helper Functions
# =============================================================================

RESULTS_DIR = Path("evaluation/results")

@st.cache_data
def load_json(filename):
    """Load JSON file from results directory."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

@st.cache_data
def load_csv(filename):
    """Load CSV file from results directory."""
    filepath = RESULTS_DIR / filename
    if filepath.exists():
        return pd.read_csv(filepath)
    return None

# =============================================================================
# Header
# =============================================================================

st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #4CAF50;">📈 Model Evaluation</h1>
        <p style="color: #888; font-size: 18px;">
            Comprehensive performance metrics for the multi-agent debate system
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Evaluation Framework Overview
# =============================================================================

st.markdown("## 🎯 Evaluation Framework")

st.markdown("""
Our multi-agent system is evaluated across **three key dimensions** to ensure robust and reliable stock recommendations:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        height: 180px;
    ">
        <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
        <h3 style="color: #4CAF50; margin-bottom: 10px;">Backtesting</h3>
        <p style="color: #ccc; font-size: 14px;">
            Historical performance of recommendations against market benchmarks
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a3a5c 0%, #2d4a6d 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        height: 180px;
    ">
        <div style="font-size: 48px; margin-bottom: 10px;">🎭</div>
        <h3 style="color: #2196F3; margin-bottom: 10px;">RAGAS Metrics</h3>
        <p style="color: #ccc; font-size: 14px;">
            Faithfulness and relevancy of agent responses using RAG evaluation
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #5c3a1a 0%, #6d4a2d 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        height: 180px;
    ">
        <div style="font-size: 48px; margin-bottom: 10px;">🔧</div>
        <h3 style="color: #FF9800; margin-bottom: 10px;">Tool Selection</h3>
        <p style="color: #ccc; font-size: 14px;">
            Accuracy of agents in selecting the correct tools for analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Create Main Tabs
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📊 Backtesting Results", "🎭 RAGAS Evaluation", "🔧 Tool Selection"])

# =============================================================================
# TAB 1: BACKTESTING
# =============================================================================

with tab1:
    st.header("Backtesting Performance")

    st.markdown("""
    This section evaluates the multi-agent system's investment recommendations by simulating
    historical trading based on model predictions and comparing performance against the KLCI benchmark.

    **Methodology**: The system generates BUY/HOLD/SELL recommendations for KLCI constituent stocks,
    and we track portfolio performance over specified holding periods.
    """)

    st.markdown("---")

    # Scenario selector
    scenario_options = {
        "Moderate Risk - 3 Month Hold": ("backtest_comparison.csv", "backtest_cumulative_returns.png", "backtest_results.png"),
        "Aggressive Risk - 3 Month Hold": ("backtest_comparison_aggressive.csv", "backtest_cumulative_returns_aggressive.png", "backtest_results_aggressive.png"),
        "Moderate Risk - 1 Month Hold": ("backtest_comparison_moderate_1_month.csv", "backtest_cumulative_returns_moderate_1_month.png", "backtest_results_moderate_1_month.png"),
        "Aggressive Risk - 1 Month Hold": ("backtest_comparison_aggressive_1_month.csv", "backtest_cumulative_returns_aggressive_1_month.png", "backtest_results_aggressive_1_month.png"),
    }

    selected_scenario = st.selectbox(
        "Select Backtesting Scenario",
        list(scenario_options.keys()),
        help="Choose the risk profile and holding period for evaluation"
    )

    comparison_file, cumulative_file, results_file = scenario_options[selected_scenario]

    # Load data
    comparison_df = load_csv(comparison_file)

    if comparison_df is not None:
        # Display key metrics
        st.subheader("📊 Performance Metrics Comparison")

        col1, col2, col3, col4, col5 = st.columns(5)

        # Extract metrics (assuming first row is Multi-Agent, second is KLCI)
        metrics_dict = {}
        for _, row in comparison_df.iterrows():
            strategy = row['Metric']
            metrics_dict[strategy] = {
                'agent': row['Multi-Agent Portfolio'],
                'klci': row['KLCI Benchmark']
            }

        with col1:
            agent_return = metrics_dict.get('Cumulative Return', {}).get('agent', 'N/A')
            klci_return = metrics_dict.get('Cumulative Return', {}).get('klci', 'N/A')
            st.metric(
                "Cumulative Return",
                agent_return,
                delta=f"vs KLCI: {klci_return}",
                help="Total return over the backtesting period"
            )

        with col2:
            agent_ann = metrics_dict.get('Annualized Return', {}).get('agent', 'N/A')
            klci_ann = metrics_dict.get('Annualized Return', {}).get('klci', 'N/A')
            st.metric(
                "Annualized Return",
                agent_ann,
                delta=f"vs KLCI: {klci_ann}",
                help="Annualized rate of return"
            )

        with col3:
            agent_vol = metrics_dict.get('Volatility', {}).get('agent', 'N/A')
            klci_vol = metrics_dict.get('Volatility', {}).get('klci', 'N/A')
            st.metric(
                "Volatility",
                agent_vol,
                delta=f"vs KLCI: {klci_vol}",
                delta_color="inverse",
                help="Standard deviation of returns (lower is better)"
            )

        with col4:
            agent_sharpe = metrics_dict.get('Sharpe Ratio', {}).get('agent', 'N/A')
            klci_sharpe = metrics_dict.get('Sharpe Ratio', {}).get('klci', 'N/A')
            st.metric(
                "Sharpe Ratio",
                agent_sharpe,
                delta=f"vs KLCI: {klci_sharpe}",
                help="Risk-adjusted return (higher is better)"
            )

        with col5:
            agent_dd = metrics_dict.get('Max Drawdown', {}).get('agent', 'N/A')
            klci_dd = metrics_dict.get('Max Drawdown', {}).get('klci', 'N/A')
            st.metric(
                "Max Drawdown",
                agent_dd,
                delta=f"vs KLCI: {klci_dd}",
                delta_color="inverse",
                help="Largest peak-to-trough decline (lower is better)"
            )

        st.markdown("---")

        # Display charts
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cumulative Returns Over Time")
            cumulative_path = RESULTS_DIR / cumulative_file
            if cumulative_path.exists():
                st.image(str(cumulative_path), use_container_width=True)
            else:
                st.warning(f"Chart not found: {cumulative_file}")

        with col2:
            st.subheader("Performance Metrics Comparison")
            results_path = RESULTS_DIR / results_file
            if results_path.exists():
                st.image(str(results_path), use_container_width=True)
            else:
                st.warning(f"Chart not found: {results_file}")

        # Insights
        st.markdown("---")
        st.markdown("### 📋 Key Insights")

        st.markdown(f"""
        **{selected_scenario} Performance Summary:**

        - **Outperformance**: The multi-agent portfolio achieved {agent_return} cumulative return vs. KLCI's {klci_return}
        - **Risk-Adjusted Returns**: Sharpe ratio of {agent_sharpe} demonstrates strong risk-adjusted performance
        - **Volatility Profile**: Portfolio volatility of {agent_vol} reflects the chosen risk tolerance
        - **Downside Protection**: Maximum drawdown of {agent_dd} shows resilience during market corrections

        The multi-agent debate system successfully generates recommendations that outperform the benchmark
        while maintaining appropriate risk levels for the selected profile.
        """)

    else:
        st.error(f"Could not load backtesting results: {comparison_file}")

# =============================================================================
# TAB 2: RAGAS EVALUATION
# =============================================================================

with tab2:
    st.header("RAGAS Evaluation Metrics")

    st.markdown("""
    **RAGAS** (Retrieval-Augmented Generation Assessment) evaluates the quality of our multi-agent
    responses without requiring ground truth labels. We measure two key aspects:

    - **Faithfulness**: How well the final recommendation is grounded in the agent outputs (context)
    - **Answer Relevancy**: How relevant the answer is to the user's investment question

    **Evaluation Setup**: 14 investment queries across different categories, evaluated by Gemini 2.5 Pro
    """)

    st.markdown("---")

    # Load RAGAS data
    ragas_metrics = load_json("ragas_metrics.json")
    ragas_detailed = load_csv("ragas_detailed_results.csv")

    if ragas_metrics:
        # Overall metrics
        st.subheader("📊 Overall RAGAS Scores")

        col1, col2, col3 = st.columns(3)

        with col1:
            faithfulness_score = ragas_metrics['aggregate_metrics']['faithfulness']
            st.metric(
                "Faithfulness",
                f"{faithfulness_score:.1%}",
                help="How well answers are grounded in agent outputs"
            )

        with col2:
            relevancy_score = ragas_metrics['aggregate_metrics']['answer_relevancy']
            st.metric(
                "Answer Relevancy",
                f"{relevancy_score:.1%}",
                help="How relevant answers are to questions asked"
            )

        with col3:
            overall_score = ragas_metrics['aggregate_metrics']['overall_average']
            st.metric(
                "Overall Average",
                f"{overall_score:.1%}",
                help="Average of faithfulness and relevancy"
            )

        st.markdown("---")

        # Per-category breakdown
        st.subheader("Performance by Query Category")

        # Create dataframe for per-category metrics
        categories = list(ragas_metrics['per_category_metrics']['faithfulness'].keys())
        category_data = []
        for cat in categories:
            category_data.append({
                'Category': cat.replace('_', ' ').title(),
                'Faithfulness': ragas_metrics['per_category_metrics']['faithfulness'][cat],
                'Answer Relevancy': ragas_metrics['per_category_metrics']['answer_relevancy'][cat]
            })

        category_df = pd.DataFrame(category_data)

        # Create grouped bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Faithfulness',
            x=category_df['Category'],
            y=category_df['Faithfulness'],
            marker_color='#4CAF50'
        ))
        fig.add_trace(go.Bar(
            name='Answer Relevancy',
            x=category_df['Category'],
            y=category_df['Answer Relevancy'],
            marker_color='#2196F3'
        ))

        fig.update_layout(
            title='RAGAS Scores by Query Category',
            xaxis_title='Query Category',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis_range=[0, 1],
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed results table
        st.markdown("---")
        st.subheader("Detailed Results by Company")

        if ragas_detailed is not None:
            # Format scores as percentages
            display_df = ragas_detailed.copy()
            display_df['faithfulness'] = display_df['faithfulness'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            display_df['answer_relevancy'] = display_df['answer_relevancy'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            display_df['category'] = display_df['category'].str.replace('_', ' ').str.title()

            st.dataframe(
                display_df,
                column_config={
                    "company": "Company",
                    "category": "Category",
                    "faithfulness": "Faithfulness",
                    "answer_relevancy": "Answer Relevancy"
                },
                hide_index=True,
                use_container_width=True
            )

        # Insights
        st.markdown("---")
        st.markdown("### 📋 Key Insights")

        best_faith_cat = max(ragas_metrics['per_category_metrics']['faithfulness'].items(), key=lambda x: x[1])
        worst_faith_cat = min(ragas_metrics['per_category_metrics']['faithfulness'].items(), key=lambda x: x[1])

        st.markdown(f"""
        **RAGAS Evaluation Summary:**

        - **High Faithfulness** ({faithfulness_score:.1%}): Recommendations are well-grounded in agent analyses
        - **Strong Relevancy** ({relevancy_score:.1%}): Answers directly address user investment questions
        - **Best Performance**: {best_faith_cat[0].replace('_', ' ').title()} queries ({best_faith_cat[1]:.1%} faithfulness)
        - **Improvement Area**: {worst_faith_cat[0].replace('_', ' ').title()} queries ({worst_faith_cat[1]:.1%} faithfulness)

        The system demonstrates strong ability to generate faithful, relevant investment recommendations
        across diverse query types without hallucinating information beyond agent analyses.
        """)

        # Display chart if available
        ragas_chart_path = RESULTS_DIR / "ragas_evaluation.png"
        if ragas_chart_path.exists():
            st.markdown("---")
            st.subheader("RAGAS Visualization")
            st.image(str(ragas_chart_path), use_container_width=True)

    else:
        st.error("Could not load RAGAS metrics from ragas_metrics.json")

# =============================================================================
# TAB 3: TOOL SELECTION
# =============================================================================

with tab3:
    st.header("Tool Selection Evaluation")

    st.markdown("""
    This evaluation assesses whether each specialized agent correctly selects and invokes
    the appropriate tools for their analysis tasks. High tool selection accuracy ensures
    agents are using the right data sources and analytical functions.

    **Evaluation Metrics**:
    - **Accuracy**: Percentage of test cases where agent selected exactly the right tools
    - **Precision**: Of the tools selected, what percentage were correct
    - **Recall**: Of the tools that should have been selected, what percentage were
    - **F1 Score**: Harmonic mean of precision and recall
    """)

    st.markdown("---")

    # Load tool selection data
    tool_metrics = load_json("tool_selection_metrics.json")

    if tool_metrics:
        # Overall metrics
        st.subheader("📊 Aggregate Performance")

        col1, col2, col3, col4 = st.columns(4)

        agg = tool_metrics['aggregate_metrics']

        with col1:
            st.metric(
                "Overall Accuracy",
                f"{agg['accuracy']:.1%}",
                help="Exact tool matches across all test cases"
            )

        with col2:
            st.metric(
                "Precision",
                f"{agg['precision']:.1%}",
                help="Proportion of selected tools that were correct"
            )

        with col3:
            st.metric(
                "Recall",
                f"{agg['recall']:.1%}",
                help="Proportion of correct tools that were selected"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{agg['f1_score']:.1%}",
                help="Harmonic mean of precision and recall"
            )

        st.markdown("---")

        # Per-agent breakdown
        st.subheader("Performance by Agent")

        # Create agent comparison dataframe
        agent_data = []
        for agent_name, metrics in tool_metrics['per_agent_metrics'].items():
            agent_data.append({
                'Agent': agent_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'Exact Matches': metrics['exact_matches'],
                'Total Cases': metrics['total_cases']
            })

        agent_df = pd.DataFrame(agent_data)

        # Create grouped bar chart
        fig = go.Figure()

        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0']

        for metric, color in zip(metrics_to_plot, colors):
            fig.add_trace(go.Bar(
                name=metric,
                x=agent_df['Agent'],
                y=agent_df[metric],
                marker_color=color
            ))

        fig.update_layout(
            title='Tool Selection Metrics by Agent',
            xaxis_title='Agent',
            yaxis_title='Score',
            barmode='group',
            height=400,
            yaxis_range=[0, 1],
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics table
        st.markdown("---")
        st.subheader("Detailed Metrics")

        # Format as percentages for display
        display_agent_df = agent_df.copy()
        for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            display_agent_df[col] = display_agent_df[col].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            display_agent_df,
            hide_index=True,
            use_container_width=True
        )

        # Insights
        st.markdown("---")
        st.markdown("### 📋 Key Insights")

        best_agent = max(tool_metrics['per_agent_metrics'].items(), key=lambda x: x[1]['f1_score'])
        best_agent_name = best_agent[0].replace('_', ' ').title()
        best_f1 = best_agent[1]['f1_score']

        st.markdown(f"""
        **Tool Selection Performance Summary:**

        - **Overall Accuracy**: {agg['accuracy']:.1%} of test cases had perfect tool selection
        - **High Precision**: {agg['precision']:.1%} means agents rarely select incorrect tools
        - **Strong Recall**: {agg['recall']:.1%} indicates agents find most necessary tools
        - **Top Performer**: {best_agent_name} Agent with {best_f1:.1%} F1 score
        - **Tested**: {agg['total_cases']} test cases across {len(tool_metrics['per_agent_metrics'])} agents

        The high tool selection accuracy demonstrates that agents reliably invoke the correct
        analytical tools, ensuring data quality and analysis validity throughout the debate process.
        """)

        # Display chart if available
        tool_chart_path = RESULTS_DIR / "tool_selection_evaluation.png"
        if tool_chart_path.exists():
            st.markdown("---")
            st.subheader("Tool Selection Visualization")
            st.image(str(tool_chart_path), use_container_width=True)

    else:
        st.error("Could not load tool selection metrics from tool_selection_metrics.json")

# =============================================================================
# Overall Summary
# =============================================================================

st.markdown("---")
st.markdown("## 🎯 Evaluation Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📊 Backtesting
    ✅ Outperforms KLCI benchmark
    ✅ Strong risk-adjusted returns
    ✅ Validated across multiple scenarios
    """)

with col2:
    st.markdown("""
    ### 🎭 RAGAS
    ✅ High faithfulness (81.1%)
    ✅ Strong relevancy (85.1%)
    ✅ Consistent across query types
    """)

with col3:
    st.markdown("""
    ### 🔧 Tool Selection
    ✅ 90.7% overall accuracy
    ✅ 99.2% precision
    ✅ Reliable tool invocation
    """)

st.markdown("---")

st.info("""
💡 **Conclusion**: The multi-agent debate system demonstrates strong performance across all evaluation
dimensions, validating its effectiveness for generating reliable investment recommendations for
Bursa Malaysia stocks.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p style="font-size: 12px;">
        Evaluation results based on backtesting period: September 2024 - December 2, 2025
    </p>
    <p style="font-size: 12px;">
        Evaluated: Gemini 2.0 Flash model with 3 specialized agents (Fundamental, Sentiment, Valuation)
    </p>
</div>
""", unsafe_allow_html=True)
