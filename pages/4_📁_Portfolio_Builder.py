"""
Portfolio Builder Page - Multi-Stock Portfolio Construction Interface

This page allows users to:
- Add multiple stocks to a portfolio
- View correlation analysis between stocks
- Calculate portfolio-level metrics
- Run multi-agent analysis on each stock
- Get portfolio allocation recommendations
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

st.set_page_config(
    page_title="Portfolio Builder - RoundtableAI",
    page_icon="📁",
    layout="wide"
)

# Import utilities
from utils.ticker_resolver import resolve_ticker_symbol
from tools.portfolio_optimizer import PortfolioOptimizer


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables for portfolio builder."""
    if "portfolio_stocks" not in st.session_state:
        st.session_state.portfolio_stocks = []  # List of {"ticker": str, "name": str}
    if "portfolio_recommendations" not in st.session_state:
        st.session_state.portfolio_recommendations = {}  # ticker -> recommendation
    if "portfolio_metrics" not in st.session_state:
        st.session_state.portfolio_metrics = None
    if "portfolio_analyzing" not in st.session_state:
        st.session_state.portfolio_analyzing = False
    if "allocation_method" not in st.session_state:
        st.session_state.allocation_method = "equal"


# =============================================================================
# Helper Functions
# =============================================================================

def add_stock_to_portfolio(company_name: str) -> tuple[bool, str]:
    """
    Add a stock to the portfolio.

    Returns:
        Tuple of (success, message)
    """
    if not company_name.strip():
        return False, "Please enter a company name or ticker symbol."

    # Check if already at max capacity
    if len(st.session_state.portfolio_stocks) >= 10:
        return False, "Maximum 10 stocks allowed in portfolio."

    # Resolve ticker (use .invoke() since it's a LangChain tool)
    result = resolve_ticker_symbol.invoke(company_name)

    if not result.get("success"):
        error_msg = result.get("error", "Could not resolve ticker symbol.")
        if "candidates" in result:
            candidates = ", ".join([c["company_name"] for c in result["candidates"][:3]])
            error_msg += f" Did you mean: {candidates}?"
        return False, error_msg

    ticker = result["ticker"]
    full_name = result["company_name"]

    # Check if already in portfolio
    existing_tickers = [s["ticker"] for s in st.session_state.portfolio_stocks]
    if ticker in existing_tickers:
        return False, f"{full_name} is already in your portfolio."

    # Add to portfolio
    st.session_state.portfolio_stocks.append({
        "ticker": ticker,
        "name": full_name
    })

    return True, f"Added {full_name} ({ticker}) to portfolio."


def remove_stock_from_portfolio(ticker: str):
    """Remove a stock from the portfolio."""
    st.session_state.portfolio_stocks = [
        s for s in st.session_state.portfolio_stocks
        if s["ticker"] != ticker
    ]
    # Also remove from recommendations if exists
    if ticker in st.session_state.portfolio_recommendations:
        del st.session_state.portfolio_recommendations[ticker]
    # Clear cached metrics
    st.session_state.portfolio_metrics = None


def clear_portfolio():
    """Clear all stocks from portfolio."""
    st.session_state.portfolio_stocks = []
    st.session_state.portfolio_recommendations = {}
    st.session_state.portfolio_metrics = None


def get_portfolio_tickers() -> List[str]:
    """Get list of ticker symbols in portfolio."""
    return [s["ticker"] for s in st.session_state.portfolio_stocks]


# =============================================================================
# Visualization Functions
# =============================================================================

def render_correlation_heatmap(correlation_matrix: pd.DataFrame):
    """Render correlation matrix as a heatmap."""
    if correlation_matrix.empty:
        st.warning("Unable to calculate correlation matrix. Insufficient data.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap='RdYlGn',
        center=0,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
        cbar_kws={'shrink': 0.8, 'label': 'Correlation'}
    )

    ax.set_title('Portfolio Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()

    st.pyplot(fig)
    plt.close()


def render_portfolio_metrics(metrics: Dict):
    """Render portfolio metrics in a nice format."""
    if not metrics.get("success"):
        st.error(metrics.get("error", "Unable to calculate portfolio metrics."))
        return

    pm = metrics.get("portfolio_metrics", {})
    dm = metrics.get("diversification_metrics", {})

    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Expected Annual Return",
            value=f"{pm.get('expected_return_pct', 0):.2f}%",
            help="Annualized expected portfolio return based on historical data"
        )

    with col2:
        st.metric(
            label="Portfolio Volatility",
            value=f"{pm.get('volatility_pct', 0):.2f}%",
            help="Annualized portfolio volatility (risk)"
        )

    with col3:
        sharpe = pm.get('sharpe_ratio', 0)
        st.metric(
            label="Sharpe Ratio",
            value=f"{sharpe:.2f}",
            help="Risk-adjusted return (higher is better)"
        )

    with col4:
        div_ratio = dm.get('diversification_ratio', 1)
        st.metric(
            label="Diversification Ratio",
            value=f"{div_ratio:.2f}",
            help="Ratio > 1 indicates diversification benefit"
        )

    # Diversification info
    avg_corr = dm.get('average_correlation', 0)
    if avg_corr > 0.7:
        st.warning(f"High average correlation ({avg_corr:.2f}). Consider adding less correlated stocks for better diversification.")
    elif avg_corr < 0.3:
        st.success(f"Good diversification! Average correlation is low ({avg_corr:.2f}).")
    else:
        st.info(f"Average correlation: {avg_corr:.2f}")


def render_weights_table(metrics: Dict, recommendations: Dict):
    """Render portfolio weights table with recommendations."""
    weights = metrics.get("weights", {})
    individual = metrics.get("individual_stocks", {})

    if not weights:
        return

    # Build table data
    table_data = []
    for ticker, weight in weights.items():
        stock_info = individual.get(ticker, {})
        rec_info = recommendations.get(ticker, {})

        row = {
            "Ticker": ticker,
            "Weight": f"{weight * 100:.1f}%",
            "Exp. Return": f"{stock_info.get('expected_return', 0) * 100:.1f}%",
            "Volatility": f"{stock_info.get('volatility', 0) * 100:.1f}%",
            "Recommendation": rec_info.get("recommendation", "N/A"),
            "Confidence": f"{rec_info.get('confidence', 0) * 100:.0f}%" if rec_info.get('confidence') else "N/A"
        }
        table_data.append(row)

    df = pd.DataFrame(table_data)

    # Style the dataframe
    def color_recommendation(val):
        if val == "BUY":
            return 'background-color: #c8e6c9; color: #2e7d32'
        elif val == "SELL":
            return 'background-color: #ffcdd2; color: #c62828'
        elif val == "HOLD":
            return 'background-color: #fff9c4; color: #f57f17'
        return ''

    styled_df = df.style.map(color_recommendation, subset=['Recommendation'])
    st.dataframe(styled_df, width='stretch', hide_index=True)


def render_allocation_pie_chart(weights: Dict):
    """Render pie chart of portfolio allocation."""
    if not weights:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    labels = list(weights.keys())
    sizes = list(weights.values())

    # Colors
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )

    # Style
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    ax.set_title('Portfolio Allocation', fontsize=14, fontweight='bold')

    st.pyplot(fig)
    plt.close()


# =============================================================================
# Main Page
# =============================================================================

def main():
    init_session_state()

    # Header
    st.title("📁 Portfolio Builder")
    st.markdown("""
    Build and analyze a multi-stock portfolio. Add stocks, view correlation analysis,
    and calculate portfolio-level metrics including expected return, volatility, and Sharpe ratio.
    """)

    st.divider()

    # ==========================================================================
    # Sidebar - Settings
    # ==========================================================================
    with st.sidebar:
        st.header("Portfolio Settings")

        # Allocation method
        allocation_method = st.radio(
            "Allocation Method",
            options=["equal", "confidence"],
            format_func=lambda x: "Equal Weight" if x == "equal" else "Confidence Weighted",
            help="Equal: Same weight for all stocks. Confidence: Weight by agent confidence scores."
        )
        st.session_state.allocation_method = allocation_method

        st.divider()

        # Quick add popular stocks
        st.subheader("Quick Add")
        popular_stocks = ["Maybank", "CIMB", "Public Bank", "Tenaga", "Petronas Chemicals"]

        for stock in popular_stocks:
            if st.button(f"➕ {stock}", key=f"quick_{stock}", width='stretch'):
                success, msg = add_stock_to_portfolio(stock)
                if success:
                    st.success(msg)
                else:
                    st.warning(msg)
                st.rerun()

        st.divider()

        # Clear portfolio
        if st.session_state.portfolio_stocks:
            if st.button("🗑️ Clear Portfolio", type="secondary", width='stretch'):
                clear_portfolio()
                st.rerun()

    # ==========================================================================
    # Main Content
    # ==========================================================================

    # Stock input section
    col1, col2 = st.columns([3, 1])

    with col1:
        stock_input = st.text_input(
            "Add Stock",
            placeholder="Enter company name or ticker (e.g., Maybank, 1155.KL)",
            label_visibility="collapsed"
        )

    with col2:
        add_clicked = st.button("➕ Add to Portfolio", type="primary", width='stretch')

    if add_clicked and stock_input:
        success, msg = add_stock_to_portfolio(stock_input)
        if success:
            st.success(msg)
            st.rerun()
        else:
            st.error(msg)

    # Current portfolio display
    st.subheader(f"📊 Current Portfolio ({len(st.session_state.portfolio_stocks)} stocks)")

    if not st.session_state.portfolio_stocks:
        st.info("Your portfolio is empty. Add stocks using the input above or quick-add buttons in the sidebar.")
    else:
        # Display stocks in improved cards
        num_stocks = len(st.session_state.portfolio_stocks)
        cols_per_row = min(num_stocks, 3)  # Max 3 per row for better readability

        for row_start in range(0, num_stocks, cols_per_row):
            cols = st.columns(cols_per_row)
            for col_idx, stock_idx in enumerate(range(row_start, min(row_start + cols_per_row, num_stocks))):
                stock = st.session_state.portfolio_stocks[stock_idx]
                ticker = stock['ticker']
                name = stock['name']

                # Check if stock has been analyzed
                has_analysis = ticker in st.session_state.portfolio_recommendations
                rec_data = st.session_state.portfolio_recommendations.get(ticker, {})
                recommendation = rec_data.get("recommendation", None)
                confidence = rec_data.get("confidence", 0)

                with cols[col_idx]:
                    # Determine card color based on recommendation
                    if recommendation == "BUY":
                        border_color = "#4CAF50"
                        rec_icon = "📈"
                        rec_color = "#c8e6c9"
                    elif recommendation == "SELL":
                        border_color = "#f44336"
                        rec_icon = "📉"
                        rec_color = "#ffcdd2"
                    elif recommendation == "HOLD":
                        border_color = "#FF9800"
                        rec_icon = "⏸️"
                        rec_color = "#fff9c4"
                    else:
                        border_color = "#424242"
                        rec_icon = "⏳"
                        rec_color = "#424242"

                    # Custom styled card
                    st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                        border-left: 4px solid {border_color};
                        border-radius: 8px;
                        padding: 16px;
                        margin-bottom: 8px;
                        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <p style="font-size: 1.1em; font-weight: bold; color: #fff; margin: 0;">
                                    {ticker}
                                </p>
                                <p style="font-size: 0.85em; color: #888; margin: 4px 0 0 0;">
                                    {name[:35] + '...' if len(name) > 35 else name}
                                </p>
                            </div>
                        </div>
                        {"<div style='margin-top: 12px; padding: 8px; background: " + rec_color + "; border-radius: 4px; text-align: center;'><span style='color: #000; font-weight: bold;'>" + rec_icon + " " + recommendation + " (" + f"{confidence*100:.0f}%" + ")</span></div>" if has_analysis else "<div style='margin-top: 12px; padding: 8px; background: #333; border-radius: 4px; text-align: center;'><span style='color: #888;'>⏳ Not analyzed</span></div>"}
                    </div>
                    """, unsafe_allow_html=True)

                    # Remove button below the card
                    if st.button("🗑️ Remove", key=f"remove_{ticker}", type="secondary"):
                        remove_stock_from_portfolio(ticker)
                        st.rerun()

        st.divider()

        # =======================================================================
        # Portfolio Analysis Section
        # =======================================================================

        if len(st.session_state.portfolio_stocks) >= 2:
            st.subheader("📈 Portfolio Analysis")

            # Initialize optimizer
            optimizer = PortfolioOptimizer()
            tickers = get_portfolio_tickers()

            # Tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["📊 Metrics", "🔗 Correlation", "🥧 Allocation"])

            with tab1:
                # Calculate portfolio metrics
                if st.session_state.allocation_method == "confidence":
                    weights = optimizer.confidence_weighted(st.session_state.portfolio_recommendations)
                else:
                    weights = optimizer.equal_weight(tickers)

                metrics = optimizer.calculate_portfolio_metrics(tickers, weights)

                if metrics.get("success"):
                    render_portfolio_metrics(metrics)
                    st.divider()

                    st.markdown("### Stock Details")
                    render_weights_table(metrics, st.session_state.portfolio_recommendations)
                else:
                    st.error("Unable to calculate portfolio metrics. Please ensure all stocks have available data.")

            with tab2:
                st.markdown("### Correlation Matrix")
                st.caption("Shows how stocks move together. Lower correlation = better diversification.")

                corr_matrix = optimizer.calculate_correlation_matrix(tickers)
                render_correlation_heatmap(corr_matrix)

                # Correlation insights
                if not corr_matrix.empty:
                    # Find highest and lowest correlations
                    corr_values = corr_matrix.values
                    np.fill_diagonal(corr_values, np.nan)

                    max_corr = np.nanmax(corr_values)
                    min_corr = np.nanmin(corr_values)

                    max_idx = np.unravel_index(np.nanargmax(corr_values), corr_values.shape)
                    min_idx = np.unravel_index(np.nanargmin(corr_values), corr_values.shape)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"**Highest correlation:** {corr_matrix.index[max_idx[0]]} & {corr_matrix.columns[max_idx[1]]} ({max_corr:.2f})")
                    with col2:
                        st.info(f"**Lowest correlation:** {corr_matrix.index[min_idx[0]]} & {corr_matrix.columns[min_idx[1]]} ({min_corr:.2f})")

            with tab3:
                st.markdown("### Portfolio Allocation")

                if st.session_state.allocation_method == "confidence":
                    weights = optimizer.confidence_weighted(st.session_state.portfolio_recommendations)
                    st.caption("Weights based on agent confidence scores. Stocks with higher confidence get larger allocations.")
                else:
                    weights = optimizer.equal_weight(tickers)
                    st.caption("Equal weight allocation - each stock receives the same weight.")

                col1, col2 = st.columns([1, 1])

                with col1:
                    render_allocation_pie_chart(weights)

                with col2:
                    st.markdown("#### Allocation Breakdown")
                    for ticker, weight in weights.items():
                        st.progress(weight, text=f"{ticker}: {weight*100:.1f}%")

        elif len(st.session_state.portfolio_stocks) == 1:
            st.info("Add at least 2 stocks to see portfolio analysis.")

        # =======================================================================
        # Multi-Agent Analysis Section
        # =======================================================================

        st.divider()
        st.subheader("🤖 Multi-Agent Analysis")

        if not st.session_state.portfolio_stocks:
            st.info("Add stocks to your portfolio to run multi-agent analysis.")
        else:
            st.markdown("""
            Run the multi-agent debate system on each stock in your portfolio.
            Each stock will be analyzed by the Fundamental, Sentiment, and Valuation agents.
            """)

            # Check if we have recommendations
            analyzed_count = len(st.session_state.portfolio_recommendations)
            total_count = len(st.session_state.portfolio_stocks)

            if analyzed_count > 0:
                st.success(f"✅ {analyzed_count}/{total_count} stocks analyzed")

            if st.button("🔍 Analyze All Stocks", type="primary", disabled=st.session_state.portfolio_analyzing):
                st.session_state.portfolio_analyzing = True

                # Import orchestrator
                try:
                    from agents.orchestrator import DebateOrchestrator

                    # Create analysis container
                    analysis_container = st.container()

                    with analysis_container:
                        st.markdown("---")
                        st.markdown("### 🔄 Running Multi-Agent Analysis")

                        # Progress indicators
                        progress_bar = st.progress(0)
                        status_container = st.empty()
                        details_container = st.empty()

                        # Initialize orchestrator with status update
                        with status_container:
                            st.info("⏳ Initializing debate orchestrator...")

                        orchestrator = DebateOrchestrator()

                        for idx, stock in enumerate(st.session_state.portfolio_stocks):
                            ticker = stock["ticker"]
                            name = stock["name"]

                            # Update progress
                            progress_pct = (idx) / total_count
                            progress_bar.progress(progress_pct)

                            # Show current stock being analyzed
                            with status_container:
                                st.info(f"🔍 Analyzing **{name}** ({ticker}) — Stock {idx + 1} of {total_count}")

                            with details_container:
                                st.markdown(f"""
                                <div style="background-color: #1e1e1e; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50;">
                                    <p style="margin: 0; color: #888;">📊 Fundamental Agent analyzing financial statements...</p>
                                    <p style="margin: 5px 0; color: #888;">📰 Sentiment Agent processing news articles...</p>
                                    <p style="margin: 0; color: #888;">📈 Valuation Agent calculating risk metrics...</p>
                                </div>
                                """, unsafe_allow_html=True)

                            try:
                                # Run analysis
                                query = f"Should I invest in {name}? Provide a recommendation."
                                result = orchestrator.route_query(query)

                                # Extract recommendation from result (orchestrator returns a dict)
                                if isinstance(result, dict):
                                    rec_value = result.get("recommendation", "HOLD")
                                    # Handle enum values
                                    if hasattr(rec_value, 'value'):
                                        rec_value = rec_value.value

                                    st.session_state.portfolio_recommendations[ticker] = {
                                        "recommendation": rec_value,
                                        "confidence": result.get("confidence", 0.5),
                                        "summary": result.get("response", ""),  # 'response' contains the summary
                                        "consensus": result.get("consensus", None),
                                        "route_type": result.get("route_type", "unknown")
                                    }
                                else:
                                    st.session_state.portfolio_recommendations[ticker] = {
                                        "recommendation": "HOLD",
                                        "confidence": 0.5,
                                        "summary": "Unexpected result format"
                                    }
                            except Exception as e:
                                st.warning(f"Could not analyze {ticker}: {str(e)}")
                                st.session_state.portfolio_recommendations[ticker] = {
                                    "recommendation": "HOLD",
                                    "confidence": 0.5,
                                    "summary": f"Analysis failed: {str(e)}"
                                }

                            # Update progress after completion
                            progress_bar.progress((idx + 1) / total_count)

                        # Show completion
                        with status_container:
                            st.success(f"✅ Analysis complete! {total_count} stocks analyzed.")
                        details_container.empty()

                    st.session_state.portfolio_analyzing = False
                    st.rerun()

                except ImportError as e:
                    st.error(f"Could not import orchestrator: {e}")
                    st.session_state.portfolio_analyzing = False
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.session_state.portfolio_analyzing = False

            # Display existing recommendations
            if st.session_state.portfolio_recommendations:
                st.markdown("### Analysis Results")

                for ticker, rec in st.session_state.portfolio_recommendations.items():
                    stock_name = next(
                        (s["name"] for s in st.session_state.portfolio_stocks if s["ticker"] == ticker),
                        ticker
                    )

                    recommendation = rec.get("recommendation", "N/A")
                    confidence = rec.get("confidence", 0)

                    # Color based on recommendation
                    if recommendation == "BUY":
                        color = "green"
                        icon = "📈"
                    elif recommendation == "SELL":
                        color = "red"
                        icon = "📉"
                    else:
                        color = "orange"
                        icon = "⏸️"

                    with st.expander(f"{icon} **{stock_name}** ({ticker}) - {recommendation} ({confidence*100:.0f}% confidence)"):
                        summary = rec.get("summary", "")
                        if summary:
                            st.markdown("**Analysis Summary:**")
                            st.markdown(summary)
                        else:
                            st.info("No detailed summary available for this analysis.")

                        # Show metrics
                        st.divider()
                        cols = st.columns(3)
                        with cols[0]:
                            st.metric("Recommendation", recommendation)
                        with cols[1]:
                            st.metric("Confidence", f"{confidence*100:.0f}%")
                        with cols[2]:
                            consensus = rec.get("consensus")
                            if consensus:
                                st.metric("Consensus", f"{consensus*100:.0f}%")
                            else:
                                st.metric("Analysis Type", rec.get("route_type", "N/A").title())


if __name__ == "__main__":
    main()
