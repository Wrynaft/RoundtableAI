"""
Portfolio Builder Page - Multi-Stock Portfolio Construction Interface

This page allows users to:
- Add multiple stocks to a portfolio
- View correlation analysis between stocks
- Calculate portfolio-level metrics
- Run multi-agent analysis on each stock
- Get portfolio allocation recommendations
- Save/Load portfolios from CSV files
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path
import io

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
    if "portfolio_risk" not in st.session_state:
        st.session_state.portfolio_risk = "Moderate"


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


def export_portfolio_to_csv() -> Tuple[str, str]:
    """
    Export current portfolio to CSV format.

    Returns:
        Tuple of (csv_string, filename)
    """
    if not st.session_state.portfolio_stocks:
        return "", ""

    # Build export data
    export_data = []
    for stock in st.session_state.portfolio_stocks:
        ticker = stock["ticker"]
        name = stock["name"]

        # Get recommendation data if available
        rec_data = st.session_state.portfolio_recommendations.get(ticker, {})

        row = {
            "ticker": ticker,
            "company_name": name,
            "recommendation": rec_data.get("recommendation", ""),
            "confidence": rec_data.get("confidence", ""),
        }
        export_data.append(row)

    # Create DataFrame
    df = pd.DataFrame(export_data)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"portfolio_{timestamp}.csv"

    # Convert to CSV string
    csv_string = df.to_csv(index=False)

    return csv_string, filename


def validate_portfolio_csv(df: pd.DataFrame) -> Tuple[bool, str, List[Dict]]:
    """
    Validate uploaded portfolio CSV file.

    Args:
        df: DataFrame from uploaded CSV

    Returns:
        Tuple of (is_valid, error_message, valid_stocks)
    """
    # Check required columns
    required_columns = ["ticker", "company_name"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}", []

    # Validate data
    valid_stocks = []
    errors = []

    for idx, row in df.iterrows():
        ticker = row.get("ticker")
        company_name = row.get("company_name")

        # Check for empty values
        if pd.isna(ticker) or pd.isna(company_name) or str(ticker).strip() == "" or str(company_name).strip() == "":
            errors.append(f"Row {idx + 1}: Missing ticker or company name")
            continue

        # Basic format validation for ticker
        ticker_str = str(ticker).strip()
        if len(ticker_str) < 2 or len(ticker_str) > 20:
            errors.append(f"Row {idx + 1}: Invalid ticker format '{ticker_str}'")
            continue

        valid_stocks.append({
            "ticker": ticker_str,
            "name": str(company_name).strip()
        })

    if not valid_stocks:
        return False, "No valid stocks found in CSV. " + "; ".join(errors[:3]), []

    if len(valid_stocks) > 10:
        return False, f"Portfolio contains {len(valid_stocks)} stocks. Maximum 10 allowed.", []

    # Return success with any warnings
    warning = ""
    if errors:
        warning = f"Loaded {len(valid_stocks)} valid stocks. Skipped {len(errors)} invalid rows."

    return True, warning, valid_stocks


def import_portfolio_from_csv(uploaded_file) -> Tuple[bool, str]:
    """
    Import portfolio from uploaded CSV file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (success, message)
    """
    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Validate
        is_valid, message, valid_stocks = validate_portfolio_csv(df)

        if not is_valid:
            return False, message

        # Check for duplicates with existing portfolio
        existing_tickers = {s["ticker"] for s in st.session_state.portfolio_stocks}
        new_stocks = [s for s in valid_stocks if s["ticker"] not in existing_tickers]
        duplicate_count = len(valid_stocks) - len(new_stocks)

        if not new_stocks:
            return False, f"All {len(valid_stocks)} stocks are already in your portfolio."

        # Check if adding would exceed limit
        total_after_import = len(st.session_state.portfolio_stocks) + len(new_stocks)
        if total_after_import > 10:
            excess = total_after_import - 10
            return False, f"Cannot import. Would exceed maximum of 10 stocks by {excess}."

        # Add new stocks
        st.session_state.portfolio_stocks.extend(new_stocks)

        # Build success message
        success_msg = f"Successfully imported {len(new_stocks)} stocks."
        if duplicate_count > 0:
            success_msg += f" Skipped {duplicate_count} duplicates."
        if message:  # Include warnings
            success_msg += f" {message}"

        return True, success_msg

    except pd.errors.EmptyDataError:
        return False, "CSV file is empty."
    except pd.errors.ParserError as e:
        return False, f"CSV parsing error: {str(e)}"
    except Exception as e:
        return False, f"Failed to import portfolio: {str(e)}"


@st.dialog("Debate Transcript")
def view_transcript_dialog(stock_name: str, transcript: str):
    """Display debate transcript in a modal dialog."""
    st.markdown(f"### Analysis Transcript for {stock_name}")
    st.text_area("Full Transcript", value=transcript, height=600, disabled=True)


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

        # Risk Settings
        st.subheader("⚙️ Analysis Settings")
        
        st.session_state.portfolio_risk = st.selectbox(
            "Risk Appetite",
            options=["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(st.session_state.portfolio_risk),
            help="Determines how agents evaluate risk and potential returns."
        )

        # Portfolio Management
        st.subheader("💾 Portfolio Management")

        # Export portfolio
        if st.session_state.portfolio_stocks:
            csv_data, filename = export_portfolio_to_csv()
            if csv_data:
                st.download_button(
                    label="📥 Export Portfolio",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    help="Download your portfolio as a CSV file",
                    use_container_width=True
                )
        else:
            st.button(
                "📥 Export Portfolio",
                disabled=True,
                help="Add stocks to your portfolio before exporting",
                use_container_width=True
            )

        # Import portfolio
        uploaded_file = st.file_uploader(
            "📤 Import Portfolio",
            type=["csv"],
            help="Upload a CSV file with columns: ticker, company_name",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            if st.button("📤 Import from CSV", type="primary", width='stretch'):
                success, message = import_portfolio_from_csv(uploaded_file)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        # CSV format info
        with st.expander("ℹ️ CSV Format Help"):
            st.markdown("""
            **Required columns:**
            - `ticker`: Stock ticker symbol (e.g., 1155.KL)
            - `company_name`: Full company name (e.g., Maybank Bhd)

            **Optional columns:**
            - `recommendation`: BUY/SELL/HOLD (from analysis)
            - `confidence`: Confidence score (0-1)

            **Example CSV:**
            ```
            ticker,company_name
            1155.KL,Maybank Bhd
            1023.KL,CIMB Group Holdings Bhd
            1295.KL,Public Bank Bhd
            ```

            **Constraints:**
            - Maximum 10 stocks per portfolio
            - Duplicate tickers will be skipped
            - Invalid rows will be skipped with warnings
            """)

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
                    col_btn1, col_btn2 = st.columns([1, 1])
                    with col_btn1:
                        if has_analysis and "transcript" in rec_data:
                            if st.button("📄 View Transcript", key=f"trans_{ticker}", use_container_width=True):
                                view_transcript_dialog(name, rec_data["transcript"])
                    
                    with col_btn2:
                        if st.button("🗑️ Remove", key=f"remove_{ticker}", type="secondary", use_container_width=True):
                            remove_stock_from_portfolio(ticker)
                            st.rerun()


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
                                # Use selected risk tolerance
                                risk_tolerance = st.session_state.portfolio_risk.lower()
                                result = orchestrator.route_query(query, risk_tolerance=risk_tolerance)

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
                                        "route_type": result.get("route_type", "unknown"),
                                        "transcript": orchestrator.get_debate_transcript() if result.get("route_type") == "debate" else None
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
