"""
EDA Page - Exploratory Data Analysis

This page will contain exploratory data analysis of the stock data,
news sentiment data, and other relevant datasets.
"""
import streamlit as st

st.set_page_config(
    page_title="EDA - RoundtableAI",
    page_icon="📊",
    layout="wide"
)

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #4CAF50;">📊 Exploratory Data Analysis</h1>
        <p style="color: #888; font-size: 18px;">
            Analyzing stock data, news sentiment, and market trends
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Placeholder content
st.markdown("""
<div style="
    background-color: #1E1E1E;
    padding: 50px;
    border-radius: 15px;
    text-align: center;
    margin: 20px 0;
">
    <div style="font-size: 80px; margin-bottom: 20px;">🚧</div>
    <h2 style="color: #FF9800;">Under Construction</h2>
    <p style="color: #888; font-size: 16px; margin-top: 20px;">
        This page will contain exploratory data analysis including:
    </p>
</div>
""", unsafe_allow_html=True)

# Planned sections
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 📈 Stock Price Analysis
    - Historical price trends
    - Volume patterns
    - Volatility analysis
    - Sector comparisons

    ### 📰 News Sentiment Analysis
    - Sentiment distribution over time
    - Company coverage frequency
    - Sentiment vs price correlation
    """)

with col2:
    st.markdown("""
    ### 💹 Financial Metrics
    - P/E ratio distributions
    - ROE comparisons
    - Dividend yield analysis
    - Market cap breakdown

    ### 🔍 Data Quality
    - Missing data analysis
    - Data coverage statistics
    - Temporal coverage
    """)

st.markdown("---")

st.info("💡 **Tip**: EDA visualizations will be added here to provide insights into the underlying data used by our multi-agent system.")
