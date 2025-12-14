"""
Introduction Page - RoundtableAI Overview

This is the main landing page for the RoundtableAI application.
"""
import streamlit as st

st.set_page_config(
    page_title="Introduction - RoundtableAI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #4CAF50; font-size: 3em;">🎯 RoundtableAI</h1>
        <p style="color: #888; font-size: 20px;">
            LLM-Based Multi-Agent System for Bursa Stock Portfolio Construction
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Project Information Tabs
tab1, tab2, tab3 = st.tabs(["📖 About This Project", "❓ Problem Statement", "🎯 Objectives"])

with tab1:
    st.markdown("""
    ## About This Project

    **RoundtableAI** is an intelligent stock analysis system that uses multiple AI agents
    to debate and reach consensus on investment recommendations for Malaysian stocks
    listed on Bursa Malaysia.

    ### Key Features

    - **Multi-Agent Debate System**: Three specialized AI agents analyze stocks from different perspectives
    - **Risk-Aware Recommendations**: Automatically infers investor risk tolerance from natural language queries
    - **Consensus-Based Decisions**: Agents debate until reaching agreement, ensuring well-rounded analysis
    - **Malaysian Market Focus**: Specialized for Bursa Malaysia stocks with local market knowledge
    """)

with tab2:
    st.markdown("## Problem Statement")

    # Problem 1
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #f44336;
        margin-bottom: 20px;
    ">
        <h4 style="color: #f44336; margin-top: 0;">1. Fragmented data Sources and information overload causes lack of nuance in stock analysis</h4>
        <ul style="color: #aaa; margin-bottom: 0;">
            <li>Many existing works tend to focus only on technical rather than fundamental analyses, which lacks comprehensiveness.
                <span style="color: #888; font-size: 12px;">(Nti, I.K. et al., 2024)</span>
            </li>
            <li>Works typically only rely on single data source like historical price, requiring an autonomous method to aggregate market data.
                <span style="color: #888; font-size: 12px;">(Lin, Y. et al., 2022; Guo, J., 2024)</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Problem 2
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin-bottom: 20px;
    ">
        <h4 style="color: #FF9800; margin-top: 0;">2. Modern portfolio optimization methods such as deep learning lack interpretability</h4>
        <ul style="color: #aaa; margin-bottom: 0;">
            <li>Investors may lack confidence in following financial portfolio advice from 'black box' models due to lack of context-aware reasoning.
                <span style="color: #888; font-size: 12px;">(Yang Zhao et al., 2025)</span>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


with tab3:
    st.markdown("## Objectives")

    # Objective 1
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 15px;
    ">
        <div style="display: flex; align-items: flex-start;">
            <span style="
                background-color: #4CAF50;
                color: white;
                padding: 5px 12px;
                border-radius: 50%;
                font-weight: bold;
                margin-right: 15px;
                flex-shrink: 0;
            ">1</span>
            <p style="color: #ccc; margin: 0;">
                To develop a <strong style="color: #4CAF50;">multi-agent system integrating LLMs</strong> utilizing specialized analytical tools for stock data retrieval, valuation, and financial metric computation within the Bursa Malaysia market
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Objective 2
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin-bottom: 15px;
    ">
        <div style="display: flex; align-items: flex-start;">
            <span style="
                background-color: #2196F3;
                color: white;
                padding: 5px 12px;
                border-radius: 50%;
                font-weight: bold;
                margin-right: 15px;
                flex-shrink: 0;
            ">2</span>
            <p style="color: #ccc; margin: 0;">
                To present a <strong style="color: #2196F3;">web application</strong> for Bursa stock portfolio construction through recommendations supported by reasoning and transparent multi-agent debates
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Objective 3
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin-bottom: 15px;
    ">
        <div style="display: flex; align-items: flex-start;">
            <span style="
                background-color: #FF9800;
                color: white;
                padding: 5px 12px;
                border-radius: 50%;
                font-weight: bold;
                margin-right: 15px;
                flex-shrink: 0;
            ">3</span>
            <p style="color: #ccc; margin: 0;">
                To <strong style="color: #FF9800;">evaluate the accuracy, relevance and effectiveness</strong> of the multi-agent system's output to ensure reliable and precise recommendations for stock actions
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")

# AI Agents section (outside tabs)
st.markdown("## 🤖 AI Agents")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        padding: 20px;
        border-radius: 10px;
        height: 200px;
    ">
        <h4 style="color: #4CAF50;">📊 Fundamental Agent</h4>
        <p style="color: #ccc; font-size: 14px;">
            Analyzes financial statements, ratios, and company fundamentals
            including P/E ratio, ROE, debt levels, and dividend yields.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a3a5c 0%, #2d4a6d 100%);
        padding: 20px;
        border-radius: 10px;
        height: 200px;
    ">
        <h4 style="color: #2196F3;">📰 Sentiment Agent</h4>
        <p style="color: #ccc; font-size: 14px;">
            Monitors news sentiment, market buzz, and public perception
            to gauge investor sentiment and market mood.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #5c3a1a 0%, #6d4a2d 100%);
        padding: 20px;
        border-radius: 10px;
        height: 200px;
    ">
        <h4 style="color: #FF9800;">📈 Valuation Agent</h4>
        <p style="color: #ccc; font-size: 14px;">
            Evaluates risk-adjusted returns, volatility metrics,
            Sharpe ratio, and technical indicators for valuation.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# How it works section (outside tabs)
st.markdown("## 🔄 How It Works")

st.markdown("""
1. **Query Input**: Enter a natural language question about a Malaysian stock
2. **Classification**: System identifies the company and infers your risk tolerance
3. **Agent Analysis**: Specialized agents analyze the stock from their perspectives
4. **Debate**: Agents debate and challenge each other's findings
5. **Consensus**: A final recommendation is synthesized based on agent agreement
""")

st.markdown("---")

# Quick start section
st.markdown("## 🚀 Quick Start")

st.markdown("""
Ready to analyze a stock? Head over to the **Stock Analysis** page and enter your question!

**Example queries:**
- "Should I invest in Maybank? I'm looking for stable dividends."
- "Is CIMB a good investment for aggressive growth?"
- "What's your recommendation on Tenaga Nasional for a conservative investor?"
""")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 View EDA", use_container_width=True):
        st.switch_page("pages/1_📊_EDA.py")

with col2:
    if st.button("📈 Model Evaluation", use_container_width=True):
        st.switch_page("pages/2_📈_Model_Evaluation.py")

with col3:
    if st.button("💬 Start Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/3_💬_Stock_Analysis.py")

with col4:
    if st.button("ℹ️ About", use_container_width=True):
        st.switch_page("pages/4_ℹ️_About.py")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>WIH3001 Data Science Project | University of Malaya</p>
</div>
""", unsafe_allow_html=True)
