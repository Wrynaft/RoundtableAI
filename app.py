"""
RoundtableAI - Multi-Agent Stock Analysis Application

Run with: streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="RoundtableAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Redirect to Introduction page
st.switch_page("pages/0_🏠_Introduction.py")
