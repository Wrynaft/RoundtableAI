"""
RoundtableAI - Multi-Agent Stock Analysis Application

Run with: streamlit run app.py
"""
import os
import streamlit as st

st.set_page_config(
    page_title="RoundtableAI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# Phoenix Tracing (Optional - set PHOENIX_ENABLED=true in .env)
# Requires Docker: docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest
# =============================================================================
@st.cache_resource
def init_phoenix():
    """Initialize Phoenix tracing (runs once on app startup)."""
    from dotenv import load_dotenv
    load_dotenv()

    if os.getenv("PHOENIX_ENABLED", "").lower() != "true":
        return None

    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        # Connect to Phoenix running in Docker
        tracer_provider = register(
            project_name="roundtable-ai",
            endpoint="http://localhost:4317"
        )
        LangChainInstrumentor(tracer_provider=tracer_provider).instrument(skip_dep_check=True)

        return "http://localhost:6006"  # Return Phoenix UI URL
    except Exception as e:
        print(f"Phoenix initialization failed: {e}")
        print("Make sure Phoenix is running in Docker:")
        print("  docker run -d --name phoenix -p 6006:6006 -p 4317:4317 arizephoenix/phoenix:latest")
        return None


# Initialize Phoenix (cached - only runs once)
phoenix_url = init_phoenix()

# Show Phoenix link in sidebar if enabled
if phoenix_url:
    st.sidebar.success(f"🔍 [Phoenix Traces]({phoenix_url})")


# Redirect to Introduction page
st.switch_page("pages/0_🏠_Introduction.py")
