"""
About Page - Changelog, Author Info, and Acknowledgements

This page provides project information, version history, and credits.
"""
import streamlit as st
import base64
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def get_image_base64(image_path: Path) -> str:
    """Convert image to base64 string for embedding in HTML."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

st.set_page_config(
    page_title="About - RoundtableAI",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #9C27B0; font-size: 2.5em;">ℹ️ About RoundtableAI</h1>
        <p style="color: #888; font-size: 18px;">
            Project Information, Changelog & Acknowledgements
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📋 Changelog", "👤 About the Author", "🙏 Special Thanks"])

# =============================================================================
# CHANGELOG TAB
# =============================================================================
with tab1:
    st.markdown("## 📋 Version History")

    # Version 1.0.0
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 20px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="color: #4CAF50; margin: 0;">v1.0.0</h3>
            <span style="
                background-color: #4CAF50;
                color: white;
                padding: 3px 10px;
                border-radius: 12px;
                font-size: 12px;
            ">Latest</span>
        </div>
        <p style="color: #888; font-size: 12px; margin: 5px 0 15px 0;">15 December 2025</p>
        <p style="color: #ccc; margin-bottom: 10px;"><strong>Initial Release - DEPLOYMENT</strong></p>
        <ul style="color: #aaa; margin-bottom: 0;">
            <li><strong style="color: #4CAF50;">Multi-Agent Debate System</strong> - Three specialized AI agents (Fundamental, Sentiment, Valuation) analyze stocks through structured debate</li>
            <li><strong style="color: #4CAF50;">Real-time Streaming</strong> - Live progress updates during debates and token streaming for single agent responses</li>
            <li><strong style="color: #4CAF50;">Risk Tolerance Classification</strong> - Automatic inference of investor risk profile from natural language queries</li>
            <li><strong style="color: #4CAF50;">Consensus-Based Recommendations</strong> - Agents debate until reaching agreement with transparent reasoning</li>
            <li><strong style="color: #4CAF50;">Single Agent Response</strong> - For general queries without debate</li>
            <li><strong style="color: #4CAF50;">Interactive UI</strong> - Streamlit-based web application with debate timeline visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Future versions placeholder
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #666;
        margin-bottom: 20px;
        opacity: 0.7;
    ">
        <h3 style="color: #888; margin: 0;">🚀 Upcoming Features</h3>
        <p style="color: #666; font-size: 12px; margin: 5px 0 15px 0;">Planned for future releases</p>
        <ul style="color: #888; margin-bottom: 0;">
            <li>Exploratory Data Analysis Update</li>
            <li>Model Evaluation</li>
            <li>Conversation Memory</li>
            <li>Portfolio Tracking (TBD)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# ABOUT THE AUTHOR TAB
# =============================================================================
with tab2:
    st.markdown("## 👤 About the Author")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Profile picture
        profile_pic_path = ASSETS_DIR / "Xorque.png"

        if profile_pic_path.exists():
            # Get image as base64 for full HTML control
            img_base64 = get_image_base64(profile_pic_path)
            img_ext = profile_pic_path.suffix.lower().replace(".", "")
            if img_ext == "jpg":
                img_ext = "jpeg"

            st.markdown(f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px 0;
            ">
                <img src="data:image/{img_ext};base64,{img_base64}"
                     style="
                         width: 200px;
                         height: 200px;
                         border-radius: 50%;
                         border: 4px solid #9C27B0;
                         box-shadow: 0 4px 15px rgba(156, 39, 176, 0.3);
                         object-fit: cover;
                     "
                />
            </div>
            """, unsafe_allow_html=True)
        else:
            # Fallback placeholder if image doesn't exist
            st.markdown("""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px 0;
            ">
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    width: 200px;
                    height: 200px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    <span style="font-size: 80px;">👨‍💻</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="
            background-color: #1E1E1E;
            padding: 25px;
            border-radius: 10px;
            margin: 10px 0;
        ">
            <h3 style="color: #9C27B0; margin-top: 0;">Ryan Chin Jian Hwa (Wrynaft)</h3>
            <p style="color: #888; margin-bottom: 15px;">
                <em>Data Science Student @ Universiti Malaya</em>
            </p>
            <p style="color: #ccc;">
                This project was developed as part of the <strong>WIH3001 Data Science Project (Final Year Project)</strong>
                course at the Universiti Malaya. It explores the application of Large Language Models
                (LLMs) and multi-agent systems in financial analysis and portfolio construction.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Contact/Links section
        st.markdown("### 🔗 Links")

        link_col1, link_col2, link_col3 = st.columns(3)

        with link_col1:
            st.markdown("""
            <a href="https://github.com/Wrynaft" target="_blank" style="text-decoration: none;">
                <div style="
                    background-color: #333;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    transition: transform 0.2s;
                ">
                    <span style="font-size: 24px;">🐙</span>
                    <p style="color: #ccc; margin: 5px 0 0 0; font-size: 14px;">GitHub</p>
                </div>
            </a>
            """, unsafe_allow_html=True)

        with link_col2:
            st.markdown("""
            <a href="https://www.linkedin.com/in/ryanchinjh/" target="_blank" style="text-decoration: none;">
                <div style="
                    background-color: #333;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                ">
                    <span style="font-size: 24px;">💼</span>
                    <p style="color: #ccc; margin: 5px 0 0 0; font-size: 14px;">LinkedIn</p>
                </div>
            </a>
            """, unsafe_allow_html=True)

        with link_col3:
            st.markdown("""
            <a href="mailto:ryanchinjh@gmail.com" style="text-decoration: none;">
                <div style="
                    background-color: #333;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                ">
                    <span style="font-size: 24px;">📧</span>
                    <p style="color: #ccc; margin: 5px 0 0 0; font-size: 14px;">Email</p>
                </div>
            </a>
            """, unsafe_allow_html=True)

# =============================================================================
# SPECIAL THANKS TAB
# =============================================================================
with tab3:
    st.markdown("## 🙏 Special Thanks")

    st.markdown("""
    <p style="color: #ccc; font-size: 16px; margin-bottom: 30px;">
        This project would not have been possible without the support and guidance of the following individuals and organizations.
    </p>
    """, unsafe_allow_html=True)

    # Academic Supervision
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #FFD700;
        margin-bottom: 20px;
    ">
        <h4 style="color: #FFD700; margin-top: 0;">🎓 Academic Supervision</h4>
        <ul style="color: #ccc; margin-bottom: 0;">
            <li><strong>Prof. Dr. Nor Liyana Bt Mohd Shuib</strong> - Project Supervisor, Universiti Malaya</li>
            <li><strong>Faculty of Computer Science and Information Technology</strong> - Universiti Malaya</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Technologies & Tools
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin-bottom: 20px;
    ">
        <h4 style="color: #2196F3; margin-top: 0;">🛠️ Technologies & Tools</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px;">
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">Streamlit</span>
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">LangChain</span>
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">LangGraph</span>
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">Google Gemini</span>
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">MongoDB</span>
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">FinBERT</span>
            <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">Python</span>
                <span style="background-color: #333; padding: 5px 15px; border-radius: 15px; color: #ccc;">BeautifulSoup</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Data Sources
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin-bottom: 20px;
    ">
        <h4 style="color: #4CAF50; margin-top: 0;">📊 Data Sources</h4>
        <ul style="color: #ccc; margin-bottom: 0;">
            <li><strong>Yahoo Finance</strong> - Historical price and company fundamental data</li>
            <li><strong>News APIs (KLSE Screener)</strong> - Market sentiment data</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Personal Acknowledgements
    st.markdown("""
    <div style="
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #E91E63;
        margin-bottom: 20px;
    ">
        <h4 style="color: #E91E63; margin-top: 0;">❤️ Personal Acknowledgements</h4>
        <ul style="color: #ccc; margin-bottom: 0;">
            <li>Family and friends for their continuous support and encouragement</li>
            <li>Classmates and peers for feedback and collaboration</li>
            <li>The open-source community for invaluable resources and tools</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Quote
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-top: 30px;
    ">
        <p style="color: #888; font-size: 18px; font-style: italic; margin-bottom: 10px;">
            "The best way to predict the future is to create it."
        </p>
        <p style="color: #666; font-size: 14px; margin: 0;">— Peter Drucker</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")

# Project info
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #888; font-size: 14px; margin: 0;">📅 Project Duration</p>
        <p style="color: #ccc; font-size: 16px; margin: 5px 0;">October 2025 - January 2026</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #888; font-size: 14px; margin: 0;">🏫 Institution</p>
        <p style="color: #ccc; font-size: 16px; margin: 5px 0;">Universiti Malaya</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center;">
        <p style="color: #888; font-size: 14px; margin: 0;">📚 Course</p>
        <p style="color: #ccc; font-size: 16px; margin: 5px 0;">WIH3001 Data Science Project</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
<div style="text-align: center; color: #666; font-size: 12px;">
    <p>Made with ❤️ using Streamlit | RoundtableAI © 2024</p>
</div>
""", unsafe_allow_html=True)
