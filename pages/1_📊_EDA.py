"""
EDA Page - Exploratory Data Analysis

This page contains exploratory data analysis of the stock data,
news sentiment data, and fundamental metrics used by RoundtableAI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from utils.database import get_mongo_collection

st.set_page_config(
    page_title="EDA - RoundtableAI",
    page_icon="📊",
    layout="wide"
)

# =============================================================================
# Data Loading Functions (with caching)
# =============================================================================

# Data cutoff date for consistency with backtesting
DATA_CUTOFF_DATE = datetime(2025, 12, 2, 23, 59, 59)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_fundamentals_data():
    """Load fundamentals data from MongoDB."""
    collection = get_mongo_collection('fundamentals')
    cursor = collection.find(
        {},
        {
            'ticker': 1,
            'company_name': 1,
            'sector': 1,
            'industry': 1,
            'metrics': 1,
            '_id': 0
        }
    )
    data = list(cursor)
    return pd.DataFrame(data)

@st.cache_data(ttl=3600)
def load_articles_data():
    """Load articles data from MongoDB (filtered to cutoff date)."""
    collection = get_mongo_collection('articles')
    cursor = collection.find(
        {
            'published': {'$lte': DATA_CUTOFF_DATE}
        },
        {
            'ticker': 1,
            'headline': 1,
            'published': 1,
            'source': 1,
            'sentiment': 1,
            '_id': 0
        }
    )
    data = list(cursor)
    df = pd.DataFrame(data)

    # Extract sentiment labels and scores
    if len(df) > 0:
        df['sentiment_label'] = df['sentiment'].apply(
            lambda x: x.get('label', 'Unknown') if isinstance(x, dict) else 'Unknown'
        )
        df['sentiment_score'] = df['sentiment'].apply(
            lambda x: x.get('score', None) if isinstance(x, dict) else None
        )

    return df

@st.cache_data(ttl=3600)
def extract_metrics_data(_fundamentals_df):
    """Extract key financial metrics from nested structure."""
    metrics_data = []

    for _, row in _fundamentals_df.iterrows():
        if 'metrics' in row and row['metrics']:
            metrics = row['metrics']
            valuation = metrics.get('valuation', {})
            financial_health = metrics.get('financial_health', {})
            growth = metrics.get('growth', {})

            metrics_data.append({
                'ticker': row.get('ticker'),
                'sector': row.get('sector'),
                # Valuation metrics
                'pe_ratio': valuation.get('pe_ratio'),
                'price_to_book': valuation.get('price_to_book'),
                'price_to_sales': valuation.get('price_to_sales'),
                # Financial health metrics
                'roe': financial_health.get('return_on_equity'),
                'roa': financial_health.get('return_on_assets'),
                'debt_to_equity': financial_health.get('debt_to_equity'),
                'current_ratio': financial_health.get('current_ratio'),
                'gross_margin': financial_health.get('gross_margins'),
                'operating_margin': financial_health.get('operating_margins'),
                'profit_margin': financial_health.get('profit_margins'),
                # Growth metrics
                'revenue_growth': growth.get('revenue_growth'),
                'earnings_growth': growth.get('earnings_growth')
            })

    return pd.DataFrame(metrics_data)

# =============================================================================
# Header
# =============================================================================

st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h1 style="color: #4CAF50;">📊 Exploratory Data Analysis</h1>
        <p style="color: #888; font-size: 18px;">
            Analyzing Bursa Malaysia stock data, news sentiment, and financial metrics
        </p>
        <p style="color: #666; font-size: 14px;">
            Data cutoff: December 2, 2025
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# Load Data
# =============================================================================

with st.spinner("Loading data from MongoDB..."):
    fundamentals_df = load_fundamentals_data()
    articles_df = load_articles_data()
    metrics_df = extract_metrics_data(fundamentals_df)

# =============================================================================
# Create Tabs
# =============================================================================

tab1, tab2, tab3 = st.tabs(["📈 Valuation", "📰 Sentiment", "💹 Fundamental"])

# =============================================================================
# TAB 1: VALUATION
# =============================================================================

with tab1:
    st.header("Valuation Analysis")

    st.markdown("""
    This section analyzes the composition of companies in our dataset across different sectors,
    providing insights into market diversity and sector representation on Bursa Malaysia.
    """)

    st.markdown("---")

    # Sector distribution
    st.subheader("Distribution of Companies by Sector")

    # Filter by sector
    sectors = ['All'] + sorted([s for s in fundamentals_df['sector'].unique() if s is not None])
    selected_sector = st.selectbox("Filter by Sector", sectors, key="sector_filter")

    # Apply filter
    if selected_sector != 'All':
        filtered_df = fundamentals_df[fundamentals_df['sector'] == selected_sector]
    else:
        filtered_df = fundamentals_df

    # Calculate sector counts
    sector_counts = filtered_df['sector'].value_counts()

    # Create visualization
    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(12, 8))
        colors = sns.color_palette('husl', len(sector_counts))
        bars = ax.barh(sector_counts.index, sector_counts.values, color=colors)

        # Add value labels
        for bar, count in zip(bars, sector_counts.values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{count}', ha='left', va='center', fontsize=10, fontweight='bold')

        ax.set_xlabel('Number of Companies', fontsize=12)
        ax.set_ylabel('Sector', fontsize=12)
        ax.set_title('Companies by Sector', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Key Statistics")
        st.metric("Total Companies", len(filtered_df))
        st.metric("Number of Sectors", fundamentals_df['sector'].nunique())
        st.metric("Number of Industries", fundamentals_df['industry'].nunique())

        if selected_sector != 'All':
            st.info(f"**Selected:** {selected_sector}")
            st.metric("Companies in Sector", len(filtered_df))

    # Insights
    st.markdown("---")
    st.markdown("### 📋 Insights")

    top_sector = sector_counts.index[0]
    top_count = sector_counts.values[0]
    total_companies = len(fundamentals_df)

    st.markdown(f"""
    - **Largest Sector**: {top_sector} with {top_count} companies ({top_count/total_companies*100:.1f}% of dataset)
    - **Sector Diversity**: {fundamentals_df['sector'].nunique()} distinct sectors represented
    - **Industry Granularity**: {fundamentals_df['industry'].nunique()} unique industries across all sectors
    - This diversity enables comprehensive sector-based comparative analysis by the Fundamental Agent
    """)

# =============================================================================
# TAB 2: SENTIMENT
# =============================================================================

with tab2:
    st.header("Sentiment Analysis")

    st.markdown("""
    This section analyzes news articles and their FinBERT sentiment scores, which the Sentiment Agent
    uses to gauge market sentiment and news coverage patterns for Bursa Malaysia stocks.
    """)

    st.markdown("---")

    # Date range filter
    if len(articles_df) > 0:
        articles_df['published'] = pd.to_datetime(articles_df['published'], errors='coerce')
        articles_with_dates = articles_df.dropna(subset=['published'])

        min_date = articles_with_dates['published'].min().date()
        max_date = articles_with_dates['published'].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", max_date, min_value=min_date, max_value=max_date)

        # Filter by date range
        mask = (articles_with_dates['published'].dt.date >= start_date) & (articles_with_dates['published'].dt.date <= end_date)
        filtered_articles = articles_with_dates[mask]
    else:
        filtered_articles = articles_df

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Articles", f"{len(filtered_articles):,}")
    with col2:
        st.metric("Companies Covered", filtered_articles['ticker'].nunique())
    with col3:
        st.metric("News Sources", filtered_articles['source'].nunique())
    with col4:
        avg_per_company = len(filtered_articles) / max(filtered_articles['ticker'].nunique(), 1)
        st.metric("Avg Articles/Company", f"{avg_per_company:.1f}")

    st.markdown("---")

    # 1. Sentiment Distribution
    st.subheader("Sentiment Distribution")

    sentiment_counts = filtered_articles['sentiment_label'].value_counts()

    col1, col2 = st.columns([3, 2])

    with col1:
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors_map = {'positive': '#4CAF50', 'negative': '#f44336', 'neutral': '#9E9E9E', 'Unknown': '#757575'}
        bar_colors = [colors_map.get(label, '#757575') for label in sentiment_counts.index]
        bars = ax.bar(sentiment_counts.index, sentiment_counts.values, color=bar_colors, edgecolor='black')

        for bar, count in zip(bars, sentiment_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Sentiment Label', fontsize=12)
        ax.set_ylabel('Number of Articles', fontsize=12)
        ax.set_title('Distribution of Sentiment Labels', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        st.pyplot(fig)
        plt.close()

    with col2:
        # Pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        pie_colors = [colors_map.get(label, '#757575') for label in sentiment_counts.index]
        wedges, texts, autotexts = ax.pie(
            sentiment_counts.values,
            labels=sentiment_counts.index,
            autopct='%1.1f%%',
            colors=pie_colors,
            explode=[0.05]*len(sentiment_counts),
            startangle=90,
            textprops={'fontsize': 12, 'weight': 'bold'}
        )

        for autotext in autotexts:
            autotext.set_color('white')

        ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold')

        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    # 2. Article Source Distribution
    st.subheader("Article Source Distribution")

    if 'source' in filtered_articles.columns:
        source_counts = filtered_articles['source'].value_counts().head(15)

        col1, col2 = st.columns([3, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(12, 8))
            colors = sns.color_palette('viridis', len(source_counts))
            bars = ax.barh(range(len(source_counts)), source_counts.values, color=colors)
            ax.set_yticks(range(len(source_counts)))
            ax.set_yticklabels(source_counts.index)
            ax.invert_yaxis()
            ax.set_xlabel('Number of Articles', fontsize=12)
            ax.set_title('Top 15 News Sources by Article Count', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)

            # Add value labels
            for bar, count in zip(bars, source_counts.values):
                ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                        f'{count}', ha='left', va='center', fontsize=9)

            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Top Sources")
            for i, (source, count) in enumerate(source_counts.head(5).items(), 1):
                pct = count / len(filtered_articles) * 100
                st.markdown(f"**{i}. {source}**")
                st.progress(pct/100)
                st.caption(f"{count} articles ({pct:.1f}%)")

    # Insights
    st.markdown("---")
    st.markdown("### 📋 Insights")

    if len(sentiment_counts) > 0:
        dominant_sentiment = sentiment_counts.index[0]
        dominant_pct = sentiment_counts.values[0] / len(filtered_articles) * 100

        positive_pct = sentiment_counts.get('positive', 0) / len(filtered_articles) * 100
        negative_pct = sentiment_counts.get('negative', 0) / len(filtered_articles) * 100

        st.markdown(f"""
        - **Overall Sentiment**: {dominant_sentiment.capitalize()} sentiment dominates ({dominant_pct:.1f}% of articles)
        - **Positive Coverage**: {positive_pct:.1f}% of articles have positive sentiment
        - **Negative Coverage**: {negative_pct:.1f}% of articles have negative sentiment
        - **News Coverage**: {filtered_articles['ticker'].nunique()} companies have news coverage
        - **Source Diversity**: {filtered_articles['source'].nunique()} unique news sources provide market coverage
        - The Sentiment Agent uses these FinBERT-scored articles to gauge market sentiment and news momentum
        """)

# =============================================================================
# TAB 3: FUNDAMENTAL
# =============================================================================

with tab3:
    st.header("Fundamental Metrics Analysis")

    st.markdown("""
    This section analyzes key financial metrics and their relationships, revealing patterns in
    profitability, valuation, and financial health that the Fundamental Agent uses for stock analysis.
    """)

    st.markdown("---")

    # Sector filter for fundamental analysis
    sectors_fund = ['All'] + sorted([s for s in metrics_df['sector'].unique() if pd.notna(s)])
    selected_sector_fund = st.selectbox("Filter by Sector", sectors_fund, key="sector_filter_fund")

    # Apply filter
    if selected_sector_fund != 'All':
        filtered_metrics = metrics_df[metrics_df['sector'] == selected_sector_fund]
    else:
        filtered_metrics = metrics_df

    # Correlation Heatmap
    st.subheader("Correlation Heatmap of Key Fundamental Metrics")

    st.markdown("""
    This heatmap reveals relationships between financial metrics. Strong correlations indicate metrics
    that tend to move together, while weak correlations suggest independent aspects of financial health.
    """)

    # Select numeric columns for correlation
    numeric_cols = ['pe_ratio', 'price_to_book', 'price_to_sales',
                    'roe', 'roa', 'debt_to_equity', 'current_ratio',
                    'gross_margin', 'operating_margin', 'profit_margin',
                    'revenue_growth', 'earnings_growth']

    # Convert to numeric and filter outliers
    metrics_filtered = filtered_metrics[numeric_cols].copy()
    for col in numeric_cols:
        metrics_filtered[col] = pd.to_numeric(metrics_filtered[col], errors='coerce')

    # Remove extreme outliers for better visualization (5th to 95th percentile)
    for col in numeric_cols:
        q05 = metrics_filtered[col].quantile(0.05)
        q95 = metrics_filtered[col].quantile(0.95)
        metrics_filtered[col] = metrics_filtered[col].clip(q05, q95)

    # Calculate correlation
    correlation_matrix = metrics_filtered.corr()

    # Display correlation heatmap
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

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
        cbar_kws={'shrink': 0.8},
        vmin=-1,
        vmax=1
    )

    ax.set_title('Correlation Heatmap of Fundamental Metrics', fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    # Display key metrics statistics
    st.subheader("Key Metrics Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        median_pe = pd.to_numeric(filtered_metrics['pe_ratio'], errors='coerce').median()
        st.metric("Median P/E Ratio", f"{median_pe:.2f}" if pd.notna(median_pe) else "N/A")

        median_pb = pd.to_numeric(filtered_metrics['price_to_book'], errors='coerce').median()
        st.metric("Median P/B Ratio", f"{median_pb:.2f}" if pd.notna(median_pb) else "N/A")

    with col2:
        median_roe = pd.to_numeric(filtered_metrics['roe'], errors='coerce').median()
        st.metric("Median ROE", f"{median_roe*100:.2f}%" if pd.notna(median_roe) else "N/A")

        median_roa = pd.to_numeric(filtered_metrics['roa'], errors='coerce').median()
        st.metric("Median ROA", f"{median_roa*100:.2f}%" if pd.notna(median_roa) else "N/A")

    with col3:
        median_de = pd.to_numeric(filtered_metrics['debt_to_equity'], errors='coerce').median()
        st.metric("Median D/E Ratio", f"{median_de:.2f}" if pd.notna(median_de) else "N/A")

        median_cr = pd.to_numeric(filtered_metrics['current_ratio'], errors='coerce').median()
        st.metric("Median Current Ratio", f"{median_cr:.2f}" if pd.notna(median_cr) else "N/A")

    with col4:
        median_pm = pd.to_numeric(filtered_metrics['profit_margin'], errors='coerce').median()
        st.metric("Median Profit Margin", f"{median_pm*100:.2f}%" if pd.notna(median_pm) else "N/A")

        median_rg = pd.to_numeric(filtered_metrics['revenue_growth'], errors='coerce').median()
        st.metric("Median Revenue Growth", f"{median_rg*100:.2f}%" if pd.notna(median_rg) else "N/A")

    # Insights
    st.markdown("---")
    st.markdown("### 📋 Insights")

    # Find strongest correlations (excluding diagonal)
    corr_unstacked = correlation_matrix.where(~mask).unstack()
    top_correlations = corr_unstacked.abs().nlargest(6)[1:]  # Skip the highest (always 1.0)

    st.markdown("""
    **Key Correlation Patterns:**
    """)

    for (metric1, metric2), corr_value in top_correlations.items():
        actual_value = correlation_matrix.loc[metric1, metric2]
        direction = "positive" if actual_value > 0 else "negative"
        st.markdown(f"- **{metric1}** vs **{metric2}**: {direction} correlation ({actual_value:.2f})")

    st.markdown(f"""

    **Financial Health Overview** ({selected_sector_fund} Sector):
    - These metrics help the Fundamental Agent assess company financial health comprehensively
    - Strong correlations between profitability metrics (gross/operating/profit margins) indicate consistent operational efficiency
    - The heatmap reveals how valuation, profitability, and leverage interact in {selected_sector_fund if selected_sector_fund != 'All' else 'the market'}
    - Companies with {len(filtered_metrics)} records provide robust statistical foundation for analysis
    """)

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p>Data sourced from MongoDB | Updated: {}</p>
    <p style="font-size: 12px;">All time-series data filtered to December 2, 2025 for consistency</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M")), unsafe_allow_html=True)
