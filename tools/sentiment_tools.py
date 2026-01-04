"""
Sentiment analysis tools for extracting and analyzing news article sentiment.

This module provides tools for:
- Fetching recent news articles for a ticker
- Retrieving precomputed FinBERT sentiment scores
- Aggregating sentiment data over time windows
"""
from datetime import datetime, timedelta
from pymongo import MongoClient
from urllib.parse import quote_plus
from langchain.tools import tool
from .config import get_reference_date


def normalize_ticker(ticker: str) -> str:
    """Remove .KL suffix from ticker for articles collection query."""
    return ticker.upper().replace(".KL", "")


@tool
def get_recent_articles(ticker: str, days: int = 7) -> dict:
    """
    Fetches recent news articles for a given ticker symbol from MongoDB.

    Retrieves articles published within the specified lookback window,
    sorted by publication date (most recent first).

    Args:
        ticker: Stock ticker symbol (e.g., "1155.KL" or "1155")
        days: Number of days to look back for articles (default: 7)

    Returns:
        Dictionary containing:
        - success: Whether the operation was successful
        - ticker: The ticker symbol queried
        - lookback_days: Number of days in the lookback window
        - article_count: Number of articles found
        - articles: List of article dictionaries with:
            - headline: Article headline
            - published: Publication date
            - source: News source
            - body: Article body text
    """
    # Connect to MongoDB
    username = quote_plus("Wrynaft")
    password = quote_plus("Ryan@120104")

    # Normalize ticker (remove .KL suffix for articles collection)
    normalized_ticker = normalize_ticker(ticker)

    try:
        client = MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0"
        )
        db = client['roundtable_ai']
        print("Connected to MongoDB")

        col = db["articles"]

        # Use configurable reference date
        current_date = get_reference_date()
        lookback_date = current_date - timedelta(days=days)

        # Query for articles within the time window
        # Use datetime object since MongoDB stores dates as datetime
        query = {
            "ticker": normalized_ticker,
            "published": {"$gte": lookback_date}
        }

        cursor = col.find(query).sort("published", -1)

        # Extract article data
        articles = []
        for doc in cursor:
            articles.append({
                "headline": doc.get("headline", ""),
                "published": doc.get("published", ""),
                "source": doc.get("source", ""),
                "body": doc.get("body", "")
            })

        result = {
            "success": True,
            "ticker": ticker,
            "lookback_days": days,
            "article_count": len(articles),
            "articles": articles
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@tool
def get_article_sentiment(ticker: str, days: int = 7) -> dict:
    """
    Fetches precomputed sentiment scores using FinBERT for recent news articles.

    Retrieves FinBERT sentiment analysis results for articles published
    within the specified lookback window. FinBERT is a pre-trained NLP model
    specialized for financial sentiment analysis.

    Args:
        ticker: Stock ticker symbol (e.g., "1155.KL" or "1155")
        days: Number of days to look back for articles (default: 7)

    Returns:
        Dictionary containing:
        - success: Whether the operation was successful
        - ticker: The ticker symbol queried
        - lookback_days: Number of days in the lookback window
        - returned_articles: Number of articles with sentiment data
        - sentiments: Dictionary mapping article IDs to sentiment scores
                     Each sentiment contains FinBERT classification scores
    """
    # Connect to MongoDB
    username = quote_plus("Wrynaft")
    password = quote_plus("Ryan@120104")

    # Normalize ticker (remove .KL suffix for articles collection)
    normalized_ticker = normalize_ticker(ticker)

    try:
        client = MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0"
        )
        db = client['roundtable_ai']
        print("Connected to MongoDB")

        col = db["articles"]

        # Use configurable reference date
        current_date = get_reference_date()
        lookback_date = current_date - timedelta(days=days)

        # Query for articles with sentiment data
        # Use datetime object since MongoDB stores dates as datetime
        query = {
            "ticker": normalized_ticker,
            "published": {"$gte": lookback_date},
            "sentiment": {"$exists": True}
        }

        cursor = col.find(
            query,
            {"_id": 1, "sentiment": 1}
        ).sort("published", -1)

        # Extract sentiment data
        sentiment_data = {}
        for doc in cursor:
            sentiment_data[str(doc["_id"])] = doc["sentiment"]

        result = {
            "success": True,
            "ticker": ticker,
            "lookback_days": days,
            "returned_articles": len(sentiment_data),
            "sentiments": sentiment_data
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
