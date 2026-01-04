"""
Valuation analysis tools for stock risk and return metrics.

This module provides tools for:
- Fetching historical stock price data from MongoDB
- Analyzing risk metrics (volatility, Sharpe ratio, VaR, drawdown)
- Computing return metrics and volume statistics
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from langchain.tools import tool
from urllib.parse import quote_plus
from pymongo import MongoClient
from .config import get_reference_date


@tool
def get_stock_data(ticker_symbol: str, period: str = "1y") -> str:
    """
    Fetches historical stock data for the given ticker symbol from MongoDB.

    Args:
        ticker_symbol: Stock ticker symbol (e.g., "1155.KL")
        period: Time period for data retrieval. Options:
                "1mo" (1 month), "3mo" (3 months), "6mo" (6 months),
                "1y" (1 year, default), "5y" (5 years)

    Returns:
        CSV string of the stock data with columns:
        date, close, volume (optimized for token efficiency)
    """
    try:
        # Use configurable reference date
        current_date = get_reference_date()

        # Connect to MongoDB
        username = quote_plus("Wrynaft")
        password = quote_plus("Ryan@120104")
        client = MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0"
        )
        db = client['roundtable_ai']
        col = db['stock_prices']

        # Map period strings to days
        period_map = {
            "1y": 365,
            "6mo": 180,
            "3mo": 90,
            "1mo": 30,
            "5y": 365 * 5
        }

        days = period_map.get(period, 365)  # default to 1 year
        start_date = current_date - timedelta(days=days)
        # Convert to string format for MongoDB query (dates stored as strings)
        start_date_str = start_date.strftime("%Y-%m-%d")

        # Query MongoDB - only fetch close and volume for token efficiency
        cursor = col.find(
            {
                "ticker": ticker_symbol,
                "date": {"$gte": start_date_str}
            },
            {
                "_id": 0,
                "date": 1,
                "close": 1,
                "volume": 1
            }
        ).sort("date", 1)

        df = pd.DataFrame(list(cursor))

        if df.empty:
            return f"No data found for ticker symbol: {ticker_symbol}"

        # Dates are already strings in MongoDB, no conversion needed
        return df.to_csv(index=False)

    except Exception as e:
        return f"Error fetching stock data for {ticker_symbol}: {str(e)}"


@tool
def analyze_stock_metrics(ticker_symbol: str, period: str = "1y", risk_free_rate: float = 0.05) -> dict:
    """
    Analyzes stock metrics including volatility, risk metrics, and returns.

    Fetches data directly from MongoDB and calculates comprehensive risk and return metrics including:
    - Annualized returns and volatility
    - Sharpe ratio
    - Maximum drawdown
    - Value at Risk (VaR) at 5% and 1% confidence levels
    - Distribution metrics (skewness, kurtosis)
    - Volume statistics

    Args:
        ticker_symbol: Stock ticker symbol (e.g., "1155.KL")
        period: Time period for analysis - "1mo", "3mo", "6mo", "1y" (default), "5y"
        risk_free_rate: Risk-free rate for Sharpe ratio calculation (default: 0.05)

    Returns:
        Dictionary containing:
        - success: Whether analysis was successful
        - symbol: Stock ticker
        - analysis_period: Start/end dates and trading days
        - price_metrics: Returns and price performance
        - volatility_metrics: Daily and annualized volatility
        - risk_metrics: Sharpe ratio, drawdown, VaR
        - distribution_metrics: Skewness, kurtosis, positive/negative days
        - volume_metrics: Average volume and volatility
    """
    try:
        # Use configurable reference date
        current_date = get_reference_date()

        # Connect to MongoDB
        username = quote_plus("Wrynaft")
        password = quote_plus("Ryan@120104")
        client = MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0",
            tls=True,
            tlsAllowInvalidCertificates=True
        )
        db = client['roundtable_ai']
        col = db['stock_prices']

        # Map period strings to days
        period_map = {
            "1y": 365,
            "6mo": 180,
            "3mo": 90,
            "1mo": 30,
            "5y": 365 * 5
        }

        days = period_map.get(period, 365)
        start_date = current_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%d")

        # Query MongoDB
        cursor = col.find(
            {
                "ticker": ticker_symbol,
                "date": {"$gte": start_date_str}
            },
            {
                "_id": 0,
                "date": 1,
                "close": 1,
                "volume": 1
            }
        ).sort("date", 1)

        df = pd.DataFrame(list(cursor))

        if df.empty:
            return {
                "success": False,
                "error": f"No data found for ticker symbol: {ticker_symbol}"
            }

        # Set date as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        # Calculate daily returns
        daily_returns = df["close"].pct_change().dropna()

        if len(daily_returns) < 2:
            return {
                "success": False,
                "error": "Insufficient data for calculations",
            }

        mean_daily_return = daily_returns.mean()
        daily_volatility = daily_returns.std()

        # Calculate cumulative return for proper annualized return
        start_price = df["close"].iloc[0]
        end_price = df["close"].iloc[-1]
        cumulative_return = end_price / start_price - 1
        trading_days = len(df)

        # Annualized metrics
        annualized_return = (1 + cumulative_return) ** (252 / trading_days) - 1
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Sharpe ratio
        sharpe_ratio = (
            (annualized_return - risk_free_rate) / annualized_volatility
            if annualized_volatility > 0
            else 0
        )

        # Maximum drawdown calculation
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Value at Risk (VaR) - 5% and 1%
        var_5 = np.percentile(daily_returns, 5)
        var_1 = np.percentile(daily_returns, 1)

        # Additional statistics
        skewness = daily_returns.skew()
        kurtosis = daily_returns.kurtosis()

        # Price performance metrics
        total_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

        result = {
            "success": True,
            "symbol": ticker_symbol.upper(),
            "analysis_period": {
                "start_date": df.index[0].strftime("%Y-%m-%d"),
                "end_date": df.index[-1].strftime("%Y-%m-%d"),
                "trading_days": len(df)
            },
            "price_metrics": {
                "start_price": float(df['close'].iloc[0]),
                "end_price": float(df['close'].iloc[-1]),
                "total_return": float(total_return),
                "annualized_return": float(annualized_return)
            },
            "volatility_metrics": {
                "daily_volatility": float(daily_volatility),
                "annualized_volatility": float(annualized_volatility),
                "volatility_percentage": float(annualized_volatility * 100)
            },
            "risk_metrics": {
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "max_drawdown_percentage": float(max_drawdown * 100),
                "var_5_percent": float(var_5),
                "var_1_percent": float(var_1),
                "risk_free_rate": float(risk_free_rate)
            },
            "distribution_metrics": {
                "mean_daily_return": float(mean_daily_return),
                "skewness": float(skewness),
                "kurtosis": float(kurtosis),
                "positive_days": int((daily_returns > 0).sum()),
                "negative_days": int((daily_returns < 0).sum())
            },
            "volume_metrics": {
                "average_volume": float(df['volume'].mean()),
                "volume_volatility": float(df['volume'].std()),
                "latest_volume": float(df['volume'].iloc[-1])
            }
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "error": f"Error analyzing stock metrics: {str(e)}"
        }
