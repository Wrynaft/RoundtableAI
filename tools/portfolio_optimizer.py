"""
Portfolio Optimization Tools for Multi-Stock Analysis.

This module provides tools for:
- Portfolio weight allocation (equal-weight, confidence-weighted)
- Correlation matrix calculation
- Portfolio-level metrics (return, volatility, Sharpe ratio)
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from urllib.parse import quote_plus
from pymongo import MongoClient


class PortfolioOptimizer:
    """
    Portfolio optimizer for calculating allocations and metrics.

    Supports:
    - Equal-weight allocation
    - Confidence-weighted allocation (based on agent recommendations)
    - Portfolio metrics calculation
    - Correlation analysis
    """

    def __init__(self, risk_free_rate: float = 0.03):
        """
        Initialize the portfolio optimizer.

        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation.
                           Default is 3% (Malaysia approximate rate).
        """
        self.risk_free_rate = risk_free_rate
        self._client = None
        self._db = None

    def _get_db_connection(self):
        """Get MongoDB connection (lazy initialization)."""
        if self._client is None:
            username = quote_plus("Wrynaft")
            password = quote_plus("Ryan@120104")
            self._client = MongoClient(
                f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0"
            )
            self._db = self._client['roundtable_ai']
        return self._db

    def get_returns_dataframe(
        self,
        tickers: List[str],
        period: str = "1y"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch historical returns for multiple tickers.

        Args:
            tickers: List of ticker symbols (e.g., ["1155.KL", "1295.KL"])
            period: Time period - "1mo", "3mo", "6mo", "1y" (default), "5y"

        Returns:
            Tuple of (prices_df, returns_df) with tickers as columns
        """
        db = self._get_db_connection()
        col = db['stock_prices']

        # Map period to days
        period_map = {
            "1y": 365,
            "6mo": 180,
            "3mo": 90,
            "1mo": 30,
            "5y": 365 * 5
        }

        days = period_map.get(period, 365)
        current_date = datetime(2025, 12, 2)  # Fixed date as per original implementation
        start_date = current_date - timedelta(days=days)
        start_date_str = start_date.strftime("%Y-%m-%d")

        prices_dict = {}

        for ticker in tickers:
            cursor = col.find(
                {
                    "ticker": ticker,
                    "date": {"$gte": start_date_str}
                },
                {"_id": 0, "date": 1, "close": 1}
            ).sort("date", 1)

            df = pd.DataFrame(list(cursor))
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                prices_dict[ticker] = df['close']

        if not prices_dict:
            return pd.DataFrame(), pd.DataFrame()

        # Combine into single DataFrame
        prices_df = pd.DataFrame(prices_dict)

        # Forward fill missing values, then backward fill
        prices_df = prices_df.ffill().bfill()

        # Calculate daily returns
        returns_df = prices_df.pct_change().dropna()

        return prices_df, returns_df

    def equal_weight(self, tickers: List[str]) -> Dict[str, float]:
        """
        Calculate equal-weight allocation.

        Args:
            tickers: List of ticker symbols

        Returns:
            Dictionary mapping ticker to weight (all equal)
        """
        if not tickers:
            return {}

        weight = 1.0 / len(tickers)
        return {ticker: weight for ticker in tickers}

    def confidence_weighted(
        self,
        recommendations: Dict[str, Dict]
    ) -> Dict[str, float]:
        """
        Calculate weights based on agent confidence scores.

        Higher confidence recommendations get higher weights.
        SELL recommendations get zero weight.

        Args:
            recommendations: Dictionary mapping ticker to recommendation info
                           e.g., {"1155.KL": {"recommendation": "BUY", "confidence": 0.82}}

        Returns:
            Dictionary mapping ticker to weight
        """
        if not recommendations:
            return {}

        # Filter out SELL recommendations (zero weight)
        eligible = {
            ticker: info for ticker, info in recommendations.items()
            if info.get('recommendation', '').upper() != 'SELL'
        }

        if not eligible:
            # If all are SELL, return equal weight anyway
            return self.equal_weight(list(recommendations.keys()))

        # Calculate weights based on confidence
        total_confidence = sum(
            info.get('confidence', 0.5) for info in eligible.values()
        )

        if total_confidence == 0:
            return self.equal_weight(list(eligible.keys()))

        weights = {}
        for ticker, info in eligible.items():
            conf = info.get('confidence', 0.5)
            weights[ticker] = conf / total_confidence

        # SELL recommendations get 0 weight
        for ticker in recommendations:
            if ticker not in weights:
                weights[ticker] = 0.0

        return weights

    def calculate_correlation_matrix(
        self,
        tickers: List[str],
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for portfolio stocks.

        Args:
            tickers: List of ticker symbols
            period: Time period for correlation calculation

        Returns:
            Correlation matrix as DataFrame
        """
        _, returns_df = self.get_returns_dataframe(tickers, period)

        if returns_df.empty:
            return pd.DataFrame()

        return returns_df.corr()

    def calculate_portfolio_metrics(
        self,
        tickers: List[str],
        weights: Dict[str, float],
        period: str = "1y"
    ) -> Dict:
        """
        Calculate portfolio-level metrics.

        Args:
            tickers: List of ticker symbols
            weights: Dictionary mapping ticker to weight
            period: Time period for analysis

        Returns:
            Dictionary containing portfolio metrics:
            - expected_return: Annualized expected return
            - volatility: Annualized portfolio volatility
            - sharpe_ratio: Risk-adjusted return
            - correlation_avg: Average pairwise correlation
            - diversification_ratio: Measure of diversification benefit
        """
        prices_df, returns_df = self.get_returns_dataframe(tickers, period)

        if returns_df.empty:
            return {
                "success": False,
                "error": "No data available for portfolio calculation"
            }

        # Ensure all tickers have weights
        weights_array = np.array([weights.get(t, 0) for t in returns_df.columns])

        # Normalize weights to sum to 1
        if weights_array.sum() > 0:
            weights_array = weights_array / weights_array.sum()
        else:
            weights_array = np.ones(len(tickers)) / len(tickers)

        # Expected returns (annualized)
        mean_returns = returns_df.mean() * 252
        portfolio_return = np.dot(weights_array, mean_returns)

        # Covariance matrix (annualized)
        cov_matrix = returns_df.cov() * 252

        # Portfolio volatility
        portfolio_variance = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
        portfolio_volatility = np.sqrt(portfolio_variance)

        # Sharpe ratio
        sharpe_ratio = (
            (portfolio_return - self.risk_free_rate) / portfolio_volatility
            if portfolio_volatility > 0 else 0
        )

        # Correlation analysis
        corr_matrix = returns_df.corr()

        # Average correlation (excluding diagonal)
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_correlation = corr_matrix.values[mask].mean() if mask.sum() > 0 else 0

        # Individual volatilities for diversification ratio
        individual_volatilities = returns_df.std() * np.sqrt(252)
        weighted_avg_volatility = np.dot(weights_array, individual_volatilities)

        # Diversification ratio (higher is better)
        diversification_ratio = (
            weighted_avg_volatility / portfolio_volatility
            if portfolio_volatility > 0 else 1
        )

        # Individual stock metrics
        individual_metrics = {}
        for ticker in returns_df.columns:
            stock_returns = returns_df[ticker]
            individual_metrics[ticker] = {
                "expected_return": float(stock_returns.mean() * 252),
                "volatility": float(stock_returns.std() * np.sqrt(252)),
                "weight": float(weights.get(ticker, 0))
            }

        return {
            "success": True,
            "portfolio_metrics": {
                "expected_return": float(portfolio_return),
                "expected_return_pct": float(portfolio_return * 100),
                "volatility": float(portfolio_volatility),
                "volatility_pct": float(portfolio_volatility * 100),
                "sharpe_ratio": float(sharpe_ratio),
                "risk_free_rate": float(self.risk_free_rate)
            },
            "diversification_metrics": {
                "average_correlation": float(avg_correlation),
                "diversification_ratio": float(diversification_ratio),
                "number_of_stocks": len(tickers)
            },
            "individual_stocks": individual_metrics,
            "correlation_matrix": corr_matrix.to_dict(),
            "weights": {t: float(w) for t, w in zip(returns_df.columns, weights_array)}
        }

    def get_portfolio_summary(
        self,
        tickers: List[str],
        recommendations: Dict[str, Dict],
        allocation_method: str = "equal"
    ) -> Dict:
        """
        Get comprehensive portfolio summary.

        Args:
            tickers: List of ticker symbols
            recommendations: Dictionary of agent recommendations per ticker
            allocation_method: "equal" or "confidence"

        Returns:
            Complete portfolio analysis including weights and metrics
        """
        if not tickers:
            return {"success": False, "error": "No tickers provided"}

        # Calculate weights based on method
        if allocation_method == "confidence" and recommendations:
            weights = self.confidence_weighted(recommendations)
        else:
            weights = self.equal_weight(tickers)

        # Get portfolio metrics
        metrics = self.calculate_portfolio_metrics(tickers, weights)

        if not metrics.get("success", False):
            return metrics

        # Add recommendations to summary
        metrics["recommendations"] = recommendations
        metrics["allocation_method"] = allocation_method

        return metrics


# Convenience functions for direct use
def calculate_equal_weight_portfolio(tickers: List[str]) -> Dict:
    """Calculate equal-weight portfolio metrics."""
    optimizer = PortfolioOptimizer()
    weights = optimizer.equal_weight(tickers)
    return optimizer.calculate_portfolio_metrics(tickers, weights)


def calculate_correlation(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """Calculate correlation matrix for given tickers."""
    optimizer = PortfolioOptimizer()
    return optimizer.calculate_correlation_matrix(tickers, period)
