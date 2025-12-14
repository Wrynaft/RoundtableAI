"""
Ticker resolution utility for converting company names to ticker symbols.
"""
import os
from pathlib import Path
import pandas as pd
import yfinance as yf
from rapidfuzz import process, fuzz
from functools import lru_cache
from langchain.tools import tool

# Get the project root directory (parent of utils/)
PROJECT_ROOT = Path(__file__).parent.parent
TICKER_LIST_PATH = PROJECT_ROOT / "ticker_list.csv"

# Common aliases for Malaysian companies
# Maps popular nicknames/short names to official company names
COMPANY_ALIASES = {
    # Banks
    "maybank": "MALAYAN BANKING BERHAD",
    "malayan banking": "MALAYAN BANKING BERHAD",
    "cimb": "CIMB GROUP HOLDINGS BERHAD",
    "public bank": "PUBLIC BANK BERHAD",
    "pbbank": "PUBLIC BANK BERHAD",
    "rhb": "RHB BANK BERHAD",
    "rhb bank": "RHB BANK BERHAD",
    "hong leong bank": "HONG LEONG BANK BERHAD",
    "hlbank": "HONG LEONG BANK BERHAD",
    "ambank": "AMMB HOLDINGS BERHAD",
    "affin": "AFFIN BANK BERHAD",

    # Telcos
    "celcom": "AXIATA GROUP BERHAD",
    "axiata": "AXIATA GROUP BERHAD",
    "maxis": "MAXIS BERHAD",
    "digi": "CK HUTCHISON HOLDINGS LIMITED",
    "tm": "TELEKOM MALAYSIA BERHAD",
    "telekom": "TELEKOM MALAYSIA BERHAD",
    "telekom malaysia": "TELEKOM MALAYSIA BERHAD",

    # Energy/Utilities
    "tenaga": "TENAGA NASIONAL BERHAD",
    "tnb": "TENAGA NASIONAL BERHAD",
    "tenaga nasional": "TENAGA NASIONAL BERHAD",
    "petronas gas": "PETRONAS GAS BERHAD",
    "petgas": "PETRONAS GAS BERHAD",
    "petronas chemicals": "PETRONAS CHEMICALS GROUP BERHAD",
    "pchem": "PETRONAS CHEMICALS GROUP BERHAD",
    "petronas dagangan": "PETRONAS DAGANGAN BERHAD",

    # Plantations
    "sime darby": "SIME DARBY BERHAD",
    "sime": "SIME DARBY BERHAD",
    "ioi": "IOI CORPORATION BERHAD",
    "ioi corp": "IOI CORPORATION BERHAD",
    "klk": "KUALA LUMPUR KEPONG BERHAD",
    "kuala lumpur kepong": "KUALA LUMPUR KEPONG BERHAD",
    "genting": "GENTING BERHAD",
    "genting malaysia": "GENTING MALAYSIA BERHAD",
    "genm": "GENTING MALAYSIA BERHAD",

    # Others
    "nestle": "NESTLE (MALAYSIA) BERHAD",
    "nestle malaysia": "NESTLE (MALAYSIA) BERHAD",
    "dutch lady": "DUTCH LADY MILK INDUSTRIES BERHAD",
    "top glove": "TOP GLOVE CORPORATION BERHAD",
    "topglove": "TOP GLOVE CORPORATION BERHAD",
    "hartalega": "HARTALEGA HOLDINGS BERHAD",
    "press metal": "PRESS METAL ALUMINIUM HOLDINGS BERHAD",
    "pmetal": "PRESS METAL ALUMINIUM HOLDINGS BERHAD",
    "yinson": "YINSON HOLDINGS BERHAD",
    "airasia": "CAPITAL A BERHAD",
    "capital a": "CAPITAL A BERHAD",
    "mr diy": "MR D.I.Y. GROUP (M) BERHAD",
    "mrdiy": "MR D.I.Y. GROUP (M) BERHAD",
}


@lru_cache(maxsize=1)
def load_company_mapping() -> pd.DataFrame:
    """
    Load and cache company name to ticker mapping from CSV.

    Returns:
        DataFrame with normalized company and ticker columns
    """
    df = pd.read_csv(TICKER_LIST_PATH)
    df['company_normalized'] = df['company_name'].str.lower().str.strip()
    df['ticker_normalized'] = df['ticker'].str.upper().str.strip()
    return df


@tool
def resolve_ticker_symbol(company_name: str) -> dict:
    """
    Resolves a company name or partial name to its ticker symbol using multiple methods:
    0. Alias lookup (common nicknames like "Maybank" -> "Malayan Banking Berhad")
    1. Exact match on company name
    2. Direct ticker lookup
    3. Partial name matching
    4. Fuzzy matching (70% threshold)
    5. Yahoo Finance fallback

    Args:
        company_name: Company name or ticker symbol to resolve

    Returns:
        Dictionary with resolution results including:
        - success: Whether resolution was successful
        - ticker: Resolved ticker symbol
        - company_name: Full company name
        - resolution_method: Method used for resolution
        - confidence: Match confidence score (for fuzzy matching)
        - error: Error message (if unsuccessful)
        - suggestions: Helpful suggestions (if unsuccessful)
    """
    df = load_company_mapping()
    query_norm = company_name.lower().strip()

    # Method 0: Alias lookup (common nicknames)
    if query_norm in COMPANY_ALIASES:
        official_name = COMPANY_ALIASES[query_norm]
        alias_match = df[df['company_name'].str.upper() == official_name.upper()]
        if not alias_match.empty:
            row = alias_match.iloc[0]
            return {
                "success": True,
                "query": company_name,
                "ticker": row['ticker_normalized'],
                "company_name": row['company_name'],
                "resolution_method": "alias_match",
                "alias_used": query_norm
            }

    # Method 1: Exact match on company name
    exact_match = df[df['company_normalized'] == query_norm]
    if not exact_match.empty:
        row = exact_match.iloc[0]
        return {
            "success": True,
            "query": company_name,
            "ticker": row['ticker_normalized'],
            "company_name": row['company_name'],
            "resolution_method": "exact_match"
        }

    # Method 2: Direct ticker lookup
    ticker_match = df[df['ticker_normalized'] == company_name.upper().strip()]
    if not ticker_match.empty:
        row = ticker_match.iloc[0]
        return {
            "success": True,
            "query": company_name,
            "ticker": row['ticker_normalized'],
            "company_name": row['company_name'],
            "resolution_method": "ticker_match"
        }

    # Method 3: Partial name matching
    partial_matches = df[df['company_normalized'].str.contains(query_norm, na=False)]
    if len(partial_matches) == 1:
        row = partial_matches.iloc[0]
        return {
            "success": True,
            "query": company_name,
            "ticker": row['ticker_normalized'],
            "company_name": row['company_name'],
            "resolution_method": "partial_match"
        }
    elif len(partial_matches) > 1:
        return {
            "success": False,
            "resolution_method": "multiple_partial_matches",
            "query": company_name,
            "candidates": [
                {
                    "ticker": row['ticker_normalized'],
                    "company_name": row['company_name']
                } for _, row in partial_matches.iterrows()
            ]
        }

    # Method 4: Fuzzy matching
    all_companies = df['company_name'].tolist()
    match_result = process.extractOne(company_name, all_companies, scorer=fuzz.WRatio, score_cutoff=70)

    if match_result:
        best_match, score, idx = match_result
        row = df.iloc[idx]
        return {
            "success": True,
            "query": company_name,
            "ticker": row['ticker_normalized'],
            "company_name": row['company_name'],
            "resolution_method": "fuzzy_match",
            "confidence": score
        }

    # Method 5: Yahoo Finance fallback
    try:
        potential_ticker = company_name.upper().strip()
        info = yf.Ticker(potential_ticker).info
        if info and 'symbol' in info:
            return {
                "success": True,
                "query": company_name,
                "ticker": potential_ticker,
                "company_name": info.get('longName', 'Unknown'),
                "resolution_method": "yfinance_lookup"
            }
    except Exception:
        pass

    # Resolution failed
    return {
        "success": False,
        "query": company_name,
        "error": f"Could not resolve '{company_name}' to ticker symbol.",
        "suggestions": [
            "Try using the stock ticker directly (e.g. 1155.KL for Maybank)",
            "Check spelling of the company name",
            "Use the full official company name like 'Malayan Banking Berhad'"
        ]
    }
