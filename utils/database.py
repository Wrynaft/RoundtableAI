"""
Database module for MongoDB connection and collection access.
"""
from pymongo import MongoClient
from urllib.parse import quote_plus
from typing import Optional

# Singleton client instance
_client: Optional[MongoClient] = None
_db = None


def get_mongo_client() -> MongoClient:
    """
    Returns singleton MongoDB client.

    Returns:
        MongoClient instance
    """
    global _client
    if _client is None:
        username = quote_plus("Wrynaft")
        password = quote_plus("Ryan@120104")
        _client = MongoClient(
            f"mongodb+srv://{username}:{password}@cluster0.bjjt9fa.mongodb.net/?appName=Cluster0",
            tls=True,
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000,
            connectTimeoutMS=30000,
            socketTimeoutMS=30000,
            retryWrites=True
        )
    return _client


def get_mongo_database(db_name: str = "roundtable_ai"):
    """
    Returns MongoDB database.

    Args:
        db_name: Name of the database (default: 'roundtable_ai')

    Returns:
        Database instance
    """
    global _db
    if _db is None:
        client = get_mongo_client()
        _db = client[db_name]
    return _db


def get_mongo_collection(collection_name: str, db_name: str = "roundtable_ai"):
    """
    Returns MongoDB collection.

    Args:
        collection_name: Name of collection (stock_prices, fundamentals, articles)
        db_name: Name of the database (default: 'roundtable_ai')

    Returns:
        Collection instance
    """
    db = get_mongo_database(db_name)
    return db[collection_name]


def close_mongo_connection():
    """Close MongoDB connection."""
    global _client, _db
    if _client is not None:
        _client.close()
        _client = None
        _db = None
