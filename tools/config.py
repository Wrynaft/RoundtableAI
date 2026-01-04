"""
Configuration module for tools.

This module provides a configurable reference date for backtesting purposes.
By default, tools use a fixed date, but this can be overridden for backtesting.
"""
from datetime import datetime
from typing import Optional

# Default reference date (used in production/normal operation)
_DEFAULT_DATE = datetime(2025, 12, 2)

# Configurable reference date (None means use default)
_reference_date: Optional[datetime] = None


def set_reference_date(date: Optional[datetime]) -> None:
    """
    Set the reference date for all tools.

    Args:
        date: The reference date to use, or None to reset to default.

    Example:
        # For backtesting from September 1, 2025
        from tools.config import set_reference_date
        set_reference_date(datetime(2025, 9, 1))

        # Reset to default
        set_reference_date(None)
    """
    global _reference_date
    _reference_date = date
    if date:
        print(f"[Tools Config] Reference date set to: {date.strftime('%Y-%m-%d')}")
    else:
        print(f"[Tools Config] Reference date reset to default: {_DEFAULT_DATE.strftime('%Y-%m-%d')}")


def get_reference_date() -> datetime:
    """
    Get the current reference date.

    Returns:
        The configured reference date, or the default if not set.
    """
    return _reference_date if _reference_date is not None else _DEFAULT_DATE


def get_default_date() -> datetime:
    """
    Get the default reference date.

    Returns:
        The default date (2025-12-02).
    """
    return _DEFAULT_DATE
