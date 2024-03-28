"""
Exceptions for ahcore
"""

from __future__ import annotations


class ConfigurationError(Exception):
    def __init__(self, message: str | None) -> None:
        self.message = message


class RecordNotFoundError(Exception):
    """Exception for the database manager."""

    pass
