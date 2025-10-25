"""Custom exception hierarchy for abx-next."""

from __future__ import annotations


class AbxError(Exception):
    """Base exception for library-specific errors."""


class ValidationError(ValueError, AbxError):
    """Raised when input validation fails."""


class StatError(RuntimeError, AbxError):
    """Raised when statistical routines fail to produce a valid result."""

