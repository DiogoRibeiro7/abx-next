"""Application-wide logging helpers with environment-based level control."""

from __future__ import annotations

import logging
import os

LOGGER_NAME = "abx_next"
_ENV_VAR = "LOG_LEVEL"


def _resolve_level() -> int:
    """Return the configured logging level, falling back to INFO."""
    env_level = os.getenv(_ENV_VAR)
    if env_level:
        candidate = getattr(logging, env_level.upper(), None)
        if isinstance(candidate, int):
            return candidate
    return logging.INFO


_BASE_LOGGER = logging.getLogger(LOGGER_NAME)
_BASE_LOGGER.setLevel(_resolve_level())
_BASE_LOGGER.addHandler(logging.NullHandler())


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a child logger of the package-level ``abx_next`` logger.

    Parameters
    ----------
    name:
        Optional dotted suffix to create hierarchical loggers, e.g.
        ``get_logger("sim.power_mean")``.
    """
    if name:
        return _BASE_LOGGER.getChild(name)
    return _BASE_LOGGER


__all__ = ["LOGGER_NAME", "get_logger"]

