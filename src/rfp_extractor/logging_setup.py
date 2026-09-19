"""Structured logging configuration.

Runs emit JSON-ish key/value logs so a run can be audited (which rules fired,
which documents were superseded) without re-executing anything.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging + structlog with a shared processor chain."""
    logging.basicConfig(
        format="%(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=None,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )
