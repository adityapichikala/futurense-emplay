"""Structured logging configuration.

Runs emit JSON-ish key/value logs so a run can be audited (which rules fired,
which documents were superseded) without re-executing anything.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure stdlib logging + structlog with a shared processor chain.

    The explicit ``setLevel`` matters: :func:`logging.basicConfig` is a no-op
    once the root logger has handlers, so without it a second call (for example
    lowering to DEBUG) would silently do nothing.
    """
    resolved = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(levelname)s %(name)s %(message)s",
        level=resolved,
        stream=None,
    )
    logging.getLogger().setLevel(resolved)
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
