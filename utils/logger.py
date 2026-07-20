"""
Structured logging configuration for ETHOS.

Provides a single, consistent logging setup that can be imported by any module.
"""

import logging
import sys
from typing import Literal


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _StdoutStderrFilter(logging.Filter):
    """Route messages by severity: INFO/DEBUG to stdout, WARNING+ to stderr."""

    def __init__(self, level: int, name: str = "") -> None:
        super().__init__(name)
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.level


def configure_logging(
    level: LogLevel = "INFO",
    fmt: str | None = None,
) -> None:
    """
    Configure root logging for the application.

    Args:
        level: Minimum log level emitted.
        fmt: Optional custom log format string.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(_StdoutStderrFilter(logging.WARNING))
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper()))
    root.handlers.clear()
    root.addHandler(stdout_handler)
    root.addHandler(stderr_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured with the application's settings."""
    return logging.getLogger(name)
