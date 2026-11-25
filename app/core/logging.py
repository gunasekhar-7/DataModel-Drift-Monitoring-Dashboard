import logging

import structlog


def setup_logging():
    """Initialize application logging structure."""
    structlog.configure(
        processors=[
            structlog.processors.JSONRenderer(),
        ]
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance, initializing application logging if needed."""
    setup_logging()
    return logging.getLogger(name or __name__)
