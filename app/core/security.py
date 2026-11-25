"""Security utilities: request ID and correlation ID generation."""

import uuid

from fastapi import Request


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req-{uuid.uuid4().hex[:12]}"


def get_request_id_from_request(request: Request) -> str | None:
    """Extract request ID from request state or headers."""
    # Check for X-Request-ID header first
    request_id = request.headers.get("X-Request-ID")
    if request_id:
        return request_id
    # Generate new one
    return generate_request_id()


def add_request_id_to_logger(logger, request_id: str):
    """Add request ID to a logger adapter for contextual logging."""
    import logging
    adapter = logging.LoggerAdapter(logger, {"request_id": request_id})
    return adapter