"""Core package."""


def create_app(*args, **kwargs):
    """Application factory for uvicorn app.core:create_app entrypoint."""
    from app import get_app

    return get_app(*args, **kwargs)


def get_app(*args, **kwargs):
    """Application factory for uvicorn app.core:get_app entrypoint."""
    from app import get_app as _get_app

    return _get_app(*args, **kwargs)


__all__ = ["create_app", "get_app"]
