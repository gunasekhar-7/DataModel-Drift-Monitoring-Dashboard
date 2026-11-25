"""Database session and engine management."""

import contextlib
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import settings

# Engine configuration with dialect-appropriate pooling
engine_kwargs: dict = {"echo": False}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update(
        {
            "pool_pre_ping": True,
            "pool_size": 5,
            "max_overflow": 10,
            "pool_timeout": 30,
        }
    )

engine = create_engine(settings.database_url, **engine_kwargs)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and ensures it's closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextlib.contextmanager
def get_db_context() -> Generator[Session, None, None]:
    """Context manager for standalone script / task database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """Get a raw database session instance."""
    return SessionLocal()


def init_db():
    """Initialize the database (create all tables)."""
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (for testing)."""
    Base.metadata.drop_all(bind=engine)
