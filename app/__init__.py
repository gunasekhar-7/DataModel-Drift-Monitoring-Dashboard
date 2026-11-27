"""Main FastAPI application factory."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest

from app.api.monitoring_routes import monitoring_router
from app.api.routers import health_router, predict_router
from app.core.config import settings
from app.core.database import init_db
from app.core.logging import get_logger

# Initialize logger
logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "app_requests_total",
    "Total number of requests",
    ["method", "endpoint", "status"],
)
REQUEST_DURATION = Histogram(
    "app_request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for application startup and shutdown."""
    logger.info(
        "Application starting",
        extra={
            "app_name": settings.app_name,
            "version": settings.app_version,
            "environment": settings.environment,
        },
    )
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Application shutting down")


def get_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Data & Model Drift Monitoring Platform",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Set up CORS
    origins = [
        o.strip()
        for o in settings.cors_origins.split(",")
        if o.strip()
    ] if settings.cors_origins else ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(health_router)
    app.include_router(predict_router)
    app.include_router(monitoring_router)

    # Prometheus metrics endpoint
    @app.get("/metrics")
    async def metrics():
        """Expose Prometheus metrics."""
        return generate_latest()

    return app


# Backwards compatibility and alternative entrypoints
create_app = get_app
app = get_app()
