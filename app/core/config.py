"""Core configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Data & Model Drift Monitoring Platform"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # API settings
    api_v1_str: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://localhost:8080"

    # Model settings
    model_artifact_path: str = "app/ml/model.pkl"
    preprocessor_artifact_path: str = "app/ml/preprocessor.pkl"
    feature_schema_path: str = "app/ml/features.json"
    model_metadata_path: str = "app/ml/metadata.json"

    # Database settings
    database_url: str = "sqlite:///./app.db"

    # Simulation settings
    simulator_reference_data_path: str = "data/processed/processed_telco.csv"
    simulator_batch_output_path: str = "data/processed/batches"
    simulator_default_batch_size: int = 100
    simulator_random_seed: int = 42

    # Logging settings
    log_level: str = "INFO"
    log_format: str = "json"


# Singleton instance
settings = Settings()
