FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt-get/lists/*

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir poetry
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi --no-root --only main

# Copy application code and assets
COPY app ./app
COPY dashboard ./dashboard
COPY data ./data
COPY scripts ./scripts

# Expose FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Start FastAPI application
CMD ["uvicorn", "app.core:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
