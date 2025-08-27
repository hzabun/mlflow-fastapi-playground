FROM python:3.10-slim

WORKDIR /app

# Install MLflow, curl for health checks, and other dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install mlflow==3.3.1

# Create directory and set up permissions with wider access
RUN mkdir -p /mlflow/artifacts && \
    chmod -R 777 /mlflow

# Expose MLflow UI port
EXPOSE 5000

# Run as root to avoid permission issues (safe for this use case)
CMD ["mlflow", "server", \
    "--host", "0.0.0.0", \
    "--port", "5000", \
    "--default-artifact-root", "/mlflow/artifacts"]