FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy training script
COPY train.py .

# Set MLflow environment variables
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]