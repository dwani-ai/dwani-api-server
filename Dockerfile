# Use official Python runtime as base image
FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
WORKDIR /app

# Copy application code
COPY . .

# Expose port from settings
EXPOSE 7860

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:7860/v1/health || exit 1

# Command to run the application
CMD ["python", "/app/src/server/main.py", "--host", "0.0.0.0", "--port", "7860"]