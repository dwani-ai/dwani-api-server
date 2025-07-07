# Build stage
FROM python:3.10-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y gcc curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app

# Copy only the installed packages from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app/requirements.txt .

# Ensure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:7860/v1/health || exit 1

CMD ["python", "-m" , "src.app.main", "--host", "0.0.0.0", "--port", "7860"]
