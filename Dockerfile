# Multi-stage build for production runtime
# Base image: Python 3.11 slim
FROM python:3.13.7-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash botuser

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=botuser:botuser . .

# Create necessary directories
RUN mkdir -p /app/data/cache /app/data/runtime /app/logs && \
    chown -R botuser:botuser /app/data /app/logs

# Switch to non-root user
USER botuser

# Expose port for admin API (if needed)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import config.settings; print('OK')" || exit 1

# Default command
CMD ["python", "-m", "apps.paper.run_paper", "--hours", "24"]
