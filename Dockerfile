# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd -r glassbox && useradd -r -g glassbox glassbox

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY pyproject.toml ./

# Install Python dependencies (without the package itself)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir ".[embeddings,vector-stores,databases,multimodal,telemetry,auth,reranking]"

# Copy source code
COPY src/ ./src/

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Create data directory
RUN mkdir -p /app/data && chown -R glassbox:glassbox /app

# Switch to non-root user
USER glassbox

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["glassbox-rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
