FROM python:3.12-slim AS builder

WORKDIR /app

# System dependencies for networkx, pyvis plotting if native compilation is needed
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Build wheels for production dependencies
COPY requirements-prod.txt .
RUN pip wheel --no-cache-dir -r requirements-prod.txt -w /wheels

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /wheels /wheels
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements-prod.txt \
    && rm -rf /wheels

# Copy project files
COPY . .

# Run as a non-root user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose OpenEnv port
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:7860/healthz || exit 1

# Command to run the OpenEnv server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
