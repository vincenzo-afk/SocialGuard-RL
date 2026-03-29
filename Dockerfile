FROM python:3.12-slim

WORKDIR /app

# System dependencies for networkx, pyvis plotting if native compilation is needed
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run as a non-root user
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose streamlit port
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Command to run the dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
