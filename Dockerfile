# syntax=docker/dockerfile:1
# ── Multi-stage Dockerfile for all four RAG services ──────────────────────────
#
# Build args (set per-service in docker-compose or ACA):
#   SERVICE_MODULE  — uvicorn module:app path (default: agents.main_agent:app)
#   SERVICE_PORT    — port the service listens on (default: 8000)
#
# Usage examples:
#   docker build --build-arg SERVICE_MODULE=agents.orchestrator_agent:app \
#                --build-arg SERVICE_PORT=8001 -t rag-orchestrator .
#
#   docker build --build-arg SERVICE_MODULE=teams_bot:app \
#                --build-arg SERVICE_PORT=3978 -t rag-teams-bot .

# ── Stage 1: dependency installation ──────────────────────────────────────────
FROM python:3.12-slim AS deps

WORKDIR /app

# Install system packages needed by some Python deps (e.g. cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Stage 2: final image ───────────────────────────────────────────────────────
FROM python:3.12-slim AS final

WORKDIR /app

# Non-root user — ACA and Kubernetes best practice.
RUN useradd --no-create-home --shell /bin/false appuser

# Copy installed packages from the deps stage
COPY --from=deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=deps /usr/local/bin            /usr/local/bin

# Copy application source
COPY . .

RUN chown -R appuser:appuser /app
USER appuser

# Build-time arguments — can be overridden per service at docker build time.
ARG SERVICE_MODULE=agents.main_agent:app
ARG SERVICE_PORT=8000

# Expose as env vars so the CMD can reference them at runtime.
ENV SERVICE_MODULE=${SERVICE_MODULE}
ENV SERVICE_PORT=${SERVICE_PORT}

EXPOSE ${SERVICE_PORT}

# ACA liveness probe — lightweight Python urllib call; no extra tools needed.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request, os, sys; \
         port=os.environ.get('SERVICE_PORT','8000'); \
         urllib.request.urlopen(f'http://localhost:{port}/health/live', timeout=5); \
         sys.exit(0)"

CMD uvicorn ${SERVICE_MODULE} \
        --host 0.0.0.0 \
        --port ${SERVICE_PORT} \
        --workers 1 \
        --no-access-log
