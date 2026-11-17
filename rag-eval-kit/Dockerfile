# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Run as an unprivileged user. The audit log directory is a volume.
RUN useradd --create-home --uid 10001 rag && mkdir -p /data/audit && chown -R rag:rag /data
USER rag
ENV RAG_AUDIT_LOG=/data/audit/audit.jsonl

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status == 200 else 1)"

# RAG_API_KEYS must be provided at run time. The service refuses to start without it.
CMD ["rag-eval-kit", "serve", "--host", "0.0.0.0", "--port", "8000"]
