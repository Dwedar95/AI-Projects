# Not built in the authoring environment (no Docker daemon available there).
FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . uvicorn

RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "agentic_orchestrator.api:app", "--host", "0.0.0.0", "--port", "8000"]
