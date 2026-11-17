# Inference service image. Not built in the authoring environment (no Docker daemon there).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir . uvicorn

RUN useradd --uid 10001 --create-home appuser
USER 10001
ENV LLMOPS_REGISTRY=/registry LLMOPS_MODEL=intent-classifier
EXPOSE 8080
CMD ["uvicorn", "llmops_gate.service:app", "--host", "0.0.0.0", "--port", "8080"]
