FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app/src

COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY configs /app/configs
COPY artifacts /app/artifacts
COPY data /app/data

RUN pip install --no-cache-dir -e ".[dev]"

EXPOSE 8101
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8101/health')"

CMD ["uvicorn", "resolvelm.api.app:app", "--host", "0.0.0.0", "--port", "8101"]
