FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PORT=5000 \
    ENVIRONMENT=production \
    ENABLE_DOCS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY templates ./templates
COPY static ./static
COPY Model/class_indices.json ./Model/class_indices.json
COPY Model/plant_disease_model.h5 ./Model/plant_disease_model.h5
COPY app.py ./app.py

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && mkdir -p /app/static/uploads \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT}/health/live || exit 1

CMD ["sh", "-c", "uvicorn plant_disease_api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
