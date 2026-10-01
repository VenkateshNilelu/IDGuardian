# IDGuardian production image. Serves the Flask app via waitress (see
# app.py's __main__ block) on port 5000. Models (models/*.joblib) and the
# demo lookup dataset (data/demo_*.csv) are baked into the image -- this app
# has no build step and no external model registry, so "the image" *is* the
# deployable artifact.
FROM python:3.12-slim

# libgomp1: required at runtime by XGBoost's OpenMP-based prediction code.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first (separate layer, cached across code-only rebuilds).
# Deliberately NOT installing torch/sentence-transformers here -- app.py's
# serving path encodes text with onnxruntime directly (see its semantic-layer
# comment and requirements.txt's), and skipping torch saves ~250MB of resident
# memory that this image has no use for (that's what was OOM-killing the app
# on Render's 512MB free tier). Retraining needs requirements-train.txt instead.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

ENV HOST=0.0.0.0 \
    PORT=5000 \
    PYTHONUNBUFFERED=1

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5000/healthz', timeout=4)" || exit 1

CMD ["python", "app.py"]
