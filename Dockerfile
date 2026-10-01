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
# torch's CPU-only wheel comes from PyTorch's own index -- installing from
# default PyPI can otherwise resolve a much larger CUDA-enabled build that
# this CPU-only app never uses.
COPY requirements.txt .
# sentence-transformers and optimum-onnx install last, with --no-deps -- their
# declared transformers version ranges don't overlap each other at all, even
# though both work correctly against the transformers version pinned in
# requirements.txt (see the comment there). Resolving all three together in
# one pass is a genuine ResolutionImpossible, not a resource/timeout issue.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps sentence-transformers==6.0.1 optimum-onnx==0.1.0

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
