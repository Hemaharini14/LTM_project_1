# SmartRouteAI - Flask app (delay prediction + trip planning).
#
# Model artifacts (models/artifacts/), the trained datasets (dataset/, outputs/)
# and the SQLite database are all gitignored - regenerated, not shipped in the
# repo - so they are NOT baked into this image either. Mount them as a volume,
# or run the training scripts inside the running container once. Without
# models/artifacts/, the app still starts (every LLM/model path degrades
# gracefully - see README's "Where an LLM deliberately does not run"), it just
# has no delay predictions until a model exists.
FROM python:3.11-slim

WORKDIR /app

# torch's CPU wheel and a couple of scientific packages need a C toolchain to
# resolve/build on slim images; this keeps the final image small by not
# shipping the dev packages, only what pip needed to install the wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    FLASK_ENV=production

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:5000/health', timeout=4).status_code==200 else 1)"

# gunicorn, not the Flask dev server (api.py's own `app.run(debug=True)` is
# dev-only and single-threaded) - 2 workers is a reasonable default for the
# torch model's memory footprint; tune via WEB_CONCURRENCY at deploy time.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:5000 --workers ${WEB_CONCURRENCY:-2} --timeout 120 api:app"]
