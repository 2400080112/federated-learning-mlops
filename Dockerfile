FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# dependencies first, so code edits do not invalidate the install layer
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY tests/ ./tests/
COPY README.md api.http ./

# the SQLite database lives on a volume in compose; this is just the default
ENV FEDFL_DATABASE_URL=sqlite:////app/data/app.db

WORKDIR /app/backend
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

# serve the app; run tests with: docker run --rm -w /app <image> python -m pytest -q
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
