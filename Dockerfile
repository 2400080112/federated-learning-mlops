
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest

COPY main.py .
COPY config.json .
COPY src/ ./src/
COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "-v"]
