
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY config.json .
COPY src/ ./src/
COPY registry/ ./registry/

CMD ["python", "main.py", "--help"]

