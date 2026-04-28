FROM python:3.11-slim

WORKDIR /app

# System deps needed by TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create log directory expected by middleware/logging.py
RUN mkdir -p logs

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/quick')"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
