FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y build-essential && apt-get autoremove -y && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY . .

# DEBUG: List all files and their sizes (top 50 biggest)
RUN apt-get update && apt-get install -y --no-install-recommends tree && \
    echo "=== FILE SIZE DEBUG ===" && \
    du -ah /app | sort -rh | head -n 50 && \
    echo "=== END DEBUG ==="


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
