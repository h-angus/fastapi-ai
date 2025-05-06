FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first – this layer will be cached if requirements.txt is unchanged
COPY requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git \
 && pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir "huggingface_hub[hf_xet]" \
 && pip install --no-cache-dir -r requirements.txt \
 && apt-get remove -y build-essential git && apt-get autoremove -y && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Then copy source files – only triggers rebuild if your code changes
COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
