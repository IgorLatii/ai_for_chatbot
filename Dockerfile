FROM python:3.10-slim

WORKDIR /app

# Install minimal build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "scripts.main:app", "--host", "0.0.0.0", "--port", "8000"]