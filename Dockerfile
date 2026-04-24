FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install torch (CPU only) – required for zero-shot classifier (transformers)
RUN pip install --default-timeout=600 --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
RUN pip install --default-timeout=600 --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set Hugging Face cache directory
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache/huggingface

EXPOSE 8000

# CHANGED: Points to app/main.py inside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]