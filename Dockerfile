FROM python:3.10-slim

WORKDIR /app

# System dependencies for audio decoding + downloading the model
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    wget \
  && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Your script
COPY txt-format.py /app/txt-format.py

ENTRYPOINT ["python", "/app/txt-format.py"]
