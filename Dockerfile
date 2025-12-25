FROM python:3.10-slim

# System dependencies for audio decoding + soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script into the image
COPY txt-format.py /app/txt-format.py

# Run the script by default
ENTRYPOINT ["python", "/app/txt-format.py"]
