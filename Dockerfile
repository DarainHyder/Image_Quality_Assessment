FROM python:3.11-slim

WORKDIR /app

# system deps for pillow/opencv (if you later add opencv)
RUN apt-get clean && rm -rf /var/lib/apt/lists/* \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN python -m pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# copy repo
COPY . .

# ensure artifacts folder exists at runtime (model should be baked in or downloaded)
RUN mkdir -p /app/artifacts

# Port — uvicorn will read from $PORT, default to 8000
ENV PORT=8000

EXPOSE 8000

# Use shell form so $PORT expands
CMD ["sh", "-c", "uvicorn src.inference_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
