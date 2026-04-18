FROM python:3.11-slim

# Set up a new user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# system deps for pillow/opencv
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
       build-essential \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
USER user

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy the rest of the application
COPY --chown=user . .

# Ensure artifacts directory exists
RUN mkdir -p artifacts

# HF Spaces usually use port 7860
ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.inference_api:app --host 0.0.0.0 --port ${PORT:-7860}"]
