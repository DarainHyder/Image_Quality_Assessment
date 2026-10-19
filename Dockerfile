# Lumina IQA inference API — CPU, ONNX Runtime. Used by the Hugging Face Space and the GHCR image.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    LUMINA_MODEL_DIR=/home/user/app/artifacts \
    PORT=7860
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user lumina/__init__.py lumina/constants.py lumina/analysis.py lumina/
COPY --chown=user server/ server/

# Bake the int8 ONNX weights + model card into the image (pulled from the HF Hub model repo).
RUN python -c "from server.app import _ensure_model; print(_ensure_model())"

EXPOSE 7860
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
