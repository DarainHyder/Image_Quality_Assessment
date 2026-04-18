# src/inference_api.py
from pathlib import Path
import io
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Import your real model class (must exist in src/model.py)
from src.model import QualityNet

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACTS / "model.pth"
MODEL_FINAL = ARTIFACTS / "model_final.pth"
NEW_FINAL_MODEL = ARTIFACTS / "new_final_model.pth"
ALT_MODEL = ROOT.parents[0] / "model.pth"

app = FastAPI(title="Image Quality Assessment API")

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Preprocess transform (same used in train.py/dataset)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Try loading model robustly
model = None
model_loaded = False
load_errors = []

candidate_paths = [NEW_FINAL_MODEL, MODEL_PATH, MODEL_FINAL, ALT_MODEL]
for p in candidate_paths:
    if not p.exists():
        continue
    try:
        # map_location='cpu' is important for HF Spaces if no GPU is available
        checkpoint = torch.load(p, map_location="cpu")
        
        # instantiate model then try load
        m = QualityNet()
        
        # Handle different checkpoint formats
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

        m.load_state_dict(state_dict, strict=True)
        model = m
        model_loaded = True
        print(f"[model] loaded from {p}")
        break
    except Exception as e:
        load_errors.append((p, str(e)))

if not model_loaded:
    print("Model NOT loaded. Tried paths:", [str(p) for p in candidate_paths])
else:
    model.eval()

@app.get("/")
async def root():
    return {
        "status": "online",
        "model_loaded": bool(model_loaded),
        "info": "Image Quality Assessment API (EfficientNet-B0)"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not model_loaded or model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server")
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {e}")

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        score = float(out.item())
    
    # We round the score for the UI
    return JSONResponse({
        "quality_score": round(score, 4),
        "status": "success"
    })
