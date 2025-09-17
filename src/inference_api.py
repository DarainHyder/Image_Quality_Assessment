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
import gradio as gr

# Import your real model class (must exist in src/model.py)
from src.model import QualityNet

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "model.pth"
MODEL_FINAL = ARTIFACTS / "model_final.pth"
ALT_MODEL = ROOT.parents[0] / "model.pth"  # possible alternate location

app = FastAPI(title="Image Quality Assessment API")

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

candidate_paths = [MODEL_PATH, MODEL_FINAL, ALT_MODEL]
for p in candidate_paths:
    if not p.exists():
        continue
    try:
        checkpoint = torch.load(p, map_location="cpu")
        # Case A: full nn.Module object saved (not common, but possible)
        if isinstance(checkpoint, nn.Module):
            model = checkpoint
            model_loaded = True
            print(f"[model] loaded full module from {p}")
            break

        # Case B: dict — maybe state_dict or checkpoint with key 'model_state_dict' or 'state_dict'
        state_dict = None
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                # assume it's already a state_dict
                state_dict = checkpoint

        # instantiate model then try load
        m = QualityNet()
        try:
            m.load_state_dict(state_dict)  # strict True first
            model = m
            model_loaded = True
            print(f"[model] loaded state_dict (strict) from {p}")
            break
        except Exception as e_strict:
            # try non-strict (allow missing / unexpected keys)
            try:
                m.load_state_dict(state_dict, strict=False)
                model = m
                model_loaded = True
                print(f"[model] loaded state_dict (non-strict) from {p} (some keys ignored)")
                break
            except Exception as e_non_strict:
                load_errors.append((p, str(e_non_strict)))
    except Exception as e:
        load_errors.append((p, str(e)))

if not model_loaded:
    print("Model NOT loaded. Tried paths:", [str(p) for p in candidate_paths])
    for p, err in load_errors:
        print(f" - failed {p}: {err}")
else:
    model.eval()

# --------------------
# FastAPI endpoints
# --------------------
@app.get("/")
async def root():
    return {
        "status": "ok",
        "model_loaded": bool(model_loaded),
        "gradio_ui": "/gradio"
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
    return JSONResponse({"quality_score": score})

# --------------------
# Gradio UI (mounted at /gradio)
# --------------------
def gradio_infer(image):
    if not model_loaded or model is None:
        return "Model not loaded on server. Check server logs."
    # Gradio sometimes sends numpy arrays; coerce to PIL
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if not isinstance(image, Image.Image):
        return "Invalid image input."
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        score = float(out.item())
    return f"{score:.4f}"

gr_interface = gr.Interface(
    fn=gradio_infer,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs=gr.Textbox(label="Quality score"),
    title="📸 Image Quality Assessment",
    description="Upload image → get predicted quality score (MOS-like)"
)

from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request


# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates
templates = Jinja2Templates(directory="templates")

@app.get("/ui")
async def ui_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Mount Gradio app inside FastAPI (ASGI compatible)
app = gr.mount_gradio_app(app, gr_interface, path="/gradio")
