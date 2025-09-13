from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
from torchvision import transforms
from model import QualityNet
from pathlib import Path
import io

app = FastAPI(title="Image Quality Assessment API")

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pth"
FALLBACK_MODEL = ARTIFACTS / "model_final.pth"

# Load model (fallback to final)
model = QualityNet()
load_path = MODEL_PATH if MODEL_PATH.exists() else (FALLBACK_MODEL if FALLBACK_MODEL.exists() else None)
if load_path is None:
    # If running in dev where user saved model elsewhere, try relative parent
    alt = Path(__file__).resolve().parents[2] / "model.pth"
    if alt.exists():
        load_path = alt

if load_path is None:
    # don't crash entire app, but requests will fail
    print("Warning: no model artifact found. Place model at", MODEL_PATH)
else:
    state = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("Loaded model from", load_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.get("/health")
def health():
    return {"ok": True, "model_loaded": model is not None and any(p.numel()>0 for p in model.parameters())}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or not any(p.numel()>0 for p in model.parameters()):
        raise HTTPException(status_code=503, detail="Model not loaded on server")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image uploaded: {e}")

    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        out = model(tensor)
        # ensure python float
        score = float(out.item())

    # return a simple JSON with the numeric MOS-like score
    return {"quality_score": score}
