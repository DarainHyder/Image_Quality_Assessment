from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from torchvision import transforms
from model import QualityNet

app = FastAPI()

model = QualityNet()
model.load_state_dict(torch.load("../model.pth", map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        score = model(tensor).item()
    return {"quality_score": score}
