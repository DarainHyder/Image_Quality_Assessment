from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from dataset import KonIQDataset
from model import QualityNet
import os

# --- Project paths (always resolve relative to repo root)
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "labels.csv"
IMG_DIR = ROOT / "data" / "images"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# --- Hyperparams (tweak as needed)
EPOCHS = int(os.getenv("EPOCHS", 3))
BATCH = int(os.getenv("BATCH_SIZE", 16))
LR = float(os.getenv("LR", 1e-4))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Dataset & loaders
dataset = KonIQDataset(csv_path=CSV, img_dir=IMG_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=2)

# --- Model, loss, optimizer
model = QualityNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --- Training loop (lightweight)
best_val_loss = float("inf")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}")

    # validation
    model.eval()
    val_loss = 0.0
    with torch.inference_mode():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{EPOCHS} - Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_path = ARTIFACTS / "model.pth"
        torch.save(model.state_dict(), save_path)
        print("Saved best model to", save_path)

# final save (ensure something exists)
final_save = ARTIFACTS / "model_final.pth"
torch.save(model.state_dict(), final_save)
print("Saved final model to", final_save)
