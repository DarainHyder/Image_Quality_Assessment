import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from dataset import KonIQDataset
from model import QualityNet

CSV = "../data/labels.csv"
IMG_DIR = "../data/images"
EPOCHS = 3
BATCH = 16
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = KonIQDataset(CSV, IMG_DIR)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH)

model = QualityNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "../model.pth")
