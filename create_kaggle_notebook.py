import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Image Quality Assessment - Kaggle Training Notebook\n",
                "This notebook contains everything you need to train the EfficientNet-B0 model on Kaggle using a free GPU.\n",
                "\n",
                "**Setup Instructions:**\n",
                "1. Turn on GPU in the notebook settings (Accelerator: GPU T4 x2 or P100).\n",
                "2. Upload your existing codebase data directory via Kaggle Dataset.\n",
                "3. Adjust the `CSV` and `IMG_DIR` paths in the second code cell below to point to your data directory."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "from pathlib import Path\n",
                "import pandas as pd\n",
                "from PIL import Image\n",
                "import torch\n",
                "from torch import nn, optim\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from torchvision import transforms, models\n",
                "\n",
                "# --- TODO: Adjust these paths to point to your Kaggle data directory ---\n",
                "# For example, if you uploaded the dataset as 'iqa-data':\n",
                "# CSV = Path('/kaggle/input/iqa-data/data/labels.csv')\n",
                "CSV = Path('./data/labels.csv')\n",
                "IMG_DIR = Path('./data/images/')\n",
                "\n",
                "EPOCHS = 10\n",
                "BATCH = 32\n",
                "LR = 1e-4\n",
                "\n",
                "print(f\"Using dataset from {CSV}\")"
            ],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "class KonIQDataset(Dataset):\n",
                "    def __init__(self, dataframe, img_dir, is_train=False):\n",
                "        self.data = dataframe\n",
                "        self.img_dir = Path(img_dir)\n",
                "        \n",
                "        if is_train:\n",
                "            self.transform = transforms.Compose([\n",
                "                transforms.Resize((224, 224)),\n",
                "                transforms.RandomHorizontalFlip(),\n",
                "                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),\n",
                "                transforms.ToTensor(),\n",
                "                transforms.Normalize([0.485, 0.456, 0.406],\n",
                "                                     [0.229, 0.224, 0.225])\n",
                "            ])\n",
                "        else:\n",
                "            self.transform = transforms.Compose([\n",
                "                transforms.Resize((224, 224)),\n",
                "                transforms.ToTensor(),\n",
                "                transforms.Normalize([0.485, 0.456, 0.406],\n",
                "                                     [0.229, 0.224, 0.225])\n",
                "            ])\n",
                "\n",
                "    def __len__(self):\n",
                "        return len(self.data)\n",
                "\n",
                "    def __getitem__(self, idx):\n",
                "        row = self.data.iloc[idx]\n",
                "        image_name = row.get('image_name') or row.get('img_name') or row.get('filename')\n",
                "        img_path = self.img_dir / str(image_name)\n",
                "        image = Image.open(img_path).convert('RGB')\n",
                "        image = self.transform(image)\n",
                "        label = float(row['mos']) if 'mos' in row.index else float(row.iloc[-1])\n",
                "        return image, label"
            ],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "class QualityNet(nn.Module):\n",
                "    def __init__(self):\n",
                "        super().__init__()\n",
                "        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)\n",
                "        num_feats = base.classifier[1].in_features\n",
                "        base.classifier = nn.Identity()\n",
                "        self.base = base\n",
                "        self.regressor = nn.Sequential(\n",
                "            nn.Dropout(p=0.2),\n",
                "            nn.Linear(num_feats, 1)\n",
                "        )\n",
                "\n",
                "    def forward(self, x):\n",
                "        x = self.base(x)\n",
                "        return self.regressor(x).squeeze(1)"
            ],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print('Device:', device)\n",
                "\n",
                "full_df = pd.read_csv(CSV)\n",
                "df_shuffled = full_df.sample(frac=1, random_state=42).reset_index(drop=True)\n",
                "train_size = int(0.8 * len(df_shuffled))\n",
                "train_df = df_shuffled.iloc[:train_size]\n",
                "val_df = df_shuffled.iloc[train_size:]\n",
                "\n",
                "train_ds = KonIQDataset(dataframe=train_df, img_dir=IMG_DIR, is_train=True)\n",
                "val_ds = KonIQDataset(dataframe=val_df, img_dir=IMG_DIR, is_train=False)\n",
                "\n",
                "train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)\n",
                "val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)"
            ],
            "execution_count": None
        },
        {
            "cell_type": "code",
            "metadata": {},
            "outputs": [],
            "source": [
                "model = QualityNet().to(device)\n",
                "criterion = nn.MSELoss()\n",
                "optimizer = optim.Adam(model.parameters(), lr=LR)\n",
                "scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)\n",
                "scaler = torch.cuda.amp.GradScaler(enabled=device.type=='cuda')\n",
                "\n",
                "best_val_loss = float('inf')\n",
                "\n",
                "for epoch in range(EPOCHS):\n",
                "    # --- Train ---\n",
                "    model.train()\n",
                "    total_loss = 0.0\n",
                "    for imgs, labels in train_loader:\n",
                "        imgs, labels = imgs.to(device), labels.to(device).float()\n",
                "        optimizer.zero_grad()\n",
                "        \n",
                "        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):\n",
                "            outputs = model(imgs)\n",
                "            loss = criterion(outputs, labels)\n",
                "            \n",
                "        scaler.scale(loss).backward()\n",
                "        scaler.step(optimizer)\n",
                "        scaler.update()\n",
                "        total_loss += loss.item() * imgs.size(0)\n",
                "        \n",
                "    scheduler.step()\n",
                "    avg_train_loss = total_loss / len(train_loader.dataset)\n",
                "    \n",
                "    # --- Eval ---\n",
                "    model.eval()\n",
                "    val_loss = 0.0\n",
                "    with torch.no_grad():\n",
                "        for imgs, labels in val_loader:\n",
                "            imgs, labels = imgs.to(device), labels.to(device).float()\n",
                "            outputs = model(imgs)\n",
                "            val_loss += criterion(outputs, labels).item() * imgs.size(0)\n",
                "            \n",
                "    avg_val_loss = val_loss / len(val_loader.dataset)\n",
                "    print(f'Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')\n",
                "    \n",
                "    if avg_val_loss < best_val_loss:\n",
                "        best_val_loss = avg_val_loss\n",
                "        torch.save(model.state_dict(), 'model.pth')\n",
                "        print(f'-> Saved best model with Val Loss: {best_val_loss:.4f}')\n",
                "\n",
                "print(\"Training complete! Download 'model.pth' and move it to the 'artifacts' directory in your project locally for deployment.\")"
            ],
            "execution_count": None
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("kaggle_train.ipynb", "w", encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
print("Notebook created successfully!")
