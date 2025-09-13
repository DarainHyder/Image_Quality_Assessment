from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class KonIQDataset(Dataset):
    def __init__(self, csv_path=None, img_dir=None, transform=None):
        # Resolve project root and sensible defaults
        root = Path(__file__).resolve().parents[1]
        if csv_path is None:
            csv_path = root / "data" / "labels.csv"
        else:
            csv_path = Path(csv_path)

        if img_dir is None:
            img_dir = root / "data" / "images"
        else:
            img_dir = Path(img_dir)

        if not csv_path.exists():
            raise FileNotFoundError(f"labels CSV not found at {csv_path}")

        if not img_dir.exists():
            raise FileNotFoundError(f"image directory not found at {img_dir}")

        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_name = row.get('image_name') or row.get('img_name') or row.get('filename')
        if pd.isna(image_name):
            raise ValueError(f"Missing image name at row {idx}")

        img_path = self.img_dir / str(image_name)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        label = float(row['mos']) if 'mos' in row.index else float(row.iloc[-1])
        return image, label
