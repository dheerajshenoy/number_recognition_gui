import torch
import pandas as pd
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.iloc[:, 1:].values  # all columns except last
        self.y = self.data.iloc[:, 0].values   # last column as label
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.X[idx].reshape(28, 28).astype("uint8")  # reshape to image
        y = self.y[idx]

        if self.transform:
            from PIL import Image
            x = Image.fromarray(x)
            x = self.transform(x)
        else:
            x = torch.tensor(x, dtype=torch.float32).unsqueeze(0) / 255.0  # fallback

        return x, torch.tensor(y, dtype=torch.long)
