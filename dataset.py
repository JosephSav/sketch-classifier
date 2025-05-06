from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
import gcsfs
import os

import matplotlib.pyplot as plt
import numpy as np

class SketchDatasetFromCloud(Dataset):
    def __init__(self, gcs_path_prefix, class_names, transform=None, limit_per_class=1000):
        self.fs = gcsfs.GCSFileSystem()
        self.samples = []
        self.transform = transform

        for label, class_name in enumerate(class_names):
            file_path = f"{gcs_path_prefix}/{class_name}.npy"
            with self.fs.open(file_path, 'rb') as f:
                data = np.load(f)
                for img_array in data[:limit_per_class]:
                    self.samples.append((img_array, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_array, label = self.samples[idx]
        image = Image.fromarray(img_array.reshape(28, 28).astype(np.uint8), mode='L')
        if self.transform:
            image = self.transform(image)
        return image, label
    
class SketchDataset(Dataset):
    def __init__(self, local_path_prefix, class_names, transform=None, limit_per_class=10000):
        self.samples = []
        self.transform = transform

        for label, class_name in enumerate(class_names):
            file_path = os.path.join(local_path_prefix, f"{class_name}.npy")
            data = np.load(file_path)
            for img_array in data[:limit_per_class]:
                self.samples.append((img_array, label))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_array, label = self.samples[idx]
        image = Image.fromarray(img_array.reshape(28, 28).astype(np.uint8), mode='L')
        if self.transform:
            image = self.transform(image)
        return image, label
