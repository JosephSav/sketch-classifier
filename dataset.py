from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from io import BytesIO
import gcsfs

import matplotlib.pyplot as plt
import numpy as np

class SketchDataset(Dataset):
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
    

# Example usage
class_names = ["cat", "apple", "dog"]
transform = transforms.ToTensor()
dataset = SketchDataset("quickdraw_dataset/full/numpy_bitmap", class_names, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for batch_idx, (images, labels) in enumerate(loader):
    for i in range(images.size(0)):
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f"Label: {class_names[labels[i]]}")
        plt.show()
    
    if batch_idx == 3:  # stop after 2 batches
        break