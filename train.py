import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.nn.modules import loss as ls
from torchvision import transforms
from dataset import SketchDataset, SketchDatasetFromCloud

from modules import CNN

def train_loop():
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("No access to CUDA device, model being trained on CPU.")


    class_names = ["cat", "apple", "key", "bed", "basketball", 
                   "cake", "cloud", "crown", "duck", "fish"
                   ]
    transform = transforms.ToTensor()
    # dataset = SketchDatasetFromCloud("quickdraw_dataset/full/numpy_bitmap", class_names, transform=transform)
    dataset = SketchDataset("../zzDatasets/quickdraw_downloads", class_names, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    # Create splits
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # for i, (images, labels) in enumerate(train_loader):
    #     for idx in range(images.size[0]):
    #         print(images.size())
    #         break

    model = CNN(len(class_names)).to(device)
    criterion = ls.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase 
        model.train()
        running_loss = 0.0

        for batch_images, batch_labels in train_loader:

            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_images)

            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_dataset)
        print(f"Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_loss = 0.0
        # [correct count, wrong count]
        correct = [0, 0]
        with torch.no_grad():
            for batch_images, batch_labels in val_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                correct[0] += (preds == batch_labels).sum().item()
                correct[1] += (preds != batch_labels).sum().item()

        epoch_val_loss = val_loss / len(val_dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        print(correct)

train_loop()


