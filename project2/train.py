import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
import json
import json
import shutil
from torch.utils.tensorboard import SummaryWriter
import sys

def load_metadata(data_dir):
    meta_path = os.path.join(data_dir, "dataset_config.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}. Run generate_data.py first.")

    with open(meta_path, "r") as f:
        config = json.load(f)
        return tuple(config["img_shape"])

class CartPoleDataset(Dataset):
    def __init__(self, data_dir="data", transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.labels = np.load(os.path.join(data_dir, "labels.npy"))
        try:
             img_shape = load_metadata(data_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                "dataset_config.json not found. Please run generate_data.py to create the dataset and metadata."
            )

        num_samples = len(self.labels)
        self.images = np.memmap(os.path.join(data_dir, "images.mmap"), dtype=np.uint8, mode='r', shape=(num_samples, *img_shape))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

class SimpleCNN(nn.Module):
    def __init__(self, input_size=400):
        super(SimpleCNN, self).__init__()
        final_dim = input_size // 4

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_input_features = 32 * final_dim * final_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_features, 120),
            nn.ReLU(),
            nn.Linear(120, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

def get_model(input_size):
    return SimpleCNN(input_size=input_size)

def train_model(data_dir="data", model_save_path="position_detection.pth", num_epochs=25, batch_size=32, learning_rate=0.001):
    if os.path.exists("runs"):
        shutil.rmtree("runs")
    writer = SummaryWriter(log_dir="runs/cartpole_experiment")

    try:
        img_shape = load_metadata(data_dir)
    except FileNotFoundError:
         print(f"Error: Data files not found in {data_dir}. Please run generate_data.py first.")
         return

    crop_size = min(img_shape[0], img_shape[1])
    print(f"Detected Image Shape: {img_shape}. Using Crop Size: {crop_size}x{crop_size}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(crop_size),
    ])

    try:
        dataset = CartPoleDataset(data_dir=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Data files not found in {data_dir}. Please run generate_data.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = get_model(input_size=crop_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting training with SimpleCNN and TensorBoard logging...")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar("Training Loss/Epoch", epoch_loss, epoch)

    print("Finished training.")

    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()
    print("TensorBoard logs saved to runs/cartpole_experiment")

if __name__ == '__main__':
    train_model(num_epochs=25, batch_size=512, learning_rate=5e-5)
