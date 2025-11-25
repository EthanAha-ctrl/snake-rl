import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class CartPoleDataset(Dataset):
    """
    CartPole dataset that reads from a memory-mapped file.
    This is very efficient for large datasets as it doesn't load the whole
    dataset into RAM.
    """
    def __init__(self, data_dir="data", transform=None):
        """
        Args:
            data_dir (string): Directory with the images.mmap and labels.npy files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        
        # Load labels to get dataset length
        self.labels = np.load(os.path.join(data_dir, "labels.npy"))
        
        # To determine the shape of the images, load metadata saved during generation
        try:
            with open(os.path.join(data_dir, "metadata.txt"), "r") as f:
                shape_str = f.read()
                img_shape = tuple(map(int, shape_str.split(',')))
        except FileNotFoundError:
            raise FileNotFoundError(
                "metadata.txt not found. Please run generate_data.py to create the dataset and metadata."
            )

        num_samples = len(self.labels)
        self.images = np.memmap(os.path.join(data_dir, "images.mmap"), dtype=np.uint8, mode='r', shape=(num_samples, *img_shape))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Accessing the memmap array by index is very fast.
        # The OS handles loading the specific chunk of the file.
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.FloatTensor(label)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # The transform resizes to 224x224
        # After first max pool: 112x112
        # After second max pool: 56x56
        self.fc_input_features = 32 * 56 * 56 
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.fc_input_features, 120),
            nn.ReLU(),
            nn.Linear(120, 2) # 2 outputs: cart_position and pole_angle
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x

def get_model():
    """Returns the SimpleCNN model."""
    return SimpleCNN()

def train_model(data_dir="data", model_save_path="position_detection.pth", num_epochs=25, batch_size=32, learning_rate=0.001):
    """
    Trains the perception model.
    """
    writer = SummaryWriter(log_dir="runs/cartpole_experiment")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    try:
        dataset = CartPoleDataset(data_dir=data_dir, transform=transform)
    except FileNotFoundError:
        print(f"Error: Data files not found in {data_dir}. Please run generate_data.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = get_model()
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
    print("This script is for training the model.")
    print("To train the model, you would run a command like:")
    print("python cartpole_perception/train.py")
    # To run the training, uncomment the following line:
    train_model()
    pass

