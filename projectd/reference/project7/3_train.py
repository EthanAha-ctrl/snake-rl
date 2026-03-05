import os
import time
import pickle
import lmdb
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import MiniHRNetMIL
from torch.utils.tensorboard import SummaryWriter

# --- Configuration ---
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
VAL_SPLIT = 0.1 # 10% for validation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(SCRIPT_DIR, "data")
LMDB_PATH = os.path.join(DATA_ROOT, "coc_img_10x480x640.lmdb")
META_PATH = os.path.join(DATA_ROOT, "coc_meta.pkl")
SAVE_DIR = os.path.join(SCRIPT_DIR, "checkpoints")
os.makedirs(SAVE_DIR, exist_ok=True)

class CoCDataset(Dataset):
    def __init__(self, lmdb_path, meta_path, transform=None):
        self.lmdb_path = lmdb_path
        self.meta_path = meta_path
        self.transform = transform
        
        with open(self.meta_path, 'rb') as f:
            self.meta_info = pickle.load(f)
            
        self.env = None # Lazy init in __getitem__ for multiprocessing

    def __len__(self):
        return len(self.meta_info)

    def __getitem__(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
            self.txn = self.env.begin(write=False)
            
        # Handle both old (2 args) and new (3 args) metadata formats
        item = self.meta_info[idx]
        if len(item) == 3:
            key_str, label_int, _ = item
        else:
            key_str, label_int = item
        
        # Read Bytes
        img_bytes = self.txn.get(key_str.encode('ascii'))
        
        # Decode from PNG bytes
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
             raise ValueError(f"Failed to decode image for key {key_str}")
        
        # To Tensor (0-1 float)
        # img is [H, W], add channel dim -> [1, H, W]
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)
        
        if self.transform:
            img = self.transform(img)
            
        # Label: 0-9
        label = torch.tensor(label_int, dtype=torch.long)
        
        return img, label

def main():
    print(f"Using device: {DEVICE}")
    
    # TensorBoard Setup
    writer = SummaryWriter(log_dir=os.path.join(SCRIPT_DIR, 'runs', 'coc_mil_patch_experiment_1'))

    # 1. Dataset & Dataloader
    train_transform = transforms.Compose([
         transforms.RandomHorizontalFlip(p=0.5),
    ])
    
    full_dataset = CoCDataset(LMDB_PATH, META_PATH, transform=train_transform)
    
    # Split
    total_size = len(full_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    
    train_set, val_set = random_split(full_dataset, [train_size, val_size], 
                                      generator=torch.Generator().manual_seed(42))
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # 2. Model: New MiniHRNetMIL
    model = MiniHRNetMIL(num_classes=10).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_acc = 0.0
    
    # 3. Loop
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        
        # --- Train ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            
            # outputs is now [B, 10] (global max pooled logits)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
                current_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), current_step)
                
        train_loss /= len(train_loader.dataset)
        train_acc = 100. * correct / total
        
        # Log Train Metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                
                # outputs: [B, 10, 15, 20] in eval mode
                spatial_outputs = model(inputs)
                
                # We need to simulate the Max Pooling logic for validation metric calculation
                # [B, 10, 15, 20] -> [B, 10, 300] -> max over dim 2 -> [B, 10]
                B = spatial_outputs.size(0)
                outputs_flat = spatial_outputs.view(B, 10, -1)
                global_outputs, _ = torch.max(outputs_flat, dim=2)
                
                loss = criterion(global_outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = global_outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = 100. * val_correct / val_total
        
        # Log Val Metrics
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        # Scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_patch.pth"))
            print(f"Saved Best Model (Acc: {best_acc:.2f}%)")
            
        # Save Last
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_model_patch.pth"))
    
    writer.close()

if __name__ == "__main__":
    main()
