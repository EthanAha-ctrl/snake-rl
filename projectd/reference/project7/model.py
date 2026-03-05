import torch
import torch.nn as nn

class MILPatchCNN(nn.Module):
    """
    Multiple Instance Learning (MIL) built on 32x32 patches.
    It completely physically isolates the receptive field to 32x32 blocks, preventing
    global features from bleeding into background focus predictions.
    """
    def __init__(self, num_classes=10):
        super(MILPatchCNN, self).__init__()
        
        # Simple, highly localized Feature Extractor 
        # Input: [B*300, 1, 32, 32] -> Output: [B*300, 64, 4, 4] -> [B*300, 1024]
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 8x8
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # 4x4
        )
        
        # Classifier for each 32x32 patch independently
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x: [B, 1, 480, 640]
        B = x.size(0)
        
        # Hard physical slice: Unfold creates 300 non-overlapping 32x32 patches
        # 480 / 32 = 15; 640 / 32 = 20
        # Dim 2 is Height, Dim 3 is Width
        patches = x.unfold(2, 32, 32).unfold(3, 32, 32) # [B, 1, 15, 20, 32, 32]
        
        # Reshape to [B, 300, 1, 32, 32]
        patches = patches.contiguous().view(B, 300, 1, 32, 32)
        
        # Merge Batch and Patch dim to run through standard CNN
        # This absolutely guarantees 0 receptive field crossover!
        patches_flat = patches.view(B * 300, 1, 32, 32)
        
        # Feature Extraction
        feats = self.features(patches_flat)
        feats = feats.view(feats.size(0), -1)
        
        # Independent patch-level spatial logits: [B*300, 10]
        patch_logits = self.classifier(feats) 
        
        # Reshape back to [B, 300, 10]
        patch_logits = patch_logits.view(B, 300, 10)
        
        if self.training:
            # During training, we use Max Pooling over the 300 patches to find the most 
            # "confident" patch (the sharpest foreground object).
            global_logits, _ = torch.max(patch_logits, dim=1) 
            return global_logits
            
        else:
            # During inference (generate_tensor_db), we return the dense [B, 10, 15, 20] tensor!
            spatial_logits = patch_logits.permute(0, 2, 1).view(B, 10, 15, 20)
            return spatial_logits
