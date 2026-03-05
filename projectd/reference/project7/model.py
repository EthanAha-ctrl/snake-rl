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

class BasicBlock(nn.Module):
    """标准的 ResNet 基础残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class MiniHRNetMIL(nn.Module):
    """
    超轻量级 MiniHRNet-MIL:
    1. 输入 32x32 非重叠切片 (300 块)。
    2. 3 个并行分支阶段 (B1:32x32, B2:16x16, B3:8x8)。
    3. 所有通道固定为 16 (NPU 友好, 减少内存带宽)。
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1 (Branch 1: 32x32)
        self.layer1 = BasicBlock(16, 16)
        
        # Transition 1 to 2
        self.trans1_to_2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Stage 2 (B1: 32x32, B2: 16x16)
        self.layer2_b1 = BasicBlock(16, 16)
        self.layer2_b2 = BasicBlock(16, 16)
        
        self.fuse2_21 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16))
        self.fuse2_12 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16))
        
        # Transition 2 to 3
        self.trans2_to_3 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Stage 3 (B1: 32x32, B2: 16x16, B3: 8x8)
        self.layer3_b1 = BasicBlock(16, 16)
        self.layer3_b2 = BasicBlock(16, 16)
        self.layer3_b3 = BasicBlock(16, 16)
        
        # Fusion 3
        self.fuse3_21 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16))
        self.fuse3_32 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=1, bias=False), nn.BatchNorm2d(16))
        self.fuse3_12 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16))
        self.fuse3_23 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(16))

        # Head (Combine B1, B2, B3)
        self.head_b1_down = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.head_b2_down = nn.Sequential(
            nn.Conv2d(16, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        import torch.nn.functional as F
        B = x.size(0)
        
        # 1. 物理切片：32x32 非重叠，[B, 1, 480, 640] -> [B, 300, 1, 32, 32]
        patches = x.unfold(2, 32, 32).unfold(3, 32, 32) 
        patches = patches.contiguous().view(B, 300, 1, 32, 32)
        patches_flat = patches.view(B * 300, 1, 32, 32)
        
        # 2. 多尺度特征提取
        out_stem = self.stem(patches_flat) 
        b1 = self.layer1(out_stem)
        b2 = self.trans1_to_2(b1) 
        
        b1_out = self.layer2_b1(b1)
        b2_out = self.layer2_b2(b2)
        b1_new = F.relu(b1_out + F.interpolate(self.fuse2_21(b2_out), scale_factor=2.0))
        b2_new = F.relu(b2_out + self.fuse2_12(b1_out))
        
        b3 = self.trans2_to_3(b2_new) 
        b1_out = self.layer3_b1(b1_new)
        b2_out = self.layer3_b2(b2_new)
        b3_out = self.layer3_b3(b3)
        
        b1_new = F.relu(b1_out + F.interpolate(self.fuse3_21(b2_out), scale_factor=2.0))
        b2_new = F.relu(b2_out + self.fuse3_12(b1_out) + F.interpolate(self.fuse3_32(b3_out), scale_factor=2.0))
        b3_new = F.relu(b3_out + self.fuse3_23(b2_out))

        # 3. 分类降维输出
        b1_reduced = self.head_b1_down(b1_new)
        b12 = b1_reduced + b2_new
        b12_reduced = self.head_b2_down(b12)
        b123 = b12_reduced + b3_new
        
        feats = F.adaptive_avg_pool2d(b123, (1, 1)).view(B * 300, -1)
        
        patch_logits = self.classifier(feats) 
        patch_logits = patch_logits.view(B, 300, 10)
        
        if self.training:
            global_logits, _ = torch.max(patch_logits, dim=1) 
            return global_logits
        else:
            spatial_logits = patch_logits.permute(0, 2, 1).view(B, 10, 15, 20)
            return spatial_logits
