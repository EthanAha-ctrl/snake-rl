import torch
import torch.nn as nn
import numpy as np

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, history_len=10, channels=10, H=15, W=20, d_model=64, nhead=1, num_layers=2, out_dim=256, action_history_dim=20):
        super().__init__()
        self.history_len = history_len
        self.channels = channels
        self.H = H
        self.W = W
        self.d_model = d_model
        
        # Linear projection from 10 channels to d_model (64)
        self.proj = nn.Linear(channels, d_model)
        
        # 3D Absolute Positional Encodings
        self.pe_t = nn.Embedding(history_len, d_model)
        self.pe_y = nn.Embedding(H, d_model)
        self.pe_x = nn.Embedding(W, d_model)
        
        # [CLS] Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection: [CLS] (d_model) + Action History (action_history_dim) -> out_dim (256)
        # Note: We will handle the integration in sac_trainer.py directly, 
        # but encapsulating the final projection here is cleaner.
        self.out_proj = nn.Linear(d_model + action_history_dim, out_dim)

    def forward(self, obs_stack):
        """
        obs_stack: [B, 30000 + 20] corresponding to (history_len * channels * H * W) + action_history
        returns: [B, out_dim]
        """
        B = obs_stack.shape[0]
        
        # 1. Split Vision and Action History
        vision_dim = self.history_len * self.channels * self.H * self.W
        vision_feat = obs_stack[:, :vision_dim]     # [B, 30000]
        action_feat = obs_stack[:, vision_dim:]     # [B, 20]
        
        # 2. Reshape Vision to [B, T, C, H, W] then [B, T, H, W, C]
        vision_feat = vision_feat.view(B, self.history_len, self.channels, self.H, self.W)
        vision_feat = vision_feat.permute(0, 1, 3, 4, 2) # [B, 10, 15, 20, 10]
        
        # 3. Apply Linear Projection to C dimension -> [B, 10, 15, 20, 64]
        vision_proj = self.proj(vision_feat)
        
        # 4. Add 3D Positional Encodings
        # Create coordinate grids
        device = vision_proj.device
        t_coords = torch.arange(self.history_len, device=device).view(self.history_len, 1, 1)
        y_coords = torch.arange(self.H, device=device).view(1, self.H, 1)
        x_coords = torch.arange(self.W, device=device).view(1, 1, self.W)
        
        # Compute embeddings and sum them up (Broadcasting handles the dimensions)
        pos_emb = self.pe_t(t_coords) + self.pe_y(y_coords) + self.pe_x(x_coords) # [10, 15, 20, 64]
        
        # Add PE to projected features
        vision_with_pe = vision_proj + pos_emb.unsqueeze(0) # [B, 10, 15, 20, 64]
        
        # 5. Flatten Spatio-Temporal dimensions to form sequence
        seqence_feat = vision_with_pe.view(B, -1, self.d_model) # [B, 3000, 64]
        
        # 6. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 64]
        seqence_feat = torch.cat([cls_tokens, seqence_feat], dim=1) # [B, 3001, 64]
        
        # 7. Pass through Transformer
        encoded_seq = self.transformer(seqence_feat) # [B, 3001, 64]
        
        # 8. Extract [CLS] token output
        cls_out = encoded_seq[:, 0, :] # [B, 64]
        
        # 9. Concatenate with Action History
        final_feat = torch.cat([cls_out, action_feat], dim=1) # [B, 64 + 20 = 84]
        
        # 10. Project to 256 dimensions to mimic old MLP feature
        out = self.out_proj(final_feat) # [B, 256]
        
        return out
