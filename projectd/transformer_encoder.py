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
        
        # Linear projection for the sharpness scalar token
        self.sharpness_proj = nn.Linear(1, d_model)
        
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
        obs_stack: [B, 30010 + 20] corresponding to (history_len * (3000 + 1)) + action_history
        returns: [B, out_dim]
        """
        B = obs_stack.shape[0]
        
        # 1. Split Vision+Sharpness and Action History
        frame_dim = self.channels * self.H * self.W  # 3000
        vision_total_dim = self.history_len * (frame_dim + 1) # 30010
        obs_feat = obs_stack[:, :vision_total_dim]     # [B, 30010]
        action_feat = obs_stack[:, vision_total_dim:]  # [B, 20]
        
        # 2. Reshape obs_feat to [B, T, 3001] and split
        obs_feat = obs_feat.view(B, self.history_len, frame_dim + 1)
        sharpness_raw = obs_feat[:, :, 0:1] # [B, T, 1]
        vision_raw = obs_feat[:, :, 1:] # [B, T, 3000]
        
        # 3. Process Vision Pathway
        vision_feat = vision_raw.view(B, self.history_len, self.channels, self.H, self.W)
        vision_feat = vision_feat.permute(0, 1, 3, 4, 2) # [B, T=10, H=15, W=20, C=10]
        vision_proj = self.proj(vision_feat) # [B, 10, 15, 20, 64]
        
        # 4. Process Sharpness Pathway
        sharpness_proj = self.sharpness_proj(sharpness_raw) # [B, T=10, 64]
        
        # 5. Add 3D Positional Encodings
        device = vision_proj.device
        t_coords = torch.arange(self.history_len, device=device).view(self.history_len, 1, 1)
        y_coords = torch.arange(self.H, device=device).view(1, self.H, 1)
        x_coords = torch.arange(self.W, device=device).view(1, 1, self.W)
        
        # Vision PE
        pos_emb_v = self.pe_t(t_coords) + self.pe_y(y_coords) + self.pe_x(x_coords) # [10, 15, 20, 64]
        vision_with_pe = vision_proj + pos_emb_v.unsqueeze(0) # [B, 10, 15, 20, 64]
        vision_tokens = vision_with_pe.view(B, self.history_len, -1, self.d_model) # [B, 10, 300, 64]
        
        # Sharpness PE (Temporal only)
        t_coords_s = torch.arange(self.history_len, device=device)
        pos_emb_s = self.pe_t(t_coords_s) # [10, 64]
        sharpness_with_pe = sharpness_proj + pos_emb_s.unsqueeze(0) # [B, 10, 64]
        sharpness_tokens = sharpness_with_pe.unsqueeze(2) # [B, 10, 1, 64]
        
        # 6. Concatenate Vision and Sharpness tokens for each frame
        # [B, T, 300, 64] concat [B, T, 1, 64] -> [B, T, 301, 64]
        frame_tokens = torch.cat([vision_tokens, sharpness_tokens], dim=2)
        
        # 7. Flatten Spatio-Temporal dimensions to form sequence
        seqence_feat = frame_tokens.view(B, -1, self.d_model) # [B, 3010, 64]
        
        # 8. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 64]
        seqence_feat = torch.cat([cls_tokens, seqence_feat], dim=1) # [B, 3011, 64]
        
        # 9. Pass through Transformer
        encoded_seq = self.transformer(seqence_feat) # [B, 3011, 64]
        
        # 10. Extract [CLS] token output
        cls_out = encoded_seq[:, 0, :] # [B, 64]
        
        # 11. Concatenate with Action History
        final_feat = torch.cat([cls_out, action_feat], dim=1) # [B, 64 + 20 = 84]
        
        # 12. Project to 256 dimensions to mimic old MLP feature
        out = self.out_proj(final_feat) # [B, 256]
        
        return out
