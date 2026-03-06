import torch
import torch.nn as nn
import numpy as np

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, history_len=10, channels=11, H=15, W=20, d_model=64, nhead=1, num_layers=2, out_dim=256, action_history_dim=20):
        super().__init__()
        self.history_len = history_len
        self.channels = channels
        self.H = H
        self.W = W
        self.d_model = d_model
        
        # Linear projection from channels (11) to d_model (64)
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
        self.out_proj = nn.Linear(d_model + action_history_dim, out_dim)

    def forward(self, obs_stack):
        """
        obs_stack: [B, 33020] corresponding to (history_len * 3300) + action_history
        returns: [B, out_dim]
        """
        B = obs_stack.shape[0]
        
        # 1. Split Vision (11 channels) and Action History
        frame_dim = self.channels * self.H * self.W  # 11 * 15 * 20 = 3300
        vision_total_dim = self.history_len * frame_dim # 33000
        obs_feat = obs_stack[:, :vision_total_dim]     # [B, 33000]
        action_feat = obs_stack[:, vision_total_dim:]  # [B, 20]
        
        # 2. Reshape obs_feat to [B, T, C, H, W]
        vision_raw = obs_feat.reshape(B, self.history_len, self.channels, self.H, self.W)
        vision_feat = vision_raw.permute(0, 1, 3, 4, 2) # [B, T=10, H=15, W=20, C=11]
        
        # 3. Project Vision to d_model
        vision_proj = self.proj(vision_feat) # [B, 10, 15, 20, 64]
        
        # 4. Add 3D Positional Encodings
        device = vision_proj.device
        t_coords = torch.arange(self.history_len, device=device).view(self.history_len, 1, 1)
        y_coords = torch.arange(self.H, device=device).view(1, self.H, 1)
        x_coords = torch.arange(self.W, device=device).view(1, 1, self.W)
        
        pos_emb_v = self.pe_t(t_coords) + self.pe_y(y_coords) + self.pe_x(x_coords) # [10, 15, 20, 64]
        vision_with_pe = vision_proj + pos_emb_v.unsqueeze(0) # [B, 10, 15, 20, 64]
        vision_tokens = vision_with_pe.reshape(B, self.history_len, -1, self.d_model) # [B, 10, 300, 64]
        
        # 5. Flatten Spatio-Temporal dimensions to form sequence
        seqence_feat = vision_tokens.reshape(B, -1, self.d_model) # [B, 3000, 64]
        
        # 6. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 64]
        seqence_feat = torch.cat([cls_tokens, seqence_feat], dim=1).contiguous() # [B, 3001, 64]
        
        # 7. Pass through Transformer
        encoded_seq = self.transformer(seqence_feat) # [B, 3001, 64]
        
        # 8. Extract [CLS] token output
        cls_out = encoded_seq[:, 0, :] # [B, 64]
        
        # 9. Concatenate with Action History
        final_feat = torch.cat([cls_out, action_feat], dim=1) # [B, 64 + 20 = 84] (Assuming default config values, actually 32+20=52 for this run)
        
        # 10. Removed Projection to 256 dimensions to keep RL bottleneck compact (like projectc)
        # out = self.out_proj(final_feat) 
        
        return final_feat
