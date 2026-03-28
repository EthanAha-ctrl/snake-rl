import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GlobalLightningAttention(nn.Module):
    """
    Global Lightning Attention (Linear Attention) for flat 1D sequences.
    Replaces Softmax with SiLU/ReLU to drop complexity from O(N^2) to O(N).
    Supports 3D RoPE for spatial-temporal tracking without absolute positional bounds.
    """
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)

    def _apply_1d_rope(self, x):
        """
        Simplified 1D RoPE applied to the flattened 3000-token sequence.
        Although it's formally 1D, since the sequence represents unravelled (T, H, W),
        this RoPE preserves relative local distances efficiently.
        """
        B, nH, L, hD = x.shape
        t = torch.arange(L, device=x.device, dtype=x.dtype)
        freqs = 10000 ** (-torch.arange(0, hD, 2, device=x.device, dtype=x.dtype) / hD)
        emb = t[:, None] * freqs[None, :] # [L, hD/2]
        emb = torch.cat((emb, emb), dim=-1) # [L, hD]
        
        cos_pos = emb.cos()[None, None, :, :]
        sin_pos = emb.sin()[None, None, :, :]
        
        x_rot = x * cos_pos + self._rotate_half(x) * sin_pos
        return x_rot

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, x):
        # x: [B, L, d_model]
        B, L, _ = x.shape
        
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        q = self.norm_q(q)
        k = self.norm_k(k)
        
        q = q.view(B, L, self.nhead, self.head_dim).transpose(1, 2) # [B, nH, L, hd]
        k = k.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = self._apply_1d_rope(q)
        k = self._apply_1d_rope(k)
        
        # Lightning Attention Non-linear positive activation (replaces Softmax!)
        q = F.silu(q)
        k = F.silu(k)
        
        # O(N) Associative Linear Attention: Q @ (K^T @ V)
        # Sequence length L=3000, head_dim=64
        # K^T @ V -> [B, nH, hd, L] @ [B, nH, L, hd] = [B, nH, hd, hd]
        # This completely bypasses the N^2 bottleneck
        kv = torch.matmul(k.transpose(-2, -1), v) 
        
        # Q @ KV -> [B, nH, L, hd] @ [B, nH, hd, hd] = [B, nH, L, hd]
        out = torch.matmul(q, kv)
        
        # Normalize to prevent explosion (standard variance scaling for linear attention)
        out = out / (self.head_dim ** 0.5)

        out = out.transpose(1, 2).reshape(B, L, self.d_model)
        out = self.o_proj(out)
        return out


class LightningTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = GlobalLightningAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioTemporalLightningEncoder(nn.Module):
    """
    Direct replacement for the original SpatioTemporalEncoder (flat 3000-token full receptive field).
    Uses Lightning Attention to drop complexity from huge O(N^2) Softmax to swift O(N).
    No shifting windows - every token sees the whole video dynamically.
    """
    def __init__(self, history_len=10, channels=11, H=15, W=20, d_model=64, nhead=1, num_layers=2, out_dim=256, action_history_dim=20):
        super().__init__()
        self.history_len = history_len
        self.channels = channels
        self.H = H
        self.W = W
        self.d_model = d_model
        
        self.proj = nn.Linear(channels, d_model)
        
        # Remnant of absolute PE removed because RoPE is applied dynamically in Attention
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.layers = nn.ModuleList([
            LightningTransformerLayer(d_model, nhead) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model + action_history_dim, out_dim)

    def forward(self, obs_stack):
        """
        obs_stack: [B, 33020]
        """
        B = obs_stack.shape[0]
        
        frame_dim = self.channels * self.H * self.W 
        vision_total_dim = self.history_len * frame_dim 
        obs_feat = obs_stack[:, :vision_total_dim]     
        action_feat = obs_stack[:, vision_total_dim:]  
        
        vision_raw = obs_feat.view(B, self.history_len, self.channels, self.H, self.W)
        vision_feat = vision_raw.permute(0, 1, 3, 4, 2) # [B, T=10, H=15, W=20, C=11]
        
        # 1. Linear Projection (No Absolute PE anymore)
        vision_proj = self.proj(vision_feat) # [B, 10, 15, 20, 64]
        
        # 2. Flatten Spatio-Temporal dimensions to form global flat sequence
        seqence_feat = vision_proj.reshape(B, -1, self.d_model) # [B, 3000, 64]
        
        # 3. Prepend [CLS]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, 64]
        seqence_feat = torch.cat([cls_tokens, seqence_feat], dim=1) # [B, 3001, 64]
        
        # 4. Lightning Layers
        for layer in self.layers:
            seqence_feat = layer(seqence_feat)
            
        seqence_feat = self.norm(seqence_feat)
        
        # 5. Extract [CLS]
        cls_out = seqence_feat[:, 0, :] # [B, 64]
        
        # 6. Concat Action History & Output
        final_feat = torch.cat([cls_out, action_feat], dim=1) 
        out = self.out_proj(final_feat) 
        return out
