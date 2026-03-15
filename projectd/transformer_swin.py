import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention3D(nn.Module):
    """
    3D Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both shifted and non-shifted window schemes.
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # (Wt, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_t = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))  # 3, Wt, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wt*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wt*Wh*Ww, Wt*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wt*Wh*Ww, Wt*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wt*Wh*Ww, Wt*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1)  # Wt*Wh*Ww, Wt*Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wt*Wh*Ww, Wt*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x

class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, window_size=(5, 5, 5), shift_size=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size=self.window_size, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = self.window_size, self.shift_size

        shortcut = x
        x = self.norm1(x)

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=tuple(-i for i in shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        # Window partition
        x_windows = self.window_partition(shifted_x, window_size)  # nW*B, window_size**3, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=mask_matrix)

        # Window reverse
        attn_windows = attn_windows.view(-1, window_size[0], window_size[1], window_size[2], C)
        shifted_x = self.window_reverse(attn_windows, window_size, B, D, H, W)

        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=shift_size, dims=(1, 2, 3))
        else:
            x = shifted_x

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def window_partition(self, x, window_size):
        B, D, H, W, C = x.shape
        x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], C)
        return windows

    def window_reverse(self, windows, window_size, B, D, H, W):
        x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
        return x


class TinyVideoSwinEncoder(nn.Module):
    """
    Factorized 3D Spatio-Temporal Encoder using Tiny Video Swin Architecture.
    Significantly reduces FLOPs while maintaining strong inductive spatial-temporal biases.
    """
    def __init__(self, history_len=10, channels=11, H=15, W=20, d_model=64, num_layers=2, out_dim=256, action_history_dim=20):
        super().__init__()
        self.history_len = history_len
        self.channels = channels
        self.H = H
        self.W = W
        self.d_model = d_model
        
        # 1. 3D Patch Embedding (Treat each 1x1x1 pixel as a token, project 11 -> 64 channel)
        self.patch_embed = nn.Linear(channels, d_model)
        
        # 2. 3D Window Size Setup 
        # Since input is 10 x 15 x 20, a good window size that divides nicely:
        # window_size = (5, 5, 5) -> D: 10/5=2, H: 15/5=3, W: 20/5=4
        self.window_size = (5, 5, 5)
        self.shift_size = (self.window_size[0] // 2, self.window_size[1] // 2, self.window_size[2] // 2)

        # 3. Build Swin Transformer Blocks
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternating W-MSA and SW-MSA
            shift = (0, 0, 0) if (i % 2 == 0) else self.shift_size
            self.layers.append(
                SwinTransformerBlock3D(
                    dim=d_model, 
                    num_heads=1,  # Keep heads low to match d_model=64
                    window_size=self.window_size,
                    shift_size=shift
                )
            )
            
        # 4. Global average pooling
        self.norm = nn.LayerNorm(d_model)
        
        # 5. Output Projection (Merged with action history)
        self.out_proj = nn.Linear(d_model + action_history_dim, out_dim)
        
    def _compute_mask(self, x, shift_size):
        """
        Compute the attention mask for SW-MSA to avoid cross-boundary attention, 
        and apply Causal Mask on the Time dimension to prevent looking into the future.
        """
        B, D, H, W, C = x.shape
        window_size = self.window_size
        
        # 1. Shape Mask for Shifted Boundaries
        img_mask = torch.zeros((1, D, H, W, 1), device=x.device)
        cnt = 0
        for d in (slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None)):
            for h in (slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None)):
                for w in (slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None)):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
                    
        # Partition the mask
        mask_windows = self.layers[0].window_partition(img_mask, window_size)  # nW, window_size**3, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        # 2. Apply Causal Mask constraint on the Time dimension (D-axis of the window)
        # Inside each window_size**3 snippet, tokens are ordered (t, h, w)
        # T goes from 0 to W_t-1 (5).
        Wg = window_size[0] * window_size[1] * window_size[2]
        causal_mask = torch.zeros((Wg, Wg), device=x.device)
        
        # Calculate indices
        coords_t = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1) # 3, W_t*W_h*W_w
        
        # t_i > t_j is looking into the future. Mask it out!
        t_indices = coords_flatten[0] # Time coordinate for each token
        for i in range(Wg):
            for j in range(Wg):
                if t_indices[i] < t_indices[j]:  # Target token (i) looks at Source (j) which is strictly in the future
                    causal_mask[i, j] = -100.0
                    
        # Add causal mask shape constraints 
        attn_mask = attn_mask + causal_mask.unsqueeze(0)
        return attn_mask

    def forward(self, obs_stack):
        """
        obs_stack: [B, 33020]
        returns: [B, out_dim=256]
        """
        B = obs_stack.shape[0]
        
        frame_dim = self.channels * self.H * self.W  
        vision_total_dim = self.history_len * frame_dim 
        obs_feat = obs_stack[:, :vision_total_dim]     
        action_feat = obs_stack[:, vision_total_dim:]  
        
        vision_raw = obs_feat.view(B, self.history_len, self.channels, self.H, self.W)
        x = vision_raw.permute(0, 1, 3, 4, 2) # [B, T=10, H=15, W=20, C=11]
        
        # 1. Patch Embedding
        x = self.patch_embed(x) # [B, 10, 15, 20, 64]
        
        # 2. Determine Mask Arrays
        mask_shift = self._compute_mask(x, self.shift_size)
        mask_none = self._compute_mask(x, (0, 0, 0))
        
        # 3. Video Swin Blocks
        for i, layer in enumerate(self.layers):
            mask = mask_none if layer.shift_size == (0, 0, 0) else mask_shift
            x = layer(x, mask)
            
        # 4. Global Average Pooling over Spatial/Temporal dimensions: (10, 15, 20) -> (1, 1, 1)
        x = self.norm(x)
        # Flatten and Mean pooling
        x = x.view(B, -1, self.d_model) # [B, 3000, 64]
        x_pool = x.mean(dim=1) # [B, 64]
        
        # 5. Output Projection
        final_feat = torch.cat([x_pool, action_feat], dim=1) 
        out = self.out_proj(final_feat) 
        
        return out
