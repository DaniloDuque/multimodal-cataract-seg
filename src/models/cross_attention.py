import torch
import torch.nn as nn


## @brief Cross-attention fusion module for the dual-encoder bottleneck.
#
#  Reshapes spatial feature maps to sequences, applies multi-head cross-attention
#  (Q from RGB encoder, K/V from edge encoder), then reshapes back.
#
#  Input shapes:  x_rgb, x_edge — (B, C, H, W)
#  Output shape:  (B, C, H, W)  — fused RGB features enriched with edge structure
#
#  @param embed_dim  Channel dimension C (must match bottleneck channels).
#  @param num_heads  Number of attention heads.
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    ## @brief Fuse RGB and edge bottleneck features via cross-attention.
    #  @param x_rgb   RGB encoder bottleneck  (B, C, H, W).
    #  @param x_edge  Edge encoder bottleneck (B, C, H, W).
    #  @return Fused tensor (B, C, H, W).
    def forward(self, x_rgb: torch.Tensor, x_edge: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_rgb.shape

        # (B, C, H, W) → (B, H*W, C)
        q = x_rgb.flatten(2).transpose(1, 2)
        k = x_edge.flatten(2).transpose(1, 2)
        v = k

        fused, _ = self.attn(q, k, v)          # (B, H*W, C)
        fused = self.norm(fused + q)            # residual + layer norm

        # (B, H*W, C) → (B, C, H, W)
        return fused.transpose(1, 2).reshape(B, C, H, W)
