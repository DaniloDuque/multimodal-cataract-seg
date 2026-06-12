import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from models.cross_attention import CrossAttentionFusion


## @brief Dual-encoder U-Net with multi-scale cross-attention fusion.
#
#  Three improvements over the original single-bottleneck design:
#
#  1. Learnable scalar gate in CrossAttentionFusion — edge contribution starts
#     near-zero and grows only if it helps (see cross_attention.py).
#
#  2. Second cross-attention at encoder stage 4 (256-ch, 32×32 spatial) — injects
#     structural information one level above the bottleneck for richer context.
#
#  3. Edge encoder skip connections fused into the decoder — at each decoder stage,
#     the corresponding edge encoder feature is channel-concatenated with the RGB
#     skip and projected back to the original channel count via a 1×1 conv. This
#     propagates geometric boundary cues through the full expanding path.
#
#  Encoder feature index map for ResNet-34 / 512×512 input:
#    feats[0]: 3ch   512×512  (raw input passthrough)
#    feats[1]: 64ch  256×256
#    feats[2]: 64ch  128×128
#    feats[3]: 128ch  64×64
#    feats[4]: 256ch  32×32   ← stage-4 cross-attention
#    feats[5]: 512ch  16×16   ← bottleneck cross-attention
#
#  @param num_classes  Number of output segmentation classes.
#  @param embed_dim    Bottleneck channel count (512 for ResNet-34).
#  @param num_heads    Number of cross-attention heads.
class DualEncoderUNet(nn.Module):
    # ResNet-34 encoder channel sizes at each skip level used by SMP's decoder
    # (indices 1-4; index 0 is the raw input passthrough, index 5 is bottleneck)
    _SKIP_CHANNELS = [64, 64, 128, 256]   # feats[1..4]
    _STAGE4_DIM    = 256                   # feats[4] channel count
    _STAGE4_HEADS  = 8

    def __init__(self, num_classes: int = 1, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()

        self.model_rgb  = smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=num_classes, activation=None,
        )
        self.model_edge = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=num_classes, activation=None,
        )

        # Improvement 1 + (implicitly) 2: gated cross-attention at bottleneck
        self.fusion_bottleneck = CrossAttentionFusion(
            embed_dim=embed_dim, num_heads=num_heads
        )

        # Improvement 2: cross-attention at stage 4 (256ch)
        self.fusion_stage4 = CrossAttentionFusion(
            embed_dim=self._STAGE4_DIM, num_heads=self._STAGE4_HEADS
        )

        # Improvement 3: 1×1 projection convs that halve channels after edge-skip
        # concatenation at each decoder skip level (feats[1..4]).
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(c * 2, c, kernel_size=1, bias=False)
            for c in self._SKIP_CHANNELS
        ])

    ## @brief Forward pass.
    #  @param rgb   RGB image tensor  (B, 3, H, W).
    #  @param edge  Edge map tensor   (B, 3, H, W).
    #  @return Logits (B, num_classes, H, W).
    def forward(self, rgb: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        feats_rgb  = self.model_rgb.encoder(rgb)    # list of 6 feature maps
        feats_edge = self.model_edge.encoder(edge)

        # --- Improvement 2: fuse at stage 4 ---
        feats_rgb[4] = self.fusion_stage4(feats_rgb[4], feats_edge[4])

        # --- Improvement 1 (gate) + bottleneck fusion ---
        feats_rgb[5] = self.fusion_bottleneck(feats_rgb[5], feats_edge[5])

        # --- Improvement 3: blend edge skips into RGB skips (feats[1..4]) ---
        fused_feats = list(feats_rgb)
        for i, proj in enumerate(self.skip_projs):
            idx = i + 1   # feats[1], [2], [3], [4]
            fused_feats[idx] = proj(
                torch.cat([feats_rgb[idx], feats_edge[idx]], dim=1)
            )

        decoder_out = self.model_rgb.decoder(fused_feats)
        return self.model_rgb.segmentation_head(decoder_out)
