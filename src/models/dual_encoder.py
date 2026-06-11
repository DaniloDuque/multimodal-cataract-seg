import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from models.cross_attention import CrossAttentionFusion


## @brief Dual-encoder U-Net with bottleneck cross-attention fusion.
#
#  Architecture:
#    Encoder A (RGB)  ──┐
#                       ├─→ CrossAttentionFusion ─→ Decoder ─→ mask
#    Encoder B (Edge) ──┘
#
#  Skip connections in the decoder come from Encoder A only.
#  Both encoders share the same ResNet-34 architecture but no weights.
#
#  @param num_classes  Number of output segmentation classes.
#  @param embed_dim    Bottleneck channel count (must match encoder output).
#  @param num_heads    Number of cross-attention heads.
class DualEncoderUNet(nn.Module):
    def __init__(self, num_classes: int = 1, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()

        # Build two full U-Nets; we'll use their encoders separately
        # and share the decoder from model_rgb.
        self.model_rgb  = smp.Unet(
            encoder_name="resnet34", encoder_weights="imagenet",
            in_channels=3, classes=num_classes, activation=None,
        )
        self.model_edge = smp.Unet(
            encoder_name="resnet34", encoder_weights=None,
            in_channels=3, classes=num_classes, activation=None,
        )

        self.fusion = CrossAttentionFusion(embed_dim=embed_dim, num_heads=num_heads)

    ## @brief Forward pass.
    #  @param rgb   RGB image tensor  (B, 3, H, W).
    #  @param edge  Edge map tensor   (B, 3, H, W).
    #  @return Logits (B, num_classes, H, W).
    def forward(self, rgb: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        # --- Encode both streams ---
        feats_rgb  = self.model_rgb.encoder(rgb)    # list of feature maps
        feats_edge = self.model_edge.encoder(edge)

        # --- Fuse at bottleneck (last feature map in the list) ---
        bottleneck_rgb  = feats_rgb[-1]
        bottleneck_edge = feats_edge[-1]
        fused = self.fusion(bottleneck_rgb, bottleneck_edge)

        # Replace bottleneck in RGB feature list with fused features
        feats_fused = list(feats_rgb)
        feats_fused[-1] = fused

        # --- Decode using RGB skip connections ---
        decoder_out = self.model_rgb.decoder(*feats_fused)
        return self.model_rgb.segmentation_head(decoder_out)
