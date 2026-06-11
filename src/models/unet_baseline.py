import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


## @brief Standard U-Net baseline for unimodal segmentation.
#
#  Wraps segmentation_models_pytorch's U-Net so it can be used for:
#    - RGB-only input      (in_channels=3)
#    - Edge-only input     (in_channels=3)
#    - Early fusion        (in_channels=6, RGB+Edge concatenated)
#
#  Also exposes the bottleneck feature map via a forward hook so that
#  dual_encoder.py can access encoder features without re-running the encoder.
#
#  @param in_channels  Number of input channels (3 or 6).
#  @param num_classes  Number of output segmentation classes.
class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        self.model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
            activation=None,  # raw logits; sigmoid applied in loss/evaluate
        )
        # Hook storage for bottleneck features (used by dual_encoder)
        self._bottleneck: torch.Tensor | None = None
        self.model.encoder.register_forward_hook(self._save_bottleneck)

    def _save_bottleneck(self, module, input, output):
        # output is a list of feature maps; last one is the bottleneck
        self._bottleneck = output[-1]

    ## @brief Forward pass returning logits (B, num_classes, H, W).
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    ## @brief Returns bottleneck features (B, C, H', W') from last forward pass.
    def get_bottleneck(self) -> torch.Tensor:
        assert self._bottleneck is not None, "Call forward() first."
        return self._bottleneck
