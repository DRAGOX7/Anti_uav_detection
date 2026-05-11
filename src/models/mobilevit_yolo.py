from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn


class MobileViTFeatures(nn.Module):
    """MobileViT feature extractor that returns a list of multi-scale feature maps.

    This is designed to be used inside an Ultralytics model YAML via `parse_model`.
    It relies on `timm` for the MobileViT implementation and ImageNet-pretrained weights.

    Notes
    -----
    - The output is a list[Tensor] of length 3.
    - Each Tensor is a feature map at a different spatial resolution.
    """

    def __init__(
        self,
        variant: str = "mobilevit_s",
        pretrained: bool = True,
        out_indices: Sequence[int] = (1, 2, 3),
    ) -> None:
        super().__init__()

        try:
            import timm  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing dependency 'timm'. Install it with: pip install timm"
            ) from e

        # timm features_only returns a FeatureListNet that outputs a list of feature maps
        self.variant = str(variant)
        self.pretrained = bool(pretrained)
        self.out_indices = tuple(int(i) for i in out_indices)

        self.backbone = timm.create_model(
            self.variant,
            pretrained=self.pretrained,
            features_only=True,
            out_indices=self.out_indices,
        )

        # Feature channels for convenience/debugging
        try:
            self.out_channels: Tuple[int, ...] = tuple(self.backbone.feature_info.channels())
        except Exception:  # pragma: no cover
            self.out_channels = ()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.backbone(x)
        if not isinstance(feats, (list, tuple)):
            raise TypeError(f"Expected backbone features list/tuple, got {type(feats)}")
        feats = list(feats)
        if len(feats) != 3:
            raise ValueError(
                f"Expected 3 feature maps for YOLO Detect head, got {len(feats)}. "
                f"Check out_indices={self.out_indices} for variant={self.variant}."
            )
        return feats
