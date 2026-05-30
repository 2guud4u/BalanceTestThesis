"""
Identity (passthrough) encoder.

Performs no learned transformation — simply reshapes the input from
(B, T, F) to (B, F, T) so the segmentor receives the same channel-first
layout that all other encoders produce.

Use this when you want to feed raw skeleton features directly into the
segmentor without any representation learning.

    Input:  (B, T, F)   — flattened skeleton from PoseDataset
    Output: (B, F, T)   — feature sequence for the segmentor
"""

import torch.nn as nn


class IdentityEncoder(nn.Module):
    """
    No-op encoder that satisfies the encoder contract
    (nn.Module with ``out_dim`` attribute) while doing no computation.

    Args:
        feature_dim: Raw feature dimensionality (J * D from the dataset).
                     Exposed as ``self.out_dim`` so the segmentor can read it.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.out_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: (B, T, F) flattened skeleton from PoseDataset.

        Returns:
            (B, F, T) — unchanged features in channel-first layout.
        """
        return x.permute(0, 2, 1).contiguous()  # (B, T, F) -> (B, F, T)
