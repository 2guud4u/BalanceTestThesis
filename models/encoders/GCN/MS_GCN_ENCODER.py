"""
MS-GCN ST-GCN encoder wrapper.

Takes the ST-GCN backbone from MS-GCN (Model class in ms_gcn.py) and
strips the classification head (conv_out), so it outputs (B, filters, T)
features instead of class logits — ready for MS-TCN2 in your trainers.

    Input:  (B, T, J * C)   — flattened skeleton from PoseDataset
    Output: (B, filters, T) — feature sequence for MS-TCN2
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add MS-GCN to path so its internal imports resolve
_msgcn_dir = os.path.join(os.path.dirname(__file__), "MS-GCN")
if _msgcn_dir not in sys.path:
    sys.path.insert(0, _msgcn_dir)

from models.ms_gcn import Model as _STGCNBackbone


class MSGCNEncoder(nn.Module):
    """
    Wraps the MS-GCN `Model` (ST-GCN) as a feature encoder.
    Removes conv_out so output is (B, filters, T) instead of (B, num_class, T).

    Args:
        num_joints:   Number of skeleton joints (V).
        in_channels:  Channels per joint (C), e.g. 3 for (x, y, z).
        filters:      Hidden dim / output dim (= encoder.out_dim for MS-TCN2).
        dil:          Dilation rates for the 10 ST-GCN layers.
    """

    def __init__(
        self,
        graph_args,
        num_joints: int = 22,
        in_channels: int = 3,
        filters: int = 64,
        dil: list = [1,2,4,8,16,32,64,128,256,512],
    ):
        super().__init__()
        self.out_dim = filters  # encoder contract: trainer reads self.out_dim
        self.num_joints = num_joints
        self.in_channels = in_channels
        self.out_dim = filters  # encoder contract: trainer reads self.out_dim

        # Build the full ST-GCN model (num_class doesn't matter, we drop conv_out)
        self.backbone = _STGCNBackbone(
            graph_args=graph_args,
            in_channels=in_channels,
            num_class=2,       # placeholder — we never use conv_out
            dil=dil,
            filters=filters,
        )
        # Remove the classification head so we don't waste params
        del self.backbone.conv_out

    def forward(self, x):
        """
        Args:
            x: (B, T, J * C) flattened skeleton from PoseDataset.

        Returns:
            (B, filters, T) feature sequence for MS-TCN2.
        """
        B, T, D = x.shape
        V = self.num_joints
        C = self.in_channels
        M = 1  # single person

        assert D == V * C, f"Expected J*C = {V}*{C} = {V*C}, got {D}"

        # Reshape: (B, T, V*C) → (B, C, T, V, M=1)
        x = x.view(B, T, V, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, T, V)
        x = x.unsqueeze(-1)                      # (B, C, T, V, 1)

        # --- Run ST-GCN backbone (everything except conv_out) ---
        N = B
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N, M, V, C, T)
        x = x.view(N * M, V * C, T)
        x = self.backbone.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, M, C, T, V)
        x = x.view(N * M, C, T, V)

        x = self.backbone.conv_1x1(x)
        for gcn, importance in zip(self.backbone.st_gcn_networks, self.backbone.edge_importance):
            x, _ = gcn(x, self.backbone.A * importance)

        # V pooling


        x = F.avg_pool2d(x, kernel_size=(1, V))  # (N*M, filters, T, 1)

        # M pooling
        c = x.size(1)
        t = x.size(2)
        x = x.view(N, M, c, t).mean(dim=1).view(N, c, t)  # (B, filters, T)
        return x  # (B, D, T) — encoder contract
