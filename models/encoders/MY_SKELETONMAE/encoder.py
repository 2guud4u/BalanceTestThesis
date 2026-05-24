"""
Encoder-only feature extraction for skeleton data using pretrained MAMP models.

Usage:
    extractor = EncoderFeatureExtractor(checkpoint_path, config_path, device='cuda')
    features = extractor.extract_features(skeleton_data)  # (B,C,T,V,M) -> (B, L, hidden_dim)
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Union, Optional, Tuple
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.skeletonMapping import convertBatchVideoMPtoNTU, convertBatchVideoMBtoNTU
# Add MAMP directory to path so model_mamp can be imported
MAMP_DIR = os.path.join(os.path.dirname(__file__), '..', 'MAMP')
if MAMP_DIR not in sys.path:
    sys.path.insert(0, MAMP_DIR)

def import_class(name):
    """Import a class from a string path"""
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


import contextlib
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml


class MAEFeatureEncoder(nn.Module):
    """
    Wrap a pretrained MAMP model so it returns patch-level features for MS-TCN++.

    Input:
        x: (B, T, V, 3)  # one person, one skeleton sequence
    Output:
        feats: (B, D, T_patches)
    """
    def __init__(
        self,
        skeleton_type: str,
        checkpoint_path: str,
        config_path: str,
        freeze: bool = True,
        pool_spatial: str = "mean",   # "flatten", "mean", "max", "mean+max", "none"
        map_location: str = "cpu",
    ):
        super().__init__()
        
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.config = config
        self.skeleton_type = skeleton_type
        self.freeze = freeze
        self.pool_spatial = pool_spatial

        Model = import_class(config["model"])
        self.model = Model(**config["model_args"])

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)

        ma = config.get("model_args", {})
        # In the provided MAMP configs, temporal patch size is stored as `t_patch_size`
        self.temporal_patch_size = ma.get("temporal_patch_size", ma.get("t_patch_size"))
        if self.temporal_patch_size is None:
            raise KeyError(
                "Missing temporal patch size in config['model_args']. "
                f"Expected 't_patch_size' (or 'temporal_patch_size'). Available keys: {list(ma.keys())}"
            )
        
        # Extract output dimension from config (e.g., "dim_feat" in MAMP configs)
        self.out_dim = ma.get("dim_feat", ma.get("hidden_dim", ma.get("dim", None)))
        if self.out_dim is None:
            raise KeyError(
                "Missing output dimension in config['model_args']. "
                f"Expected 'dim_feat', 'hidden_dim', or 'dim'. Available keys: {list(ma.keys())}"
            )

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

        print(f"Loaded MAMP checkpoint from: {checkpoint_path}")
        print(f"Model class: {config['model']}")
        print(f"temporal_patch_size: {self.temporal_patch_size}")
        print(f"out_dim: {self.out_dim}")
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
    def _seq_translate_single_body(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, J, 3) in NTU joint layout
        Return:
            translated x relative to joint-2 of first valid frame
        """
        B, T, J, _ = x.shape
        flat = x.view(B, T, -1)
        non_empty = (flat.abs().sum(dim=-1) != 0)

        first_idx = torch.zeros(B, dtype=torch.long, device=x.device)
        for b in range(B):
            valid = torch.nonzero(non_empty[b], as_tuple=False)
            if valid.numel() > 0:
                first_idx[b] = valid[0, 0]
            else:
                # If sequence is all-zero, just use frame 0 (will be all zeros anyway)
                first_idx[b] = 0

        # NTU joint-2 -> index 1 in 0-based indexing
        origin = x[torch.arange(B, device=x.device), first_idx, 1:2, :]  # (B,1,3)
        origin = origin.unsqueeze(1)  # (B,1,1,3)
        return x - origin

    def transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform flat skeleton input to MAMP format.

        Input:
            x: (B, T, J*3)

        Output:
            x_mae: (B, T, 25, 3)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, J*3), got {tuple(x.shape)}")

        B, T, C_flat = x.shape
        if C_flat % 3 != 0:
            raise ValueError(f"Feature dim {C_flat} is not divisible by 3")

        J = C_flat // 3
        x = x.view(B, T, J, 3)

        if self.skeleton_type == "camera_mp_cropped_iou" or self.skeleton_type == "world_mp_cropped_iou":
            if J != 33:
                raise ValueError(f"For skeleton_type='mp', expected 33 joints, got {J}")
            # Must exist in your codebase:
            # input:  (B, T, 33, 3)
            # output: (B, T, 25, 3)
            x = convertBatchVideoMPtoNTU(x)
        if self.skeleton_type == "motionBert_cropped_iou":
            x = convertBatchVideoMBtoNTU(x)
        elif self.skeleton_type == "ntu":
            if J != 25:
                raise ValueError(f"For skeleton_type='ntu', expected 25 joints, got {J}")
        else:
            raise ValueError(f"Unsupported skeleton_type: {self.skeleton_type}")

        x = self._seq_translate_single_body(x)      # (B, T, 25, 3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, V, 3) or flat input that transform_input can convert
        returns: (B, D, T_patches)
        """
        if x.dim() != 4:
            x = self.transform_input(x)   # (B, T, V, 3)

        B, T, V, C = x.shape
        if C != 3:
            raise ValueError(f"Expected 3 coordinate channels, got {C}")

        # IMPORTANT: forward_encoder expects (B, T, V, 3), not 5D
        x_mae = x.contiguous()

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            latent, mask, ids_restore = self.model.forward_encoder(
                x_mae,
                mask_ratio=0.0,
            )

        TP = self.model.joints_embed.t_grid_size
        VP = self.model.joints_embed.grid_size
        D = latent.shape[-1]

        if latent.shape[1] != TP * VP:
            raise ValueError(
                f"Unexpected latent length {latent.shape[1]}; expected {TP * VP}."
            )

        latent = latent.view(B, TP, VP, D).mean(dim=2)   # (B, TP, D)
        latent = latent.permute(0, 2, 1).contiguous()    # (B, D, TP)
        return latent
