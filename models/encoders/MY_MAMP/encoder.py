"""
Encoder-only feature extraction for skeleton data using pretrained MAMP models.

Usage:
    extractor = MAMPEncoder(skeleton_type, checkpoint_path, config_path)
    features = extractor(skeleton_data)  # (B, T, J*3) -> (B, D, T_patches)
"""

import contextlib
import os
import sys
from typing import Union, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add project root to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.skeletonMapping import convertBatchVideoMPtoNTU, convertBatchVideoMBtoNTU, _is_torch

# NTU-convention reference: average spineMid (joint 1) → shoulderMid (joint 20) bone length.
# Used to rescale MotionBert (~0.17) and world MediaPipe (~0.25) inputs into the
# range MAMP was pretrained on. Measured directly from NTU120_XSub.npz:
#   spine mean=0.1958, median=0.1968, std=0.0100 (n=100 samples, body 0, valid frames).
# MAMP pretraining did translation only (no scale normalization), so this matches the
# raw Kinect meter scale the encoder saw during pretraining.
NTU_REF_SPINE_LEN = 0.196


def _scale_normalize_to_ntu(x, ntu_ref_bone_len: float = NTU_REF_SPINE_LEN):
    """
    Rescale (B, T, 25, 3) skeleton so the mean spineMid→shoulderMid bone length
    matches NTU. One scalar per sample; preserves geometry.
    """
    bone = x[:, :, 20, :] - x[:, :, 1, :]                         # (B, T, 3)
    if _is_torch(bone):
        lengths = bone.norm(dim=-1)                               # (B, T)
        valid = (lengths > 1e-6).float()
        scale = (lengths * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        scale = scale.clamp(min=1e-6)
        factor = (ntu_ref_bone_len / scale).view(-1, 1, 1, 1)
    else:
        lengths = ((bone ** 2).sum(-1)) ** 0.5
        valid = (lengths > 1e-6).astype(lengths.dtype)
        denom = valid.sum(axis=1)
        denom[denom < 1] = 1
        scale = (lengths * valid).sum(axis=1) / denom
        scale = scale.clip(min=1e-6)
        factor = (ntu_ref_bone_len / scale).reshape(-1, 1, 1, 1)
    return x * factor

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


class MAMPEncoder(nn.Module):
    """
    PyTorch nn.Module wrapper around the ENCODER of a pretrained MAMP model.

    Input (default):
        x: (B, T, J*3) flattened skeleton sequence
           e.g. MediaPipe 33 joints -> (B, T, 99)

    Output:
        features: patch-level temporal sequence
        shape depends on pool_spatial:
            "flatten": (B, T_patches, S * D)
            "mean":    (B, T_patches, D)      [recommended]
            "max":     (B, T_patches, D)
            "mean+max": (B, T_patches, 2*D)   [richer features]
            "none":    (B, T_patches, S, D)   [keeps spatial]
        where:
            T_patches = T // temporal_patch_size
            S         = number of spatial tokens per temporal patch (joints)
            D         = encoder hidden dim
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

    def train(self, mode: bool = True):
        """
        Override train() so a frozen encoder stays in eval mode even if
        a parent module calls .train().
        """
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

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
            x_mamp: (B, 3, T, 25, 1)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, J*3), got {tuple(x.shape)}")

        B, T, C_flat = x.shape
        if C_flat % 3 != 0:
            raise ValueError(f"Feature dim {C_flat} is not divisible by 3")

        J = C_flat // 3
        x = x.view(B, T, J, 3)
        
        if self.skeleton_type == "camera_mp_cropped_iou":
            raise ValueError(
                "camera_mp_cropped_iou is image-normalized 2D + uncorrelated pseudo-depth "
                "(x,y in [0,1] image fractions; z in a different unit ~4x larger). "
                "It is NOT a Euclidean 3D skeleton and is incompatible with NTU-pretrained "
                "MAMP/MAE encoders. Use world_mp_cropped_iou or motionBert_cropped_iou "
                "with this encoder, or pass camera_mp to MS-GCN instead."
            )
        elif self.skeleton_type == "world_mp_cropped_iou":
            if J != 33:
                raise ValueError(f"For skeleton_type='mp', expected 33 joints, got {J}")
            x = convertBatchVideoMPtoNTU(x)            # (B, T, 25, 3)
        elif self.skeleton_type == "motionBert":
            print("Converting MotionBert skeleton to NTU format...")
            x = convertBatchVideoMBtoNTU(x)

        elif self.skeleton_type == "ntu":
            if J != 25:
                raise ValueError(f"For skeleton_type='ntu', expected 25 joints, got {J}")
        else:
            raise ValueError(f"Unsupported skeleton_type: {self.skeleton_type}")

        # Rescale non-NTU sources into MAMP's pretraining scale.
        if self.skeleton_type != "ntu":
            x = _scale_normalize_to_ntu(x)

        x = self._seq_translate_single_body(x)      # (B, T, 25, 3)
        x = x.permute(0, 3, 1, 2).contiguous()      # (B, 3, T, 25)
        x = x.unsqueeze(-1)                         # (B, 3, T, 25, 1)
        return x

    def _latent_to_temporal_sequence(self, latent: torch.Tensor, T_in: int) -> torch.Tensor:
        """
        Convert encoder tokens to a temporal sequence.

        latent: (B, L, D) or (B, 1+L, D) if CLS token exists
        T_in: original number of frames before patching

        Returns:
            if pool_spatial == "flatten": (B, T_patches, S*D)
            if pool_spatial == "mean":    (B, T_patches, D)
            if pool_spatial == "max":     (B, T_patches, D)
            if pool_spatial == "none":    (B, T_patches, S, D)
        """
        B, L, D = latent.shape
        T_patches = T_in // self.temporal_patch_size
        if T_patches <= 0:
            raise ValueError(f"Invalid T_patches={T_patches} from T_in={T_in}")

        # Remove CLS token if present
        if (L - 1) > 0 and ((L - 1) % T_patches == 0):
            latent = latent[:, 1:, :]
            L = latent.size(1)

        if L % T_patches != 0:
            raise ValueError(
                f"Cannot reshape latent of length {L} into T_patches={T_patches}. "
                f"Try mask_ratio=0.0 and verify input length / patch config."
            )

        S = L // T_patches
        latent = latent.view(B, T_patches, S, D)

        if self.pool_spatial == "flatten":
            # (B, T_patches, S, D) -> (B, T_patches, S*D)
            return latent.reshape(B, T_patches, S * D)
        elif self.pool_spatial == "mean":
            # (B, T_patches, S, D) -> (B, T_patches, D)
            # Average across all spatial tokens (joints)
            return latent.mean(dim=2)
        elif self.pool_spatial == "max":
            # (B, T_patches, S, D) -> (B, T_patches, D)
            # Take max across all spatial tokens
            return latent.max(dim=2).values
        elif self.pool_spatial == "mean+max":
            # Concatenate mean and max for richer representation
            # (B, T_patches, 2*D)
            mean_pool = latent.mean(dim=2)
            max_pool = latent.max(dim=2).values
            return torch.cat([mean_pool, max_pool], dim=-1)
        elif self.pool_spatial == "none":
            # Keep spatial dimension: (B, T_patches, S, D)
            return latent
        else:
            raise ValueError(f"Unknown pool_spatial={self.pool_spatial}. "
                           f"Choose from: 'flatten', 'mean', 'max', 'mean+max', 'none'")

    def forward(
        self,
        x: Union[np.ndarray, torch.Tensor],
        mask_ratio: float = 0.0,
        already_mamp_format: bool = False,
        motion_aware_tau: float = 0.0,
    ) -> torch.Tensor:
        """
        Forward through the frozen or trainable MAMP encoder.

        Args:
            x:
                if already_mamp_format=False:
                    (B, T, J*3)
                if already_mamp_format=True:
                    (B, 3, T, 25, 1)
            mask_ratio:
                use 0.0 for downstream feature extraction
            already_mamp_format:
                skip transform_input if x is already in MAMP format
            motion_aware_tau:
                only used by some MAMP variants; safe default is 0.0

        Returns:
            torch.Tensor of shape determined by pool_spatial
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if not already_mamp_format:
            # (B,T,J*3) -> (B,3,T,25,1)
            x = self.transform_input(x)
        else:
            if x.dim() != 5:
                raise ValueError(f"Expected MAMP-format input (B,3,T,25,1), got {tuple(x.shape)}")

        x = x.to(next(self.model.parameters()).device)
        T_in = x.shape[2]

        # MAMP Transformer's forward_encoder expects imgs: (NM, T, V, 3)
        # where NM = N*M. We only support single-person (M=1) right now.
        if x.shape[-1] != 1:
            raise ValueError(f"Expected M=1 in last dim, got M={x.shape[-1]}")
        x_enc = x.squeeze(-1).permute(0, 2, 3, 1).contiguous()  # (B,T,V,3)

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            try:
                latent, mask, ids_restore = self.model.forward_encoder(x_enc, mask_ratio, motion_aware_tau)
            except TypeError:
                latent, mask, ids_restore = self.model.forward_encoder(x_enc, mask_ratio)

            print(f"Encoder output latent shape: {latent.shape}, feature in: {x_enc.shape}")
            features = self._latent_to_temporal_sequence(latent, T_in)  # (B, T_patches, D) for pool_spatial='mean'
            print(f"Features after pooling: {features.shape}")

        # MS-TCN2 expects (B, D, T)
        if features.dim() == 3:
            features = features.permute(0, 2, 1).contiguous()  # (B, D, T_patches)
        elif features.dim() == 4:
            # If user chooses pool_spatial='none', keep spatial tokens but still put D in channel dim:
            # (B, T_patches, S, D) -> (B, D, T_patches, S)
            features = features.permute(0, 3, 1, 2).contiguous()

        return features

    def extract_features_numpy(
        self,
        x: Union[np.ndarray, torch.Tensor],
        mask_ratio: float = 0.0,
        already_mamp_format: bool = False,
    ) -> np.ndarray:
        """
        Convenience helper if you want NumPy features on disk.
        """
        feats = self.forward(x, mask_ratio=mask_ratio, already_mamp_format=already_mamp_format)
        return feats.detach().cpu().numpy()
    

class MAMPFeatureEncoder(nn.Module):
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
            x_mamp: (B, T, 25, 3)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected (B, T, J*3), got {tuple(x.shape)}")

        B, T, C_flat = x.shape
        if C_flat % 3 != 0:
            raise ValueError(f"Feature dim {C_flat} is not divisible by 3")

        J = C_flat // 3
        x = x.view(B, T, J, 3)

        if self.skeleton_type == "camera_mp_cropped_iou":
            raise ValueError(
                "camera_mp_cropped_iou is image-normalized 2D + uncorrelated pseudo-depth "
                "and is not compatible with NTU-pretrained MAMP/MAE encoders. "
                "Use world_mp_cropped_iou or motionBert_cropped_iou here."
            )
        elif self.skeleton_type == "world_mp_cropped_iou":
            if J != 33:
                raise ValueError(f"For skeleton_type='mp', expected 33 joints, got {J}")
            x = convertBatchVideoMPtoNTU(x)
        elif self.skeleton_type == "motionBert_cropped_iou":
            if J != 17:
                raise ValueError(f"skeleton_type='motionBert_cropped_iou': expected 17 joints, got {J}")
            x = convertBatchVideoMBtoNTU(x)
        elif self.skeleton_type == "ntu":
            if J != 25:
                raise ValueError(f"For skeleton_type='ntu', expected 25 joints, got {J}")
        else:
            raise ValueError(f"Unsupported skeleton_type: {self.skeleton_type}")

        # Rescale non-NTU sources into MAMP's pretraining scale.
        if self.skeleton_type != "ntu":
            x = _scale_normalize_to_ntu(x)

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
        x_mamp = x.contiguous()

        ctx = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with ctx:
            latent, mask, ids_restore = self.model.forward_encoder(
                x_mamp,
                mask_ratio=0.0,
                motion_aware_tau=0.0,
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
