"""
TimeAligner — drop-in module for aligning encoder patch-level predictions
with frame-level labels.

Usage:
    aligner = TimeAligner("upsample_preds", num_classes=2)
    # or
    aligner = TimeAligner("downsample_labels", num_classes=2)

    preds_aligned, target_aligned, mask_aligned = aligner(predictions, target, mask)
"""

import torch
import torch.nn.functional as F


class TimeAligner:
    """
    Align temporal axes between predictions and labels.

    Strategies:
        "upsample_preds"   — F.interpolate predictions up to frame-level (default)
        "downsample_labels" — majority-vote labels down to patch-level
    """

    def __init__(self, strategy: str = "upsample_preds", num_classes: int = 2):
        if strategy not in ("upsample_preds", "downsample_labels"):
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                "Use 'upsample_preds' or 'downsample_labels'."
            )
        self.strategy = strategy
        self.num_classes = num_classes

    def __repr__(self):
        return f"TimeAligner(strategy='{self.strategy}', num_classes={self.num_classes})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, predictions, target, mask):
        """
        Args:
            predictions: list[(B, C, T_pred)]  — MS-TCN2 stage outputs
            target:      (B, T_frames)         — frame-level labels
            mask:        (B, 1, T_frames)       — frame-level validity mask

        Returns:
            predictions_aligned: list[(B, C, T)]
            target_aligned:      (B, T)
            mask_aligned:        (B, 1, T)
        """
        T_pred = predictions[-1].shape[-1]
        T_frames = target.shape[1]

        if T_pred == T_frames:
            return predictions, target, mask

        if self.strategy == "upsample_preds":
            return self._upsample_preds(predictions, target, mask, T_frames)
        else:
            return self._downsample_labels(predictions, target, mask, T_pred)

    # ------------------------------------------------------------------
    # Strategy 1: upsample predictions to frame-level
    # ------------------------------------------------------------------
    @staticmethod
    def _upsample_preds(predictions, target, mask, T_frames):
        predictions_aligned = [
            F.interpolate(p, size=T_frames, mode="nearest")
            for p in predictions
        ]
        return predictions_aligned, target, mask

    # ------------------------------------------------------------------
    # Strategy 2: downsample labels to patch-level (majority vote)
    # ------------------------------------------------------------------
    def _downsample_labels(self, predictions, target, mask, T_pred):
        target_ds, mask_ds = self._majority_vote(target, mask, T_pred)
        return predictions, target_ds, mask_ds

    def _majority_vote(self, target, mask, T_pred):
        """
        Reduce frame-level labels to patch-level by majority vote.

        Args:
            target: (B, T_frames) long
            mask:   (B, 1, T_frames) float {0,1}
            T_pred: number of temporal patches

        Returns:
            target_ds: (B, T_pred) long   (invalid patches → -100)
            mask_ds:   (B, 1, T_pred) float
        """
        B, T_frames = target.shape

        # Fallback: if not evenly divisible, use nearest interpolation
        if T_frames % T_pred != 0:
            target_ds = (
                F.interpolate(target.unsqueeze(1).float(), size=T_pred, mode="nearest")
                .squeeze(1)
                .long()
            )
            mask_ds = F.interpolate(mask.float(), size=T_pred, mode="nearest")
            return target_ds, mask_ds

        k = T_frames // T_pred  # frames per patch

        # Reshape into patches
        tgt = target.view(B, T_pred, k)                      # (B, T_pred, k)
        m = mask[:, 0, :].view(B, T_pred, k).float()         # (B, T_pred, k)

        # Clamp invalid labels to 0 (will be zeroed out by mask below)
        tgt_clamped = tgt.clone()
        tgt_clamped[m == 0] = 0

        # One-hot vote counting
        one_hot = F.one_hot(tgt_clamped, num_classes=self.num_classes).float()  # (B, TP, k, C)
        one_hot = one_hot * m.unsqueeze(-1)          # zero out invalid frames
        counts = one_hot.sum(dim=2)                  # (B, TP, C)
        target_ds = counts.argmax(dim=-1).long()     # (B, TP)

        # Mark fully-invalid patches with ignore_index
        valid_patch = m.sum(dim=2) > 0               # (B, TP)
        target_ds = torch.where(valid_patch, target_ds, torch.full_like(target_ds, -100))

        # Patch mask: valid if majority of frames are valid
        mask_ds = (m.mean(dim=-1) > 0.5).float().unsqueeze(1)  # (B, 1, TP)

        return target_ds, mask_ds
