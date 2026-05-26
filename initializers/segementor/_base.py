"""
Shared segmentor wrapper.

`SegmentorWithLoss` is model-agnostic: it owns the CE + smoothing MSE loss
and the TimeAligner, and delegates the actual forward pass to whatever
`model` is passed in. Any segmentor whose forward returns shape
(S, B, C, T) — with S>=1 stages — can be wrapped by this class.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "..")
from scripts.time_align import TimeAligner


class SegmentorWithLoss(nn.Module):
    """
    Thin wrapper that owns the loss function and time aligner.
    Exposes compute_loss() so src/Trainer.py can call it without knowing the
    internals of the loss computation.
    """

    def __init__(
        self,
        model,
        num_classes,
        class_weights=None,
        lambda_smooth=0.15,
        time_alignment="downsample_labels",
    ):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.lambda_smooth = lambda_smooth
        self.aligner = TimeAligner(strategy=time_alignment, num_classes=num_classes)

        # Register class_weights as a buffer so it moves with .to(device) automatically
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        self.ce = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights)
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, x):
        """x: (B, D, T) encoder features → (S, B, C, T_pred) stage logits."""
        return self.model(x)

    def compute_loss(self, outputs, targets, mask):
        """
        Args:
            outputs: (S, B, C, T_pred)  — stacked stage logits from forward()
            targets: (B, T_frames)      — frame-level ground-truth labels
            mask:    (B, 1, T_frames)   — 1 for valid frames, 0 for padding

        Returns:
            loss:         scalar
            preds:        (B, T_aligned) predicted class indices (last stage)
            target_al:    (B, T_aligned) aligned targets
            mask_al:      (B, 1, T_aligned) aligned mask
        """
        # Convert leading stage dim to list for TimeAligner
        predictions = [outputs[s] for s in range(outputs.shape[0])]

        predictions_al, target_al, mask_al = self.aligner(predictions, targets, mask)

        loss = torch.tensor(0.0, device=outputs.device)
        for p in predictions_al:
            loss = loss + self.ce(
                p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                target_al.view(-1),
            )
            loss = loss + self.lambda_smooth * torch.mean(
                torch.clamp(
                    self.mse(
                        F.log_softmax(p[:, :, 1:], dim=1),
                        F.log_softmax(p.detach()[:, :, :-1], dim=1),
                    ),
                    min=0,
                    max=16,
                )
                * mask_al[:, :, 1:]
            )

        _, preds = torch.max(predictions_al[-1], dim=1)
        return loss, preds, target_al, mask_al

    def predict(self, outputs):
        """Return hard predictions from stacked stage outputs."""
        _, preds = torch.max(outputs[-1], dim=1)
        return preds
