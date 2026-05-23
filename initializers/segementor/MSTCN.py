import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "..")
from models.segmentors.MSTCNplus import MS_TCN2
from scripts.time_align import TimeAligner


class SegmentorWithLoss(nn.Module):
    """
    Thin wrapper around MS_TCN2 that owns the loss function and time aligner.
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
        # MS_TCN2.forward stacks stages on dim 0; convert to list for TimeAligner
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


def initialize_segmentor(
    s_cfg,
    encoder,
    class_weights=None,
    lambda_smooth=0.15,
    time_alignment="downsample_labels",
):
    """
    Args:
        s_cfg:          segmentor config dict (from segmentor YAML)
        encoder:        instantiated encoder — provides encoder.out_dim
        class_weights:  optional (num_classes,) tensor for weighted CE loss
        lambda_smooth:  smoothing loss coefficient (from trainer config)
        time_alignment: 'upsample_preds' or 'downsample_labels' (from trainer config)
    """
    model = MS_TCN2(
        s_cfg["num_layers_PG"],
        s_cfg["num_layers_R"],
        s_cfg["num_R"],
        s_cfg["num_f_maps"],
        encoder.out_dim,
        s_cfg["num_classes"],
    )

    segmentor = SegmentorWithLoss(
        model=model,
        num_classes=s_cfg["num_classes"],
        class_weights=class_weights,
        lambda_smooth=lambda_smooth,
        time_alignment=time_alignment,
    )

    print(
        f"✓ MS-TCN2 segmentor initialized "
        f"(in_dim={encoder.out_dim}, classes={s_cfg['num_classes']}, "
        f"lambda_smooth={lambda_smooth}, alignment={time_alignment})"
    )
    if class_weights is not None:
        print(f"  Class weights: {class_weights.tolist()}")
    else:
        print("  Class weights: none (uniform)")

    return segmentor
