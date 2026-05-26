"""
BiLSTM segmentor.

Drop-in replacement for MS_TCN2 in this codebase: same input/output contract
so it can be wrapped by `SegmentorWithLoss` without any other change.

    Input:  (B, D, T)         — encoder features (D = encoder.out_dim)
    Output: (S=1, B, C, T)    — single-stage frame-level logits

The leading stage dimension (S=1) keeps the tensor shape compatible with the
multi-stage interface that `SegmentorWithLoss.compute_loss` iterates over.
A BiLSTM has no refinement stages, so S is always 1.
"""

import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    """
    Bidirectional LSTM frame classifier.

    Args:
        dim:          input feature dim per frame (== encoder.out_dim).
        num_classes:  number of output classes.
        hidden_size:  LSTM hidden dim per direction. Output channel of LSTM
                      is 2 * hidden_size before the linear classifier.
        num_layers:   number of stacked BiLSTM layers.
        dropout:      dropout between stacked LSTM layers AND before the
                      classifier head. Set 0.0 to disable.
    """

    def __init__(
        self,
        dim: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # PyTorch only applies inter-layer dropout when num_layers > 1.
        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: (B, D, T) encoder feature sequence.

        Returns:
            (1, B, C, T) — single-stage logits, matching the multi-stage
            shape contract used by SegmentorWithLoss.
        """
        # (B, D, T) -> (B, T, D) for batch_first LSTM
        x = x.transpose(1, 2).contiguous()

        out, _ = self.lstm(x)                   # (B, T, 2*H)
        out = self.dropout(out)
        out = self.classifier(out)              # (B, T, C)

        # (B, T, C) -> (B, C, T)
        out = out.transpose(1, 2).contiguous()

        # Add stage dim to match (S, B, C, T) contract
        return out.unsqueeze(0)
