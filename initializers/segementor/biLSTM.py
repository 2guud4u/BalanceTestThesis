import sys

sys.path.insert(0, "..")
from models.segmentors.biLSTM import BiLSTM
from initializers.segementor._base import SegmentorWithLoss


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
    model = BiLSTM(
        dim=encoder.out_dim,
        num_classes=s_cfg["num_classes"],
        hidden_size=s_cfg["hidden_size"],
        num_layers=s_cfg["num_layers"],
        dropout=s_cfg["dropout"],
    )

    segmentor = SegmentorWithLoss(
        model=model,
        num_classes=s_cfg["num_classes"],
        class_weights=class_weights,
        lambda_smooth=lambda_smooth,
        time_alignment=time_alignment,
    )

    print(
        f"✓ BiLSTM segmentor initialized "
        f"(in_dim={encoder.out_dim}, classes={s_cfg['num_classes']}, "
        f"hidden={s_cfg['hidden_size']}, layers={s_cfg['num_layers']}, "
        f"dropout={s_cfg['dropout']}, lambda_smooth={lambda_smooth}, "
        f"alignment={time_alignment})"
    )
    if class_weights is not None:
        print(f"  Class weights: {class_weights.tolist()}")
    else:
        print("  Class weights: none (uniform)")

    return segmentor
