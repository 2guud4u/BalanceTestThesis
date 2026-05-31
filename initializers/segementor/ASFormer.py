import sys

sys.path.insert(0, "..")
from models.segmentors.ASFormer import ASFormer
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
    model = ASFormer(
        dim=encoder.out_dim,
        num_classes=s_cfg["num_classes"],
        num_f_maps=s_cfg.get("num_f_maps", 64),
        num_layers=s_cfg.get("num_layers", 10),
        num_decoders=s_cfg.get("num_decoders", 3),
        r1=s_cfg.get("r1", 2),
        r2=s_cfg.get("r2", 2),
        channel_masking_rate=s_cfg.get("channel_masking_rate", 0.3),
    )

    segmentor = SegmentorWithLoss(
        model=model,
        num_classes=s_cfg["num_classes"],
        class_weights=class_weights,
        lambda_smooth=lambda_smooth,
        time_alignment=time_alignment,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(
        f"✓ ASFormer segmentor initialized "
        f"(in_dim={encoder.out_dim}, classes={s_cfg['num_classes']}, "
        f"num_f_maps={s_cfg.get('num_f_maps', 64)}, "
        f"layers={s_cfg.get('num_layers', 10)}, decoders={s_cfg.get('num_decoders', 3)}, "
        f"r1={s_cfg.get('r1', 2)}, r2={s_cfg.get('r2', 2)}, "
        f"channel_masking_rate={s_cfg.get('channel_masking_rate', 0.3)}, "
        f"lambda_smooth={lambda_smooth}, alignment={time_alignment})"
    )
    print(f"  Parameters: {n_params:,} total, {n_trainable:,} trainable")
    if class_weights is not None:
        print(f"  Class weights: {class_weights.tolist()}")
    else:
        print("  Class weights: none (uniform)")

    return segmentor
