from models.segmentors.MSTCNplus import MS_TCN2


def initialize_segmentor(s_cfg, encoder):
    """
    Args:
        s_cfg:   segmentor config dict (from segmentor YAML)
        encoder: instantiated encoder object — used to read encoder.out_dim
                 so the TCN input size matches the encoder's output dimension
    """
    segmentor = MS_TCN2(
        s_cfg["num_layers_PG"],
        s_cfg["num_layers_R"],
        s_cfg["num_R"],
        s_cfg["num_f_maps"],
        encoder.out_dim,
        s_cfg["num_classes"],
    )
    print(f"✓ MS-TCN2 segmentor initialized (in_dim={encoder.out_dim}, classes={s_cfg['num_classes']})")
    return segmentor