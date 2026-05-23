from models.segmentors.MSTCNplus import MS_TCN2
def initalize_segementor(s_cfg, e_cfg):
    segementor = MS_TCN2(
            s_cfg.num_layers_PG,
            s_cfg.num_layers_R,
            s_cfg.num_R,
            s_cfg.num_f_maps,
            e_cfg.out_dim,
            s_cfg.num_classes,
        )
    return segementor