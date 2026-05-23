#test out MAMP feature extractor
import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MS-GCN')

from models.encoders.GCN.MS_GCN_ENCODER import MSGCNEncoder

def initialize_model(d_cfg, e_cfg, t_cfg, class_weights):
    encoder = MSGCNEncoder(
        d_cfg["graph_args"],
        num_joints=d_cfg["num_joints"],
        in_channels=e_cfg["in_channels"],
        filters=e_cfg["filters"],       
        dil=e_cfg["dil"]
    )

    from scripts.Trainers import Simple_MSTCN2_Trainer


    trainer = Simple_MSTCN2_Trainer(
        encoder=encoder,
        class_weights=class_weights,
        early_stop_patience=t_cfg["early_stop_patience"],
        early_stop_min_delta=t_cfg["early_stop_min_delta"],
        early_stop_monitor=t_cfg["early_stop_monitor"],
    )
    return trainer