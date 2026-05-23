import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MS-GCN')

from models.encoders.GCN.MS_GCN_ENCODER import MSGCNEncoder

def initialize_model(d_cfg, e_cfg):
    gcn = MSGCNEncoder(
        d_cfg["graph_args"],
        num_joints=d_cfg["num_joints"],
        in_channels=e_cfg["in_channels"],
        filters=e_cfg["filters"],       
        dil=e_cfg["dil"]
    )
    
    return gcn
