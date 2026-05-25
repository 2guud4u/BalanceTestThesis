import sys
import os

# Add GCN model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../models/encoders/GCN"))

from models.encoders.GCN.MS_GCN_ENCODER import MSGCNEncoder


def initialize_encoder(d_cfg, e_cfg):
    encoder = MSGCNEncoder(
        graph_args=d_cfg["graph_args"],
        num_joints=d_cfg["num_joints"],
        in_channels=e_cfg["in_channels"],
        filters=e_cfg["filters"],
        dil=e_cfg["dil"],
    )
    print("✓ MS-GCN encoder loaded successfully")
    print(f"  Joints: {d_cfg['num_joints']}, in_channels: {e_cfg['in_channels']}, out_dim: {encoder.out_dim}")
    return encoder
