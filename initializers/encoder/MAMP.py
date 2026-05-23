import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, "../models/encoders/MAMP")

from models.encoders.MY_MAMP.encoder import MAMPFeatureEncoder


def initialize_encoder(d_cfg, e_cfg):

    mamp = MAMPFeatureEncoder(
        d_cfg["h5_key"],
        e_cfg["checkpoint_path"],  # ← checkpoint path from config
        e_cfg["config_path"],
    )
    print("✓ MAMP encoder loaded successfully")
    print(f"  Config: {mamp.config['model']}")
    print(f"  Model parameters: {sum(p.numel() for p in mamp.model.parameters())}")

    return mamp
