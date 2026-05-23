import sys
# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MAMP')

from models.encoders.MY_SKELETONMAE.encoder import MAEFeatureEncoder

def initialize_encoder(d_cfg, e_cfg):
    # Initialize the MAE encoder
    mae = MAEFeatureEncoder(
        d_cfg["h5_key"],
        e_cfg["checkpoint_path"],  # ← checkpoint path from config
        e_cfg["config_path"],
    )
    print("✓ mae encoder loaded successfully")
    print(f"  Config: {mae.config['model']}")
    print(f"  Model parameters: {sum(p.numel() for p in mae.model.parameters())}")
    return mae