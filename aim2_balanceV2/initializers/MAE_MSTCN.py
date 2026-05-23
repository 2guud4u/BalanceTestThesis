import sys
# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MAMP')

from models.encoders.MY_SKELETONMAE.encoder import MAEFeatureEncoder
from scripts.Trainers import Simple_MSTCN2_Trainer

def initialize_model(d_cfg, e_cfg, t_cfg, class_weights):
    # Initialize the MAE encoder
    mae = MAEFeatureEncoder(
        d_cfg["h5_key"],
        e_cfg["checkpoint_path"],  # ← checkpoint path from config
        e_cfg["config_path"],
        map_location=t_cfg["device"]
    )
    print("✓ mae encoder loaded successfully")
    print(f"  Config: {mae.config['model']}")
    print(f"  Model parameters: {sum(p.numel() for p in mae.model.parameters())}")

    # Initialize the MSTCN trainer with the MAE encoder
    trainer = Simple_MSTCN2_Trainer(
        encoder=mae,
        class_weights=class_weights,
        early_stop_patience=t_cfg["early_stop_patience"],
        early_stop_min_delta=t_cfg["early_stop_min_delta"],
        time_alignment=t_cfg["time_alignment"]
    )
    return trainer