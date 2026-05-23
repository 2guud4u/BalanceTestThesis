import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MAMP')

from models.encoders.MY_MAMP.encoder import MAMPFeatureEncoder
from scripts.Trainers import Simple_MSTCN2_Trainer

def initialize_model(d_cfg, e_cfg, t_cfg, class_weights):
    try:
        mamp = MAMPFeatureEncoder(
            d_cfg["h5_key"],
            '../models/encoders/MAMP/checkpoints/ntu120_xset.pth',
            '../models/encoders/MY_MAMP/pretrain_mamp_t120_layer8+5_mask90.yaml',
            map_location=t_cfg["device"]
        )
        print("✓ MAMP encoder loaded successfully")
        print(f"  Config: {mamp.config['model']}")
        print(f"  Model parameters: {sum(p.numel() for p in mamp.model.parameters())}")
    except Exception as e:
        print(f"✗ Error loading MAMP encoder: {e}")
        import traceback
        traceback.print_exc()



    trainer = Simple_MSTCN2_Trainer(
        encoder=mamp,
        class_weights=class_weights,
        early_stop_patience=t_cfg["early_stop_patience"],
        early_stop_min_delta=t_cfg["early_stop_min_delta"],
    )
    print(f"✓ Trainer initialized with MAMP encoder (out_dim={mamp.out_dim})")
    return trainer
    