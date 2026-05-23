import os
import sys
import json
import yaml
import torch
from scripts.Trainers import Simple_MSTCN2_Trainer
from scripts.Evaler import evaluate_folds
# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
save_dir = "/code/jjiang23/pathml/aim2_balanceV2/results/CAMMP_MAE_MSTCN/20260503_130451"
splits_path = "/code/jjiang23/pathml/aim2_balanceV2/data/splits.json"

# Load the config that was actually used during training (do NOT overwrite it).
config_path = os.path.join(save_dir, "config.yml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

d_cfg = config["dataloader_config"]
t_cfg = config["trainer_config"]
e_cfg = config["encoder_config"]

with open(splits_path, "r") as f:
    splits = json.load(f)

device = t_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

# -------------------------------------------------------------------
# Build encoder + trainer ONCE; reload weights per fold.
# -------------------------------------------------------------------
#test out MAMP feature extractor
import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MAMP')

from models.encoders.MY_SKELETONMAE.encoder import MAEFeatureEncoder

try:
    mae = MAEFeatureEncoder(
        d_cfg["h5_key"],
        e_cfg["checkpoint_path"],
        e_cfg["config_path"],
        map_location=t_cfg["device"]
    )
    print("✓ MAMP encoder loaded successfully")
    print(f"  Config: {mae.config['model']}")
    print(f"  Model parameters: {sum(p.numel() for p in mae.model.parameters())}")
except Exception as e:
    print(f"✗ Error loading MAMP encoder: {e}")
    import traceback
trainer = Simple_MSTCN2_Trainer(
    encoder=mae,
    early_stop_patience=t_cfg["early_stop_patience"],
    early_stop_min_delta=t_cfg["early_stop_min_delta"],
    early_stop_monitor=t_cfg["early_stop_monitor"],
)
print(f"✓ Trainer initialized (encoder out_dim={mae.out_dim})")

evaluate_folds(trainer, splits, d_cfg, t_cfg, save_dir, device)
