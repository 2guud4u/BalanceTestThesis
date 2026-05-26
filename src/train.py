import os
import argparse
import json
import yaml
import importlib.util
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# go back one level to import
import sys

sys.path.append("..")

from data.PoseDataset import PoseDataset
from Trainer import Trainer


# =========================================================
# Dynamic import from file path
# =========================================================
def load_module_from_path(file_path):
    """Load a Python module from an absolute file path."""
    spec = importlib.util.spec_from_file_location("initializer_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =========================================================
# Parse command line arguments
# =========================================================
parser = argparse.ArgumentParser(
    description="Generic skeleton action segmentation trainer"
)

parser.add_argument(
    "--encoder",
    type=str,
    required=True,
    help="Path to encoder config YAML file",
)

parser.add_argument(
    "--data",
    type=str,
    required=True,
    help="Path to dataloader config YAML file",
)

parser.add_argument(
    "--trainer",
    type=str,
    default="../configs/trainer/downsamp.yml",
    help="Path to trainer config YAML file",
)

parser.add_argument(
    "--segmentor",
    type=str,
    required=True,
    help="Path to segmentor config YAML file",
)

parser.add_argument(
    "--splits",
    type=str,
    default="/code/jjiang23/pathml/aim2_balanceV2/data/splits.json",
    help="Path to splits JSON file",
)

args = parser.parse_args()


# =========================================================
# Resolve config paths
# =========================================================
encoder_cfg_path = os.path.abspath(args.encoder)
data_cfg_path = os.path.abspath(args.data)
trainer_cfg_path = os.path.abspath(args.trainer)
segmentor_cfg_path = os.path.abspath(args.segmentor)
splits_path = os.path.abspath(args.splits)

print(f"Loading encoder config from: {encoder_cfg_path}")
print(f"Loading data config from: {data_cfg_path}")
print(f"Loading trainer config from: {trainer_cfg_path}")
print(f"Loading segmentor config from: {segmentor_cfg_path}")


# =========================================================
# Load YAML configs
# =========================================================
with open(encoder_cfg_path, "r") as f:
    e_cfg = yaml.safe_load(f)

with open(data_cfg_path, "r") as f:
    d_cfg = yaml.safe_load(f)

with open(trainer_cfg_path, "r") as f:
    t_cfg = yaml.safe_load(f)

with open(segmentor_cfg_path, "r") as f:
    s_cfg = yaml.safe_load(f)


# =========================================================
# Combine configs into one giant config
# =========================================================
config = {
    "encoder": e_cfg,
    "data": d_cfg,
    "trainer": t_cfg,
    "segmentor": s_cfg,
    "paths": {
        "encoder_cfg_path": encoder_cfg_path,
        "data_cfg_path": data_cfg_path,
        "trainer_cfg_path": trainer_cfg_path,
        "segmentor_cfg_path": segmentor_cfg_path,
        "splits_path": splits_path,
    },
}


# =========================================================
# Load encoder and segmentor initializers from their YMLs
# =========================================================
init_encoder_path = e_cfg.get("init_encoder_path")
init_segmentor_path = s_cfg.get("init_segmentor_path")

if not init_encoder_path:
    raise ValueError("init_encoder_path not found in encoder config")
if not init_segmentor_path:
    raise ValueError("init_segmentor_path not found in segmentor config")

init_encoder_path = os.path.abspath(init_encoder_path)
init_segmentor_path = os.path.abspath(init_segmentor_path)

print(f"Loading encoder initializer from: {init_encoder_path}")
print(f"Loading segmentor initializer from: {init_segmentor_path}")

encoder_init_module = load_module_from_path(init_encoder_path)
segmentor_init_module = load_module_from_path(init_segmentor_path)

initialize_encoder = getattr(encoder_init_module, "initialize_encoder")
initialize_segmentor = getattr(segmentor_init_module, "initialize_segmentor")


# =========================================================
# Create save directory
# =========================================================
current_datetime = datetime.now()

save_dir_parts = [
    "/code/jjiang23/BalanceTestThesis/results",
    os.path.splitext(os.path.basename(args.encoder))[0],
    os.path.splitext(os.path.basename(args.data))[0],
    os.path.splitext(os.path.basename(args.trainer))[0],
    os.path.splitext(os.path.basename(args.segmentor))[0],
    current_datetime.strftime("%Y%m%d_%H%M%S"),
]

save_dir = os.path.join(*save_dir_parts)

os.makedirs(save_dir, exist_ok=True)

print(f"Saving outputs to: {save_dir}")


# =========================================================
# Save merged config snapshot
# =========================================================
config_save_path = os.path.join(save_dir, "config.yml")

with open(config_save_path, "w") as f:
    yaml.dump(config, f, sort_keys=False)

print(f"Saved merged config to: {config_save_path}")


# =========================================================
# Load splits
# =========================================================
with open(splits_path, "r") as f:
    splits = json.load(f)


# =========================================================
# Train each split
# =========================================================
for split_name, split_files in splits.items():
    print(f"\n========== Training Split: {split_name} ==========")

    train_files = split_files["train"]
    val_files = split_files["val"]

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------
    train_ds = PoseDataset(
        train_files,
        window_size=d_cfg["window_size"],
        featureH5Key=d_cfg["h5_key"],
        stride=d_cfg["stride"],
        augment=True,
    )

    val_ds = PoseDataset(
        val_files,
        window_size=d_cfg["window_size"],
        featureH5Key=d_cfg["h5_key"],
        stride=d_cfg["stride"],
        augment=False,
    )

    # -----------------------------------------------------
    # Compute class weights
    # -----------------------------------------------------
    class_weights = train_ds.compute_class_weights(device=t_cfg["device"])

    print(f"Class weights: {class_weights}")

    # -----------------------------------------------------
    # Initialize encoder, segmentor, and trainer
    # Encoder is created first; segmentor receives the encoder
    # object so it can read encoder.out_dim for its input size.
    # -----------------------------------------------------
    encoder = initialize_encoder(d_cfg, e_cfg)
    segmentor = initialize_segmentor(
        s_cfg,
        encoder,
        class_weights=class_weights,
        lambda_smooth=t_cfg.get("lambda_smooth", 0.15),
        time_alignment=t_cfg.get("time_alignment", "downsample_labels"),
    )

    trainer = Trainer(
        encoder=encoder,
        segmentor=segmentor,
        early_stop_patience=t_cfg["early_stop_patience"],
        early_stop_min_delta=t_cfg["early_stop_min_delta"],
        early_stop_monitor=t_cfg["early_stop_monitor"],
    )

    # -----------------------------------------------------
    # Split save directory
    # -----------------------------------------------------
    split_save_dir = os.path.join(save_dir, split_name)

    os.makedirs(split_save_dir, exist_ok=True)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    print("Training now!")

    trainer.train(
        save_dir=split_save_dir,
        batch_gen=train_ds,
        val_batch_gen=val_ds,
        num_epochs=t_cfg["epochs"],
        batch_size=t_cfg["batch_size"],
        learning_rate=float(t_cfg["lr"]),
        weight_decay=float(t_cfg.get("weight_decay", 0.0)),
        device=t_cfg["device"],
    )

    print(f"✓ Training completed for split: {split_name}")

print(f"\n✓ All training completed! Models saved to: {save_dir}")
