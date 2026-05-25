import os
import argparse
import json
import yaml
import importlib.util

import torch
import sys

sys.path.append("..")

from Trainer import Trainer
from scripts.Evaler import evaluate_folds


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
    description="Generic skeleton action segmentation evaluator"
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

parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="Path to directory containing trained fold checkpoints (output of train.py)",
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
results_dir = os.path.abspath(args.results_dir)

print(f"Loading encoder config from:   {encoder_cfg_path}")
print(f"Loading data config from:      {data_cfg_path}")
print(f"Loading trainer config from:   {trainer_cfg_path}")
print(f"Loading segmentor config from: {segmentor_cfg_path}")
print(f"Evaluating checkpoints from:   {results_dir}")


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

print(f"Loading encoder initializer from:   {init_encoder_path}")
print(f"Loading segmentor initializer from: {init_segmentor_path}")

encoder_init_module = load_module_from_path(init_encoder_path)
segmentor_init_module = load_module_from_path(init_segmentor_path)

initialize_encoder = getattr(encoder_init_module, "initialize_encoder")
initialize_segmentor = getattr(segmentor_init_module, "initialize_segmentor")


# =========================================================
# Load splits
# =========================================================
with open(splits_path, "r") as f:
    splits = json.load(f)


# =========================================================
# Initialize encoder and segmentor (architecture only —
# weights are loaded per-fold inside evaluate_folds)
# =========================================================
encoder = initialize_encoder(d_cfg, e_cfg)
segmentor = initialize_segmentor(
    s_cfg,
    encoder,
    class_weights=None,  # not needed for eval
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


# =========================================================
# Evaluate all folds
# =========================================================
evaluate_folds(
    trainer=trainer,
    splits=splits,
    d_cfg=d_cfg,
    t_cfg=t_cfg,
    save_dir=results_dir,
    device=t_cfg["device"],
)

print(f"\n✓ Evaluation complete. Results saved to: {results_dir}")
