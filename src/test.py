import os
import argparse
import json
import yaml
import importlib.util

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
    description="Evaluate a trained skeleton action segmentation model"
)

parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="Path to a completed train.py output directory (must contain config.yml)",
)

args = parser.parse_args()

results_dir = os.path.abspath(args.results_dir)


# =========================================================
# Load merged config saved by train.py
# =========================================================
config_path = os.path.join(results_dir, "config.yml")

if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"config.yml not found in {results_dir}. "
        "Make sure --results_dir points to the root output directory of a train.py run."
    )

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

e_cfg = config["encoder"]
d_cfg = config["data"]
t_cfg = config["trainer"]
s_cfg = config["segmentor"]
splits_path = config["paths"]["splits_path"]

print(f"Loaded config from: {config_path}")
print(f"Evaluating checkpoints in: {results_dir}")


# =========================================================
# Load encoder and segmentor initializers from their paths
# (stored in the merged config)
# =========================================================
if e_cfg is not None:
    init_encoder_path = e_cfg.get("init_encoder_path")
    if not init_encoder_path:
        raise ValueError("init_encoder_path not found in encoder config")
    init_encoder_path = os.path.abspath(init_encoder_path)
    print(f"Loading encoder initializer from:   {init_encoder_path}")
    encoder_init_module = load_module_from_path(init_encoder_path)
    initialize_encoder = getattr(encoder_init_module, "initialize_encoder")
else:
    # No encoder config → use the identity (passthrough) encoder
    _identity_init_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "initializers", "encoder", "Identity.py")
    )
    print(f"Loading identity encoder initializer from: {_identity_init_path}")
    encoder_init_module = load_module_from_path(_identity_init_path)
    initialize_encoder = getattr(encoder_init_module, "initialize_encoder")

init_segmentor_path = s_cfg.get("init_segmentor_path")
if not init_segmentor_path:
    raise ValueError("init_segmentor_path not found in segmentor config")

init_segmentor_path = os.path.abspath(init_segmentor_path)
print(f"Loading segmentor initializer from: {init_segmentor_path}")

segmentor_init_module = load_module_from_path(init_segmentor_path)
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
    lambda_smooth=t_cfg.get("lambda_smooth", 0.01),
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
