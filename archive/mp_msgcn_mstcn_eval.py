import os
import sys
import json
import yaml
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

sys.path.append('..')
sys.path.insert(0, '../models/encoders/MS-GCN')

from models.encoders.GCN.MS_GCN_ENCODER import MSGCNEncoder
from scripts.Trainers import Simple_MSTCN2_Trainer
from utils.eval.metric_utils import compute_averaged_video_metrics, predict_video

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
save_dir = "/code/jjiang23/pathml/aim2_balanceV2/results/CAMMP_MSGCN_MSTCN/20260501_200300"
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
encoder = MSGCNEncoder(
    e_cfg["graph_args"],
    num_joints=e_cfg["num_joints"],
    in_channels=e_cfg["in_channels"],
    filters=e_cfg["filters"],
    dil=e_cfg["dil"],
)
trainer = Simple_MSTCN2_Trainer(
    encoder=encoder,
    early_stop_patience=t_cfg["early_stop_patience"],
    early_stop_min_delta=t_cfg["early_stop_min_delta"],
    early_stop_monitor=t_cfg["early_stop_monitor"],
)
print(f"✓ Trainer initialized (encoder out_dim={encoder.out_dim})")

# -------------------------------------------------------------------
# Per-fold evaluation
# -------------------------------------------------------------------
per_fold_results = {}
fold_metric_dicts = []

for split_name, split_files in splits.items():
    val_files = split_files["val"]

    fold_dir = os.path.join(save_dir, split_name)
    encoder_ckpt_path = os.path.join(fold_dir, "best_encoder.model")
    model_ckpt_path = os.path.join(fold_dir, "best.model")

    if not (os.path.exists(encoder_ckpt_path) and os.path.exists(model_ckpt_path)):
        print(f"⚠ Skipping {split_name}: checkpoints not found in {fold_dir}")
        continue

    print(f"\n{'='*70}\nFold: {split_name}  ({len(val_files)} val videos)\n{'='*70}")

    trainer.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
    trainer.model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
    trainer.encoder.to(device).eval()
    trainer.model.to(device).eval()
    print(f"✓ Loaded {model_ckpt_path}")
    print(f"✓ Loaded {encoder_ckpt_path}")

    pred_labels_per_video = []
    gt_labels_per_video = []

    for video in tqdm(val_files, desc=f"[{split_name}] predicting"):
        gt_labels, pred_labels, _ = predict_video(
            video, trainer.encoder, trainer.model, d_cfg, device, stride_override=1
        )
        T = min(len(pred_labels), len(gt_labels))
        pred_labels_per_video.append(np.asarray(pred_labels[:T]))
        gt_labels_per_video.append(np.asarray(gt_labels[:T]))

    # NOTE: signature is (pred_labels, gt_labels, ...) — order matters.
    metrics = compute_averaged_video_metrics(
        pred_labels=pred_labels_per_video,
        gt_labels=gt_labels_per_video,
        class_id=t_cfg.get("eval_class_id", 1),
        iou_thresholds=tuple(t_cfg.get("iou_thresholds", (0.1, 0.25, 0.5))),
        fps=t_cfg.get("fps", 30),
        ignore_index=t_cfg.get("ignore_index", -100),
    )

    # Concise per-fold print (skip the giant per_video list)
    print(f"\n[{split_name}] averaged per-video metrics (n={metrics['num_videos']}):")
    for k, v in metrics.items():
        if k in ("per_video", "num_videos"):
            continue
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Save per-fold metrics next to the fold's checkpoints
    fold_out = os.path.join(fold_dir, "eval_metrics.json")
    with open(fold_out, "w") as f:
        json.dump(
            {
                "fold": split_name,
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "metrics": metrics,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"✓ Saved per-fold metrics to {fold_out}")

    per_fold_results[split_name] = metrics
    fold_metric_dicts.append(metrics)

# -------------------------------------------------------------------
# Cross-fold summary
# -------------------------------------------------------------------
def average_across_folds(fold_metrics):
    if not fold_metrics:
        return {}
    skip = {"per_video", "num_videos"}
    keys = [k for k, v in fold_metrics[0].items()
            if k not in skip and isinstance(v, (int, float)) and v is not None]
    summary = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if m.get(k) is not None]
        if not vals:
            summary[k] = {"mean": None, "std": None, "n_folds": 0}
        else:
            arr = np.array(vals, dtype=float)
            summary[k] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "n_folds": int(len(arr)),
            }
    return summary


cross_fold = average_across_folds(fold_metric_dicts)

print(f"\n{'='*70}\nCross-fold summary (n={len(fold_metric_dicts)} folds)\n{'='*70}")
for k, v in cross_fold.items():
    if v["mean"] is None:
        print(f"  {k}: no data")
    else:
        print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n_folds']})")

out_path = os.path.join(save_dir, "eval_metrics.json")
with open(out_path, "w") as f:
    json.dump(
        {
            "save_dir": save_dir,
            "evaluated_at": datetime.now().isoformat(timespec="seconds"),
            "folds_evaluated": list(per_fold_results.keys()),
            "per_fold": per_fold_results,
            "cross_fold_summary": cross_fold,
        },
        f,
        indent=2,
        default=float,
    )
print(f"\n✓ Saved cross-fold metrics to {out_path}")
