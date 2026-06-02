import os
import sys
import json
import yaml
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from utils.eval.metric_utils import compute_averaged_video_metrics, predict_video, average_across_folds


# -------------------------------------------------------------------
# Institution extraction
# -------------------------------------------------------------------
def get_institution(video_path):
    """
    Extract institution name from a video path.

    Tries two strategies:
      1. Look for a 'heavy/' segment and return the folder right after it.
         e.g. .../heavy/uwisc/UWisc_5036_Balance.h5 → 'uwisc'
      2. Fall back to the parent directory name.
         e.g. /data/uwisc/video.h5 → 'uwisc'
    """
    parts = video_path.replace("\\", "/").split("/")
    try:
        idx = parts.index("heavy")
        return parts[idx + 1]
    except (ValueError, IndexError):
        # Fallback: parent directory
        return os.path.basename(os.path.dirname(video_path))


# -------------------------------------------------------------------
# Per-fold evaluation
# -------------------------------------------------------------------
def evaluate_folds(trainer, splits, d_cfg, t_cfg, save_dir, device, stride_override=30):
    per_fold_results = {}
    fold_metric_dicts = []

    # Collect predictions across all folds for institution-level analysis
    all_video_results = []  # list of (video_path, gt_labels, pred_labels)

    for split_name, split_files in splits.items():
        val_files = split_files["val"]

        fold_dir = os.path.join(save_dir, split_name)
        encoder_ckpt_path = os.path.join(fold_dir, "best_encoder.pt")
        model_ckpt_path = os.path.join(fold_dir, "best_segmentor.pt")

        if not (os.path.exists(encoder_ckpt_path) and os.path.exists(model_ckpt_path)):
            print(f"⚠ Skipping {split_name}: checkpoints not found in {fold_dir}")
            continue

        print(f"\n{'='*70}\nFold: {split_name}  ({len(val_files)} val videos)\n{'='*70}")

        trainer.encoder.load_state_dict(torch.load(encoder_ckpt_path, map_location=device))
        # strict=False: checkpoint contains class_weights / ce.weight buffers saved during
        # training; those are absent when segmentor is constructed with class_weights=None
        # for eval. They are only used by compute_loss, which is never called here.
        trainer.segmentor.load_state_dict(
            torch.load(model_ckpt_path, map_location=device), strict=False
        )
        trainer.encoder.to(device).eval()
        trainer.segmentor.to(device).eval()
        print(f"✓ Loaded {model_ckpt_path}")
        print(f"✓ Loaded {encoder_ckpt_path}")

        pred_labels_per_video = []
        gt_labels_per_video = []

        for video in tqdm(val_files, desc=f"[{split_name}] predicting"):
            gt_labels, pred_labels, _ = predict_video(
                video, trainer.encoder, trainer.segmentor, d_cfg, device, stride_override=stride_override
            )
            T = min(len(pred_labels), len(gt_labels))
            gt_arr = np.asarray(gt_labels[:T])
            pred_arr = np.asarray(pred_labels[:T])
            pred_labels_per_video.append(pred_arr)
            gt_labels_per_video.append(gt_arr)
            all_video_results.append((video, gt_arr, pred_arr))

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
    # Cross-fold summary (overall)
    # -------------------------------------------------------------------

    cross_fold = average_across_folds(fold_metric_dicts)

    print(f"\n{'='*70}\nCross-fold summary (n={len(fold_metric_dicts)} folds)\n{'='*70}")
    for k, v in cross_fold.items():
        if v["mean"] is None:
            print(f"  {k}: no data")
        else:
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}  (n={v['n_folds']})")

    # -------------------------------------------------------------------
    # Overall metrics (pooled across all folds)
    # -------------------------------------------------------------------
    if all_video_results:
        all_gt = [r[1] for r in all_video_results]
        all_pred = [r[2] for r in all_video_results]

        overall_metrics = compute_averaged_video_metrics(
            pred_labels=all_pred,
            gt_labels=all_gt,
            class_id=t_cfg.get("eval_class_id", 1),
            iou_thresholds=tuple(t_cfg.get("iou_thresholds", (0.1, 0.25, 0.5))),
            fps=t_cfg.get("fps", 30),
            ignore_index=t_cfg.get("ignore_index", -100),
        )

        print(f"\n{'='*70}\nOverall metrics (n={overall_metrics['num_videos']} videos)\n{'='*70}")
        for k, v in overall_metrics.items():
            if k in ("per_video", "num_videos"):
                continue
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        overall_metrics = {}

    # -------------------------------------------------------------------
    # Per-institution metrics
    # -------------------------------------------------------------------
    institution_metrics = {}

    if all_video_results:
        # Group videos by institution
        by_institution = defaultdict(lambda: {"gt": [], "pred": [], "paths": []})
        for video_path, gt_arr, pred_arr in all_video_results:
            inst = get_institution(video_path)
            by_institution[inst]["gt"].append(gt_arr)
            by_institution[inst]["pred"].append(pred_arr)
            by_institution[inst]["paths"].append(video_path)

        for inst in sorted(by_institution.keys()):
            data = by_institution[inst]
            n_vids = len(data["gt"])

            inst_metrics = compute_averaged_video_metrics(
                pred_labels=data["pred"],
                gt_labels=data["gt"],
                class_id=t_cfg.get("eval_class_id", 1),
                iou_thresholds=tuple(t_cfg.get("iou_thresholds", (0.1, 0.25, 0.5))),
                fps=t_cfg.get("fps", 30),
                ignore_index=t_cfg.get("ignore_index", -100),
            )

            print(f"\n{'='*70}\nInstitution: {inst}  (n={n_vids} videos)\n{'='*70}")
            for k, v in inst_metrics.items():
                if k in ("per_video", "num_videos"):
                    continue
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

            institution_metrics[inst] = {
                "num_videos": n_vids,
                "metrics": inst_metrics,
            }

    # -------------------------------------------------------------------
    # Save everything
    # -------------------------------------------------------------------
    out_path = os.path.join(save_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "save_dir": save_dir,
                "evaluated_at": datetime.now().isoformat(timespec="seconds"),
                "folds_evaluated": list(per_fold_results.keys()),
                "per_fold": per_fold_results,
                "cross_fold_summary": cross_fold,
                "overall": overall_metrics,
                "per_institution": institution_metrics,
            },
            f,
            indent=2,
            default=float,
        )
    print(f"\n✓ Saved all metrics to {out_path}")
