import os
import argparse
import json

def fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")
parser = argparse.ArgumentParser()

parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="Path to a completed train.py output directory (must contain config.yml)",
)

args = parser.parse_args()

with open(os.path.join(args.results_dir, "eval_metrics.json"), "r") as f:
    metrics = json.load(f)
    cross_fold_summary = metrics["cross_fold_summary"]
    f1_iou_10 = cross_fold_summary["f1_iou_0.1"]
    f1_iou_25 = cross_fold_summary["f1_iou_0.25"]
    f1_iou_50 = cross_fold_summary["f1_iou_0.5"]
    norm_edit = cross_fold_summary["edit_distance_normalized"]
    acc = cross_fold_summary["frame_accuracy"]
    bound_start = cross_fold_summary["mean_abs_start_error_seconds"]
    bound_end = cross_fold_summary["mean_abs_end_error_seconds"]
    print(
        f"& {fmt(f1_iou_10['mean'])} ± {fmt(f1_iou_10['std'])} "
        f"& {fmt(f1_iou_25['mean'])} ± {fmt(f1_iou_25['std'])} "
        f"& {fmt(f1_iou_50['mean'])} ± {fmt(f1_iou_50['std'])} "
        f"& {fmt(norm_edit['mean'])} ± {fmt(norm_edit['std'])} "
        f"& {fmt(acc['mean'])} ± {fmt(acc['std'])} "
        f"& {fmt(bound_start['mean'])} ± {fmt(bound_start['std'])} "
        f"& {fmt(bound_end['mean'])} ± {fmt(bound_end['std'])}"
    )
