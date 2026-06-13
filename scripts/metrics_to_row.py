import os
import argparse
import json

def fmt(x):
    return f"{x:.2f}".rstrip("0").rstrip(".")

def print_metric(metrics, with_std=True):
    f1_iou_10 = metrics["f1_iou_0.1"]
    f1_iou_25 = metrics["f1_iou_0.25"]
    f1_iou_50 = metrics["f1_iou_0.5"]
    norm_edit = metrics["edit_distance_normalized"]
    acc = metrics["frame_accuracy"]
    bound_start = metrics["mean_abs_start_error_seconds"]
    bound_end = metrics["mean_abs_end_error_seconds"]
    if with_std:
        print(
            f"& {fmt(f1_iou_10['mean'])} ± {fmt(f1_iou_10['std'])} "
            f"& {fmt(f1_iou_25['mean'])} ± {fmt(f1_iou_25['std'])} "
            f"& {fmt(f1_iou_50['mean'])} ± {fmt(f1_iou_50['std'])} "
            f"& {fmt(norm_edit['mean'])} ± {fmt(norm_edit['std'])} "
            f"& {fmt(acc['mean'])} ± {fmt(acc['std'])} "
            f"& {fmt(bound_start['mean'])} ± {fmt(bound_start['std'])} "
            f"& {fmt(bound_end['mean'])} ± {fmt(bound_end['std'])}"
        )
    else:
        print(
            f"& {fmt(f1_iou_10)} "
            f"& {fmt(f1_iou_25)} "
            f"& {fmt(f1_iou_50)} "
            f"& {fmt(norm_edit)} "
            f"& {fmt(acc)} "
            f"& {fmt(bound_start)} "
            f"& {fmt(bound_end)}"
        )
parser = argparse.ArgumentParser()

parser.add_argument(
    "--results_dir",
    type=str,
    required=True,
    help="Path to a completed train.py output directory (must contain config.yml)",
)

parser.add_argument(
    "--type",
    type=str,
    required=False,
    help="specific type of metrics?",
)

args = parser.parse_args()

with open(os.path.join(args.results_dir, "eval_metrics.json"), "r") as f:
    metrics = json.load(f)
    if args.type == "institution":
        metric_obj = metrics["per_institution"]
        for institution in ["cp", "uwisc", "va"]:
            target_metrics = metric_obj[institution]["metrics"]
            print(f"{institution.upper()} ", end="")
            print_metric(target_metrics, with_std=False)
    else:
        target_metrics = metrics["cross_fold_summary"]
        print_metric(target_metrics, with_std=True)
    
