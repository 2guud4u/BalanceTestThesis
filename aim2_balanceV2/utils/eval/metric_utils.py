import numpy as np
import torch
from data.PoseDataset import load_video_h5
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
def levenshtein(p, y, norm=False):
    """
    Compute Levenshtein (edit) distance between two sequences.
    
    Args:
        p: predicted sequence
        y: ground-truth sequence
        norm: if False, return raw edit distance; if True, return normalized score (0-100)
              where higher is better (similarity percentage)
    
    Returns:
        raw distance or normalized score (0-100) depending on norm flag
    """
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], dtype=float)
    
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1,      # deletion
                              D[i, j - 1] + 1,      # insertion
                              D[i - 1, j - 1] + 1)  # substitution

    if norm:
        # Normalized: 0-100 scale (higher is better, similarity %)
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        # Raw edit distance (lower is better)
        score = D[-1, -1]

    return score

# Aliases for convenience
def edit_distance(s1, s2):
    """Raw Levenshtein distance (lower is better)."""
    return int(levenshtein(s1, s2, norm=False))

def normalized_edit_distance(s1, s2):
    """Normalized edit distance as similarity % (0-100, higher is better)."""
    return levenshtein(s1, s2, norm=True)

def extract_segments(labels, class_id=1, ignore_index=-100):
    """
    labels: 1D array-like of ints (T,)
    returns list of (start_frame, end_frame) inclusive [s, e], where label==class_id
    """
    labels = np.asarray(labels)
    T = len(labels)
    segs = []
    in_seg = False
    s = 0
    for i in range(T):
        v = labels[i]
        if v == ignore_index:
            # treat as background / break segment
            if in_seg:
                segs.append((s, i-1))
                in_seg = False
        elif v == class_id:
            if not in_seg:
                s = i
                in_seg = True
        else:
            if in_seg:
                segs.append((s, i-1))
                in_seg = False
    if in_seg:
        segs.append((s, T-1))
    return segs

def segment_iou(g, p):
    # g, p are (s,e) inclusive
    s1,e1 = g
    s2,e2 = p
    inter_s = max(s1, s2)
    inter_e = min(e1, e2)
    if inter_e < inter_s:
        return 0.0
    inter = inter_e - inter_s + 1
    union = (e1 - s1 + 1) + (e2 - s2 + 1) - inter
    return inter / union

def match_segments(gt_segs, pred_segs, iou_thresh=0.25):
    """
    Greedy one-to-one matching by highest IoU.
    Returns:
      matches: list of tuples (gt_idx, pred_idx, iou)
      unmatched_gt_idxs, unmatched_pred_idxs
    """
    if len(gt_segs) == 0 or len(pred_segs) == 0:
        return [], list(range(len(gt_segs))), list(range(len(pred_segs)))

    # compute IoU matrix
    G = len(gt_segs); P = len(pred_segs)
    iou_mat = np.zeros((G, P), dtype=float)
    for i in range(G):
        for j in range(P):
            iou_mat[i, j] = segment_iou(gt_segs[i], pred_segs[j])

    matches = []
    used_g = set()
    used_p = set()

    # greedy by max IoU
    while True:
        idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        g_idx, p_idx = idx
        best = iou_mat[g_idx, p_idx]
        if best < iou_thresh:
            break
        matches.append((g_idx, p_idx, float(best)))
        used_g.add(g_idx); used_p.add(p_idx)
        # zero out row and col
        iou_mat[g_idx, :] = -1
        iou_mat[:, p_idx] = -1

    unmatched_gt = [i for i in range(G) if i not in used_g]
    unmatched_pred = [j for j in range(P) if j not in used_p]
    return matches, unmatched_gt, unmatched_pred

def compute_segmentation_metrics(gt_labels, pred_labels, class_id=1, iou_thresholds=(0.1, 0.25, 0.5),
                                 fps=30, ignore_index=-100):
    """
    gt_labels, pred_labels: lists/arrays per video (both 1D arrays of same length T)
    Returns aggregated dictionary of metrics (per IoU threshold and duration stats).
    """
    # if single video, wrap into list
    single = False
    if isinstance(gt_labels, np.ndarray) and gt_labels.ndim == 1:
        gt_labels = [gt_labels]
        pred_labels = [pred_labels]
        single = True

    assert len(gt_labels) == len(pred_labels)
    all_metrics = {f"iou_{t}": {"tp":0, "fp":0, "fn":0} for t in iou_thresholds}
    start_errors = []    # in frames
    end_errors = []
    duration_errors = [] # in frames
    per_video_stats = []

    for vid_idx, (g, p) in enumerate(zip(gt_labels, pred_labels)):
        g = np.asarray(g)
        p = np.asarray(p)
        assert g.shape == p.shape, f"GT and pred must be same length for video {vid_idx}"
        gt_segs = extract_segments(g, class_id=class_id, ignore_index=ignore_index)
        pred_segs = extract_segments(p, class_id=class_id, ignore_index=ignore_index)
        vid_stat = {"n_gt": len(gt_segs), "n_pred": len(pred_segs)}
        per_video_stats.append(vid_stat)

        for thr in iou_thresholds:
            matches, unmatched_gt, unmatched_pred = match_segments(gt_segs, pred_segs, iou_thresh=thr)
            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            all_metrics[f"iou_{thr}"]["tp"] += tp
            all_metrics[f"iou_{thr}"]["fp"] += fp
            all_metrics[f"iou_{thr}"]["fn"] += fn

        # choose a single matching threshold for boundary/duration errors (use lenient e.g., 0.1 or 0.25)
        match_thr_for_errors = 0.25
        matches, unmatched_gt, unmatched_pred = match_segments(gt_segs, pred_segs, iou_thresh=match_thr_for_errors)
        for (gi, pi, iouval) in matches:
            gs, ge = gt_segs[gi]
            ps, pe = pred_segs[pi]
            start_errors.append(ps - gs)   # predicted_start - gt_start (frames)
            end_errors.append(pe - ge)     # predicted_end - gt_end
            duration_errors.append((pe - ps + 1) - (ge - gs + 1))  # pred_len - gt_len (frames)

    # aggregate metrics
    results = {}
    
    # IoU-based metrics
    for thr in iou_thresholds:
        tp = all_metrics[f"iou_{thr}"]["tp"]
        fp = all_metrics[f"iou_{thr}"]["fp"]
        fn = all_metrics[f"iou_{thr}"]["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[f"precision_iou_{thr}"] = precision
        results[f"recall_iou_{thr}"] = recall
        results[f"f1_iou_{thr}"] = f1
        results[f"tp_iou_{thr}"] = tp
        results[f"fp_iou_{thr}"] = fp
        results[f"fn_iou_{thr}"] = fn
    
    # Frame-level edit distance (on full label sequences)
    if not single:
        # Multi-video case: compute edit distance for each video
        edit_distances = []
        for g, p in zip(gt_labels, pred_labels):
            g = np.asarray(g)
            p = np.asarray(p)
            valid = g != ignore_index
            ed = edit_distance(g[valid].tolist(), p[valid].tolist())
            edit_distances.append(ed)
        results["edit_distance_frames_per_video"] = edit_distances
        results["edit_distance_frames_mean"] = float(np.mean(edit_distances)) if edit_distances else 0.0
        results["edit_distance_frames_max"] = float(np.max(edit_distances)) if edit_distances else 0.0
    else:
        # Single video case
        g = np.asarray(gt_labels[0])
        p = np.asarray(pred_labels[0])
        valid = g != ignore_index
        ed = edit_distance(g[valid].tolist(), p[valid].tolist())
        results["edit_distance_frames"] = ed
        results["edit_distance_frames_normalized"] = normalized_edit_distance(g[valid].tolist(), p[valid].tolist())

    # duration & boundary statistics (convert to seconds)
    def stats_from_list(arr_frames):
        if len(arr_frames) == 0:
            return {"count":0, "mean":None, "median":None, "std":None}
        a = np.array(arr_frames, dtype=float)
        return {"count":len(a), "mean":float(a.mean()), "median":float(np.median(a)), "std":float(a.std())}

    res_start = stats_from_list(start_errors)
    res_end = stats_from_list(end_errors)
    res_dur = stats_from_list(duration_errors)

    # convert frames -> seconds if fps provided
    def frames_to_seconds_dict(d):
        if d["mean"] is None:
            return {k: None if v is None else None for k,v in d.items()}
        return {k: (d[k] / fps if d[k] is not None else None) for k in d}

    results["start_error_frames"] = res_start
    results["end_error_frames"] = res_end
    results["duration_error_frames"] = res_dur
    results["start_error_seconds"] = frames_to_seconds_dict(res_start)
    results["end_error_seconds"] = frames_to_seconds_dict(res_end)
    results["duration_error_seconds"] = frames_to_seconds_dict(res_dur)

    # percent within tolerance (seconds)
    tolerances = [0.5, 1.0]  # seconds
    if len(duration_errors) > 0:
        dur_secs = np.abs(np.array(duration_errors, dtype=float)) / fps
        for tol in tolerances:
            results[f"pct_duration_within_{tol}s"] = float((dur_secs <= tol).mean())
    else:
        for tol in tolerances:
            results[f"pct_duration_within_{tol}s"] = None

    results["per_video_stats"] = per_video_stats
    return results

def compute_averaged_video_metrics(pred_labels, gt_labels, class_id=1,
                                   iou_thresholds=(0.1, 0.25, 0.5),
                                   fps=30, ignore_index=-100):
    """
    Compute per-video segmentation metrics and return the averaged values across videos.

    Unlike compute_segmentation_metrics (which pools TP/FP/FN across all videos before
    computing precision/recall/F1), this function computes the metrics independently for
    each video and then averages them — i.e. macro-average over videos.

    Args:
        pred_labels: list of 1D arrays (one per video) of predicted frame labels
        gt_labels:   list of 1D arrays (one per video) of ground-truth frame labels
        class_id:    class to evaluate
        iou_thresholds: IoU thresholds for F1 computation
        fps:         frames per second (used for second-level boundary stats)
        ignore_index: label value to ignore

    Returns:
        dict with:
          - averaged metrics (precision/recall/F1 per IoU threshold, edit distance,
            normalized edit distance, frame accuracy, boundary/duration errors)
          - per_video: list of per-video metric dicts
          - num_videos
    """
    assert len(pred_labels) == len(gt_labels), "preds and truths must have same number of videos"
    n_videos = len(gt_labels)

    per_video = []

    for vid_idx, (p, g) in enumerate(zip(pred_labels, gt_labels)):
        g = np.asarray(g)
        p = np.asarray(p)
        assert g.shape == p.shape, f"GT and pred must be same length for video {vid_idx}"

        gt_segs = extract_segments(g, class_id=class_id, ignore_index=ignore_index)
        pred_segs = extract_segments(p, class_id=class_id, ignore_index=ignore_index)

        vid_metrics = {"n_gt": len(gt_segs), "n_pred": len(pred_segs)}

        # F1@IoU per video
        for thr in iou_thresholds:
            matches, unmatched_gt, unmatched_pred = match_segments(gt_segs, pred_segs, iou_thresh=thr)
            tp = len(matches)
            fp = len(unmatched_pred)
            fn = len(unmatched_gt)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            vid_metrics[f"precision_iou_{thr}"] = precision
            vid_metrics[f"recall_iou_{thr}"] = recall
            vid_metrics[f"f1_iou_{thr}"] = f1
            vid_metrics[f"tp_iou_{thr}"] = tp
            vid_metrics[f"fp_iou_{thr}"] = fp
            vid_metrics[f"fn_iou_{thr}"] = fn

        # Edit distance per video (over valid frames only)
        valid = g != ignore_index
        g_valid = g[valid].tolist()
        p_valid = p[valid].tolist()
        vid_metrics["edit_distance_frames"] = edit_distance(g_valid, p_valid)
        vid_metrics["edit_distance_normalized"] = normalized_edit_distance(g_valid, p_valid)

        # Frame-level accuracy on valid frames
        if valid.sum() > 0:
            vid_metrics["frame_accuracy"] = float((g[valid] == p[valid]).mean())
        else:
            vid_metrics["frame_accuracy"] = 0.0

        # Boundary / duration errors on matched segments at IoU 0.25
        match_thr = 0.25
        matches, _, _ = match_segments(gt_segs, pred_segs, iou_thresh=match_thr)
        starts, ends, durs = [], [], []
        for (gi, pi, _) in matches:
            gs, ge = gt_segs[gi]
            ps, pe = pred_segs[pi]
            starts.append(ps - gs)
            ends.append(pe - ge)
            durs.append((pe - ps + 1) - (ge - gs + 1))

        def _mean_abs_seconds(arr):
            if len(arr) == 0:
                return None
            return float(np.mean(np.abs(np.array(arr, dtype=float))) / fps)

        vid_metrics["mean_abs_start_error_seconds"] = _mean_abs_seconds(starts)
        vid_metrics["mean_abs_end_error_seconds"] = _mean_abs_seconds(ends)
        vid_metrics["mean_abs_duration_error_seconds"] = _mean_abs_seconds(durs)

        per_video.append(vid_metrics)

    # Average across videos. None values are skipped per metric.
    def _avg(key):
        vals = [v[key] for v in per_video if v.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    averaged = {"num_videos": n_videos}

    for thr in iou_thresholds:
        averaged[f"precision_iou_{thr}"] = _avg(f"precision_iou_{thr}")
        averaged[f"recall_iou_{thr}"] = _avg(f"recall_iou_{thr}")
        averaged[f"f1_iou_{thr}"] = _avg(f"f1_iou_{thr}")

    averaged["edit_distance_frames"] = _avg("edit_distance_frames")
    averaged["edit_distance_normalized"] = _avg("edit_distance_normalized")
    averaged["frame_accuracy"] = _avg("frame_accuracy")
    averaged["mean_abs_start_error_seconds"] = _avg("mean_abs_start_error_seconds")
    averaged["mean_abs_end_error_seconds"] = _avg("mean_abs_end_error_seconds")
    averaged["mean_abs_duration_error_seconds"] = _avg("mean_abs_duration_error_seconds")

    averaged["per_video"] = per_video
    return averaged


def print_metrics_summary(metrics, class_id=0):
    """
    Print a clean summary of key metrics:
    - F1@IoU 0.1, 0.25, 0.5
    - Edit distance (frame-level)
    - Start/end boundary differences
    - Frame-by-frame accuracy
    
    Args:
        metrics: dict returned by compute_segmentation_metrics()
        class_id: class being evaluated (for context)
    """
    print(f"\n{'='*70}")
    print(f"Segmentation Metrics Summary (Class {class_id})")
    print(f"{'='*70}")
    
    # F1@IoU metrics
    print(f"\n{'F1 Score by IoU Threshold:':<40}")
    for thr in [0.1, 0.25, 0.5]:
        f1 = metrics.get(f"f1_iou_{thr}", 0.0)
        p = metrics.get(f"precision_iou_{thr}", 0.0)
        r = metrics.get(f"recall_iou_{thr}", 0.0)
        print(f"  IoU@{thr}: F1={f1:.4f}  (P={p:.4f}, R={r:.4f})")
    
    # Edit distance
    print(f"\n{'Edit Distance (Frame-level):':<40}")
    if "edit_distance_frames" in metrics:
        ed = metrics["edit_distance_frames"]
        ed_norm = metrics.get("edit_distance_frames_normalized", 0.0)
        print(f"  Raw distance: {int(ed)} edits")
        print(f"  Similarity: {ed_norm:.2f}%")
    else:
        ed_mean = metrics.get("edit_distance_frames_mean", 0.0)
        print(f"  Mean distance: {ed_mean:.1f} edits")
    
    # Boundary errors
    print(f"\n{'Boundary Errors (matched @ IoU 0.25):':<40}")
    for key in ["start_error_seconds", "end_error_seconds", "duration_error_seconds"]:
        s = metrics.get(key, {})
        if s and s.get("mean") is not None:
            label = key.replace("_error_seconds", "").replace("_", " ").title()
            print(f"  {label}: mean={s['mean']:.2f}s, median={s['median']:.2f}s, std={s['std']:.2f}s")
        else:
            print(f"  {key}: no matched segments")
    
    # Frame-by-frame accuracy (estimated from per_video_stats if available)
    print(f"\n{'Segment Statistics:':<40}")
    stats = metrics.get("per_video_stats", [])
    if stats:
        total_gt = sum(s["n_gt"] for s in stats)
        total_pred = sum(s["n_pred"] for s in stats)
        print(f"  Total GT segments: {total_gt}")
        print(f"  Total predicted segments: {total_pred}")
        print(f"  Segment count diff: {total_pred - total_gt:+d}")
    
    print(f"\n{'='*70}\n")

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

@torch.no_grad()
def predict_video(h5_path, encoder, model, d_cfg, device="gpu", stride_override=None):
    """
    Run sliding-window inference on a full video.

    Returns:
        gt_labels:   (T,) int array — ground-truth per frame
        pred_labels: (T,) int array — predicted class per frame
        pred_probs:  (T, C) float array — averaged softmax probabilities
    """
    window_size = window_size = d_cfg["window_size"] 
    stride      = stride_override if stride_override is not None else d_cfg["stride"]
    h5_key      = d_cfg["h5_key"]

    kps, gt_labels = load_video_h5(h5_path, h5_key, allPhases=False)  # (T, J, D), (T,)
    T_total = kps.shape[0]
    J = kps.shape[1]
    D = kps.shape[2]
    num_classes = 2

    # Accumulators for stitching overlapping windows
    prob_sum   = np.zeros((T_total, num_classes), dtype=np.float64)
    vote_count = np.zeros(T_total, dtype=np.float64)

    # Slide windows
    starts = list(range(0, max(1, T_total - 1), stride))
    # Make sure the last window covers the tail
    if starts[-1] + window_size < T_total:
        starts.append(max(0, T_total - window_size))

    for s in starts:
        e = min(s + window_size, T_total)
        chunk = kps[s:e]               # (L, J, D)
        L = chunk.shape[0]

        # Pad to window_size if needed
        if L < window_size:
            pad = np.zeros((window_size - L, J, D), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad], axis=0)

        # Flatten and add batch dim
        x = torch.from_numpy(chunk.reshape(1, window_size, J * D)).float().to(device)

        enc_feat = encoder(x)             # (1, D_enc, T_enc)
        preds    = model(enc_feat)        # list[(1, C, T_pred)]
        logits   = preds[-1]             # last refinement stage (1, C, T_pred)

        # If T_pred != window_size, upsample to window_size
        T_pred = logits.shape[2]
        if T_pred != window_size:
            logits = F.interpolate(logits, size=window_size, mode='nearest')

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()  # (C, window_size)
        probs = probs.T  # (window_size, C)

        # Only use the first L frames (ignore padding)u 
        prob_sum[s:s+L]   += probs[:L]
        vote_count[s:s+L] += 1.0

    # Average overlapping predictions
    valid = vote_count > 0
    pred_probs = np.zeros_like(prob_sum)
    pred_probs[valid] = prob_sum[valid] / vote_count[valid, None]
    pred_labels = pred_probs.argmax(axis=1)

    return gt_labels, pred_labels, pred_probs
