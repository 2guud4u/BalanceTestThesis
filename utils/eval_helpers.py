from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def get_label_mapping():
    """
    Get the label mapping for the dataset.
    
    Returns:
        A dictionary mapping label names to indices.
    """
    return {
        "Phase1": 0,
        "Phase2": 1,
        "Phase3": 2,
        "Phase4": 3,
        "nonphase": 4
    }

def get_classifer_metrics(true_labels,pred_labels,saveDir):
    """    Calculate and save classification metrics including confusion matrix and classification report.
    Args:
        true_labels (list): List of true labels.
        pred_labels (list): List of predicted labels.
        saveDir (str): Directory to save the confusion matrix and classification report.
    """
    # Get label mapping
    label_mapping = get_label_mapping()
    # Create and save confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.keys()))
    disp.plot(cmap='Blues')
    disp.ax_.set_title('Confusion Matrix')

    # Save figure
    plt.savefig(saveDir+"confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

    # Save classification report
    with open(saveDir+"classification_report.txt", 'w') as f:
        f.write(classification_report(true_labels, pred_labels, target_names=list(label_mapping.keys())))

def time_stamp_phases(labels, window=30):
    """
    Detect start and end times of consistent phases in the labels.
    
    Args:
        labels (list): List of integer labels for each frame.
        window (int): Number of consistent frames required to confirm a phase.

    Returns:
        start_phase_times (dict): Phase label → start frame index.
        end_phase_times (dict): Phase label → end frame index.
    """
    start_phase_times = {}
    end_phase_times = {}
    n = len(labels)

    i = 0
    while i <= n - window:
        segment = labels[i:i + window]
        if len(set(segment)) == 1:
            label = segment[0]
            if label != 4:  # skip "non-phase"
                # Record start if not already seen
                if label not in start_phase_times:
                    start_phase_times[label] = i

                # Move forward to find end of this consistent phase
                j = i + window
                while j < n and labels[j] == label:
                    j += 1
                end_phase_times[label] = j - 1  # Last consistent index
                i = j  # Skip ahead
                continue
        i += 1

    return start_phase_times, end_phase_times

import numpy as np
import matplotlib.pyplot as plt
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

def graph_class_segments(truthLabels, predLabels, vid_path=None):
    """
    Plot predicted vs ground truth labels over frames.
    truthLabels, predLabels: 1D arrays of class labels
    vid_path: optional video file path for title
    """
    # Create valid frame mask (exclude padding frames with label -100)
    valid_idx = truthLabels != -100
    
    plt.figure(figsize=(14, 3))
    plt.plot(np.arange(len(predLabels))[valid_idx], predLabels[valid_idx], 
             label="predicted", lw=1)
    
    if truthLabels.shape[0] == predLabels.shape[0]:
        plt.plot(np.arange(len(truthLabels))[valid_idx], truthLabels[valid_idx], 
                 label="ground_truth", lw=1, alpha=0.8)
    
    plt.xlabel("frame")
    plt.ylabel("class")
    
    title = "Predicted classes"
    if vid_path:
        title += f" — file {vid_path.split('/')[-1]}"
    plt.title(title)
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def graph_probabilities(predProbs, vid_path=None):
    """
    predProbs: (T, C) array of predicted probabilities per class
    vid_path: optional video file path for title
    """
    plt.figure(figsize=(14, 4))
    for c in range(predProbs.shape[1]):
        plt.plot(predProbs[:, c], label=f"Class {c}", lw=1)
    
    plt.xlabel("frame")
    plt.ylabel("predicted probability")
    
    title = "Predicted class probabilities"
    if vid_path:
        title += f" — file {vid_path.split('/')[-1]}"
    plt.title(title)
    
    plt.legend()
    plt.tight_layout()
    plt.show()