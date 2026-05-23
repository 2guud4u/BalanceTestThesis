import numpy as np
import matplotlib.pyplot as plt

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

def graph_segments_with_probs(truthLabels, predLabels, predProbs, vid_path=None):
    """
    Two-panel figure sharing the same x-axis:
      Top:    ground-truth vs predicted class labels (segments)
      Bottom: predicted class probabilities over time
    """
    valid_idx = np.asarray(truthLabels) != -100
    frames = np.arange(len(truthLabels))

    fig, (ax_seg, ax_prob) = plt.subplots(
        2, 1, figsize=(16, 5), sharex=True,
        gridspec_kw={"height_ratios": [1, 2], "hspace": 0.08},
    )

    # ---- Top: segment labels ----
    ax_seg.plot(frames[valid_idx], np.asarray(truthLabels)[valid_idx],
                label="ground truth", lw=1.5, alpha=0.8)
    ax_seg.plot(frames[valid_idx], np.asarray(predLabels)[valid_idx],
                label="predicted", lw=1.5, alpha=0.8)
    ax_seg.set_ylabel("class")
    ax_seg.set_yticks(sorted(set(np.asarray(truthLabels)[valid_idx])))
    ax_seg.legend(loc="upper right", fontsize=8)

    title = "Segments + Probabilities"
    if vid_path:
        title += f" — {vid_path.split('/')[-1]}"
    ax_seg.set_title(title)

    # ---- Bottom: probabilities ----
    for c in range(predProbs.shape[1]):
        ax_prob.plot(frames, predProbs[:, c], label=f"Class {c}", lw=1)
    ax_prob.set_ylabel("probability")
    ax_prob.set_xlabel("frame")
    ax_prob.set_ylim(-0.05, 1.05)
    ax_prob.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.show()


