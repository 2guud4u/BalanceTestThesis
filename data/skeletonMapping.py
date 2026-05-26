import numpy as np

try:
    import torch
except Exception:  # torch is optional for pure-numpy usage
    torch = None

MPStringsToKP = {
    "nose": 0,
    "leftMouth": 9,
    "rightMouth": 10,
    "leftShoulder": 11,
    "rightShoulder": 12,
    "leftElbow": 13,
    "rightElbow": 14,
    "leftHand": 15,
    "rightHand": 16,
    "rightThumb": 22,
    "rightIndex": 20,
    "rightPinky": 18,
    "leftThumb": 21,
    "leftIndex": 19,
    "leftPinky": 17,
    "leftHip": 23,
    "rightHip": 24,
    "leftKnee": 25,
    "rightKnee": 26,
    "leftAnkle": 27,
    "rightAnkle": 28,
    "rightToe": 32,
    "rightHeel": 30,
    "leftHeel": 29,
    "leftToe": 31,
    "leftWrist": 15,
    "rightWrist": 16
}
ntu_to_MP_Mapping = {
    0: "hipMid",
    1: "spineMid",
    2: "neck",
    3: "nose",
    4: "leftShoulder",
    5: "leftElbow",
    6: "leftWrist",
    7: "leftWrist",
    8: "rightShoulder",
    9: "rightElbow",
    10: "rightWrist",
    11: "rightWrist",
    12: "leftHip",
    13: "leftKnee",
    14: "leftAnkle",
    15: "leftToe",
    16: "rightHip",
    17: "rightKnee",
    18: "rightAnkle",
    19: "rightToe",
    20: "shoulderMid",
    21: "leftHandMid",
    22: "leftThumb",
    23: "rightHandMid",
    24: "rightThumb"
}
def getMidPoint(kp1, kp2):
    return ( kp1 + kp2) / 2
    
def getCalculatedMPKPs(mp_kps):
    """
    Extract and calculate MediaPipe keypoints from raw array.
    
    Args:
        mp_kps: (J, D) array where J=33 joints, D=3 coords
        
    Returns:
        mp_part_to_kp: dict mapping keypoint names to coordinates
    """
    mp_part_to_kp = {}

    # Extract keypoints by index
    for kpName, kpIdx in MPStringsToKP.items():
        mp_part_to_kp[kpName] = mp_kps[kpIdx]
    
    # Calculate midpoints
    mp_part_to_kp["shoulderMid"] = getMidPoint(mp_part_to_kp["leftShoulder"], mp_part_to_kp["rightShoulder"])
    mp_part_to_kp["hipMid"] = getMidPoint(mp_part_to_kp["leftHip"], mp_part_to_kp["rightHip"])
    mp_part_to_kp["spineMid"] = getMidPoint(mp_part_to_kp["shoulderMid"], mp_part_to_kp["hipMid"])
    mp_part_to_kp["mouthMid"] = getMidPoint(mp_part_to_kp["leftMouth"], mp_part_to_kp["rightMouth"])
    mp_part_to_kp["neck"] = getMidPoint(mp_part_to_kp["mouthMid"], mp_part_to_kp["shoulderMid"])
    mp_part_to_kp["rightHandMid"] = getMidPoint(mp_part_to_kp["rightPinky"], mp_part_to_kp["rightThumb"])
    mp_part_to_kp["leftHandMid"] = getMidPoint(mp_part_to_kp["leftPinky"], mp_part_to_kp["leftThumb"])
    mp_part_to_kp["leftWrist"] = getMidPoint(mp_part_to_kp["leftHand"], mp_part_to_kp["leftElbow"])
    
    return mp_part_to_kp

MBStringsToKP = {
    "nose": 10,
    "neck": 9,
    "shoulderMid": 8,
    "spineMid": 7,
    "hipMid": 0,
    "leftShoulder": 11,
    "leftElbow": 12,
    "leftHand": 13,
    "rightShoulder": 14,
    "rightElbow": 15,
    "rightHand": 16,
    "leftHip":4,
    "rightHip":1,
    "leftKnee":5,
    "rightKnee":2,
    "leftAnkle":6,
    "rightAnkle":3,
}

def getCalculatedMBKPs(mp_kps):
    mb_part_to_kp = {}
    for kpName, kpIdx in MBStringsToKP.items():
        mb_part_to_kp[kpName] = mp_kps[kpIdx]
        
    mb_part_to_kp["leftWrist"] = getMidPoint(mb_part_to_kp["leftHand"], mb_part_to_kp["leftElbow"])
    mb_part_to_kp["rightWrist"] = getMidPoint(mb_part_to_kp["rightHand"], mb_part_to_kp["rightElbow"])
    return mb_part_to_kp

ntu_to_MB_Mapping = {
    0: "hipMid",
    1: "spineMid",
    2: "neck",
    3: "nose",
    4: "leftShoulder",
    5: "leftElbow",
    6: "leftWrist",
    7: "leftHand",
    8: "rightShoulder",
    9: "rightElbow",
    10: "rightWrist",
    11: "rightHand",
    12: "leftHip",
    13: "leftKnee",
    14: "leftAnkle",
    15: "leftAnkle",
    16: "rightHip",
    17: "rightKnee",
    18: "rightAnkle",
    19: "rightAnkle",
    20: "shoulderMid",
    21: "leftHand",
    22: "leftHand",
    23: "rightHand",
    24: "rightHand"
}

def _is_torch(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _zeros_like_shape(*, shape, like):
    """Create zeros array/tensor with requested shape matching dtype/device of `like`."""
    if _is_torch(like):
        return torch.zeros(shape, dtype=like.dtype, device=like.device)
    return np.zeros(shape, dtype=getattr(like, "dtype", None) or np.float32)

def convertMPtoNTU(mp_kps):
    """
    Convert MediaPipe keypoints to NTU skeleton format.

    Args:
        mp_kps: (J, D) array/tensor where J=33 joints, D=3 coords

    Returns:
        ntu_kps: (25, 3) NTU skeleton format (same type as input)
    """
    ntu_kps = _zeros_like_shape(shape=(25, 3), like=mp_kps)

    # Get calculated MediaPipe keypoints (with midpoints)
    mp_part_to_kp = getCalculatedMPKPs(mp_kps)

    # Map each NTU joint to its corresponding MediaPipe keypoint
    for ntu_idx, mp_name in ntu_to_MP_Mapping.items():
        if mp_name in mp_part_to_kp:
            ntu_kps[ntu_idx] = mp_part_to_kp[mp_name]

    return ntu_kps


def convertVideoMPtoNTU(video_mp_kps):
    """
    Convert a sequence of MediaPipe keypoints to NTU skeleton format.

    Args:
        video_mp_kps: (T, J, D) array/tensor

    Returns:
        video_ntu_kps: (T, 25, 3) NTU skeleton format (same type as input)
    """
    T = video_mp_kps.shape[0]

    # MediaPipe outputs y-down (head at negative y in world coords; small y at top
    # of image in camera coords). NTU/MAMP expects y-up. Flip once per video.
    if _is_torch(video_mp_kps):
        video_mp_kps = video_mp_kps.clone()
    else:
        video_mp_kps = video_mp_kps.copy()
    video_mp_kps[..., 1] = -video_mp_kps[..., 1]

    video_ntu_kps = _zeros_like_shape(shape=(T, 25, 3), like=video_mp_kps)
    for t in range(T):
        video_ntu_kps[t] = convertMPtoNTU(video_mp_kps[t])
    return video_ntu_kps


def convertBatchVideoMPtoNTU(batch_video_mp_kps):
    """
    Convert a batch of video sequences of MediaPipe keypoints to NTU skeleton format.

    Args:
        batch_video_mp_kps: (B, T, J, D) array/tensor

    Returns:
        batch_video_ntu_kps: (B, T, 25, 3) NTU skeleton format (same type as input)
    """
    B, T, _, _ = batch_video_mp_kps.shape
    batch_video_ntu_kps = _zeros_like_shape(shape=(B, T, 25, 3), like=batch_video_mp_kps)
    for b in range(B):
        batch_video_ntu_kps[b] = convertVideoMPtoNTU(batch_video_mp_kps[b])
    return batch_video_ntu_kps

def convertMBtoNTU(mb_kps):
    """
    Convert MotionBert keypoints to NTU skeleton format.

    Args:
        mb_kps: (J, D) array/tensor where J=17 joints, D=3 coords

    Returns:
        ntu_kps: (25, 3) NTU skeleton format (same type as input)
    """
    ntu_kps = _zeros_like_shape(shape=(25, 3), like=mb_kps)

    # Get calculated MotionBert keypoints
    mb_part_to_kp = getCalculatedMBKPs(mb_kps)

    # Map each NTU joint to its corresponding MotionBert keypoint
    for ntu_idx, mp_name in ntu_to_MB_Mapping.items():
        if mp_name in mb_part_to_kp:
            ntu_kps[ntu_idx] = mb_part_to_kp[mp_name]
    
    return ntu_kps
def convertVideoMBtoNTU(video_mb_kps):
    """
    Convert a sequence of MotionBert keypoints to NTU skeleton format.

    Args:
        video_mb_kps: (T, J, D) array/tensor

    Returns:
        video_ntu_kps: (T, 25, 3) NTU skeleton format (same type as input)
    """
    T = video_mb_kps.shape[0]

    # MotionBert outputs y-down (head at negative y, feet at positive y).
    # NTU/MAMP expects y-up. Flip y once per video before per-frame mapping.
    if _is_torch(video_mb_kps):
        video_mb_kps = video_mb_kps.clone()
    else:
        video_mb_kps = video_mb_kps.copy()
    video_mb_kps[..., 1] = -video_mb_kps[..., 1]

    video_ntu_kps = _zeros_like_shape(shape=(T, 25, 3), like=video_mb_kps)
    for t in range(T):
        video_ntu_kps[t] = convertMBtoNTU(video_mb_kps[t])
    
    
    return video_ntu_kps

def convertBatchVideoMBtoNTU(batch_video_mb_kps):
    """
    Convert a batch of video sequences of MotionBert keypoints to NTU skeleton format.

    Args:
        batch_video_mb_kps: (B, T, J, D) array/tensor

    Returns:
        batch_video_ntu_kps: (B, T, 25, 3) NTU skeleton format (same type as input)
    """
    B, T, _, _ = batch_video_mb_kps.shape
    batch_video_ntu_kps = _zeros_like_shape(shape=(B, T, 25, 3), like=batch_video_mb_kps)
    for b in range(B):
        batch_video_ntu_kps[b] = convertVideoMBtoNTU(batch_video_mb_kps[b])
    return batch_video_ntu_kps
if __name__ == "__main__":
    from PoseDataset import load_video_h5

    with open('/code/jjiang23/pathml/aim2_balance/processed_files/all_h5_files.txt', 'r') as f:
        h5_files = [line.strip() for line in f.readlines()]

    # Load one MB video and convert once
    mb_kps = load_video_h5(h5_files[1], "motionBert_cropped_iou", allPhases=False)[0]  # (T, 17, 3)
    ntu_from_mb = convertVideoMBtoNTU(mb_kps)                                          # (T, 25, 3)

    # Scale + range
    spine = (ntu_from_mb[:, 20, :] - ntu_from_mb[:, 1, :])
    spine_len = spine.norm(dim=-1).mean() if hasattr(spine, "norm") \
                else ((spine ** 2).sum(-1) ** 0.5).mean()
    print("MB→NTU range:", ntu_from_mb.min().item(), ntu_from_mb.max().item(),
          " spine bone (mean):", float(spine_len))

    # Orientation: y direction
    print("head_y (joint 3):", float(ntu_from_mb[0, 3, 1]),
          " hip_y  (joint 0):", float(ntu_from_mb[0, 0, 1]),
          " toe_y  (joint 15):", float(ntu_from_mb[0, 15, 1]))

    # Optional: try to load an NTU reference from the same h5
    for key in ["ntu_cropped_iou", "kinect_iou", "ntu"]:
        try:
            ntu_kps = load_video_h5(h5_files[1], key, allPhases=False)[0]
            spine = (ntu_kps[:, 20, :] - ntu_kps[:, 1, :])
            spine_len = spine.norm(dim=-1).mean() if hasattr(spine, "norm") \
                        else ((spine ** 2).sum(-1) ** 0.5).mean()
            print(f"NTU key '{key}' range:", ntu_kps.min().item(), ntu_kps.max().item(),
                  " spine bone:", float(spine_len))
            break
        except Exception as e:
            print(f"key {key!r} not available: {e}")
            
        # --- world_mp ---
    mp_w = load_video_h5(h5_files[1], "world_mp_cropped_iou", allPhases=False)[0]   # (T, 33, 3)
    ntu_from_mpw = convertVideoMPtoNTU(mp_w)
    spine_w = ntu_from_mpw[:, 20, :] - ntu_from_mpw[:, 1, :]
    slen_w = spine_w.norm(dim=-1).mean() if hasattr(spine_w, "norm") else ((spine_w**2).sum(-1)**0.5).mean()
    print("world_mp→NTU  range:", ntu_from_mpw.min().item(), ntu_from_mpw.max().item(),
        " spine:", float(slen_w))
    print("  head_y:", float(ntu_from_mpw[0, 3, 1]),
        " hip_y:",  float(ntu_from_mpw[0, 0, 1]),
        " toe_y:",  float(ntu_from_mpw[0, 15, 1]))

    # --- camera_mp ---
    mp_c = load_video_h5(h5_files[1], "camera_mp_cropped_iou", allPhases=False)[0]
    ntu_from_mpc = convertVideoMPtoNTU(mp_c)
    spine_c = ntu_from_mpc[:, 20, :] - ntu_from_mpc[:, 1, :]
    slen_c = spine_c.norm(dim=-1).mean() if hasattr(spine_c, "norm") else ((spine_c**2).sum(-1)**0.5).mean()
    print("camera_mp→NTU range:", ntu_from_mpc.min().item(), ntu_from_mpc.max().item(),
        " spine:", float(slen_c))
    print("  head_y:", float(ntu_from_mpc[0, 3, 1]),
        " hip_y:",  float(ntu_from_mpc[0, 0, 1]),
        " toe_y:",  float(ntu_from_mpc[0, 15, 1]))

    # --- per-axis spread (for camera_mp this will reveal the anisotropic-scale problem) ---
    for name, k in [("world_mp", ntu_from_mpw), ("camera_mp", ntu_from_mpc)]:
        print(f"{name}  axis ranges: "
            f"x[{float(k[..., 0].min()):.3f},{float(k[..., 0].max()):.3f}] "
            f"y[{float(k[..., 1].min()):.3f},{float(k[..., 1].max()):.3f}] "
            f"z[{float(k[..., 2].min()):.3f},{float(k[..., 2].max()):.3f}]")
    import numpy as np

    npz = np.load("/data2/pathml/MAMP/ntu120/NTU120_XSub.npz")
    x = npz["x_train"][:100].astype(np.float32)
    print("flat shape:", x.shape)

    # Try the most common MAMP/MS-G3D flatten order: (N, T, M, V, C)
    x5 = x.reshape(100, 300, 2, 25, 3)

    # Use body 0 (body 1 is often all-zero for single-person actions)
    body0 = x5[:, :, 0, :, :]                                   # (N, T, V=25, C=3)
    print("body0 shape:", body0.shape, "range:", body0.min(), body0.max())

    # Filter zero-padded frames (NTU pads short sequences with zeros)
    valid = (body0.reshape(100, 300, -1).any(axis=-1))          # (N, T)
    print("avg valid frames per sample:", valid.sum(axis=1).mean())

    # Spine bone length on valid frames only
    spine_lens = []
    for n in range(100):
        v = valid[n]
        if v.sum() < 5:
            continue
        bone = body0[n, v, 20, :] - body0[n, v, 1, :]           # (T_valid, 3)
        spine_lens.append(np.linalg.norm(bone, axis=-1).mean())
    spine_lens = np.array(spine_lens)
    print(f"NTU120 spine length:  mean={spine_lens.mean():.4f}  "
        f"median={np.median(spine_lens):.4f}  "
        f"std={spine_lens.std():.4f}  n={len(spine_lens)}")

    # Y-axis convention check (first valid frame of first sample)
    n = 0
    t0 = np.argmax(valid[n])
    print(f"sample 0, frame {t0}:  "
        f"head_y={body0[n, t0, 3, 1]:.3f}  "
        f"hip_y={body0[n, t0, 0, 1]:.3f}  "
        f"toe_y={body0[n, t0, 15, 1]:.3f}")
    # --- Verify scale normalization against new NTU120 reference ---
    from ..models.encoders.MY_MAMP.encoder import _scale_normalize_to_ntu, NTU_REF_SPINE_LEN

    print(f"\n--- Post-scale verification (target spine = {NTU_REF_SPINE_LEN}) ---")
    for name, ntu_kps in [("MB", ntu_from_mb), ("world_mp", ntu_from_mpw)]:
        # Add batch dim, scale, drop batch dim
        if hasattr(ntu_kps, "unsqueeze"):
            b = ntu_kps.unsqueeze(0)
        else:
            b = ntu_kps[None]
        b = _scale_normalize_to_ntu(b)[0]

        spine = b[:, 20, :] - b[:, 1, :]
        if hasattr(spine, "norm"):
            slen = float(spine.norm(dim=-1).mean())
        else:
            slen = float(((spine ** 2).sum(-1) ** 0.5).mean())

        print(f"{name}  post-scale spine: {slen:.4f}   "
            f"head_y: {float(b[0, 3, 1]):+.3f}   "
            f"hip_y: {float(b[0, 0, 1]):+.3f}   "
            f"toe_y: {float(b[0, 15, 1]):+.3f}")