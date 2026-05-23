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
    mp_part_to_kp = {}
    for kpName, kpIdx in MBStringsToKP.items():
        mp_part_to_kp[kpName] = mp_kps[kpIdx]
    return mp_part_to_kp

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
    for ntu_idx, mp_name in ntu_to_MP_Mapping.items():
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
    video_ntu_kps = _zeros_like_shape(shape=(T, 25, 3), like=video_mb_kps)
    for t in range(T):
        video_ntu_kps[t] = convertMBtoNTU(video_mb_kps[t])
    return video_ntu_kps

if __name__ == "__main__":
    from data.PoseDataset import load_video_h5
    
    with open('/code/jjiang23/pathml/aim2_balance/processed_files/all_h5_files.txt', 'r') as f:
        h5_files = [line.strip() for line in f.readlines()]
    keypoints = load_video_h5(h5_files[1], "camera_mp_cropped_iou", allPhases=False)[0]  # (T, J, D)
