MEDIAPIPE_EDGES_33 = [
    # Face (can be dropped safely)
    (0,1),(1,2),(2,3),(3,7),
    (0,4),(4,5),(5,6),(6,8),
    (9,10),

    # Torso
    (11,12),
    (11,23),(12,24),
    (23,24),

    # Left arm
    (11,13),(13,15),
    (15,17),(15,19),(15,21),

    # Right arm
    (12,14),(14,16),
    (16,18),(16,20),(16,22),

    # Left leg
    (23,25),(25,27),
    (27,29),(27,31),

    # Right leg
    (24,26),(26,28),
    (28,30),(28,32),
]


MEDIAPIPE_33 = {
    "EDGES": MEDIAPIPE_EDGES_33,
    "SHOULDER_IDX": (11, 12),
    "HIP_IDX": (23, 24),
    "skeletonType": "mp"
}

MEDIAPIPE_EDGES_22 = [
    # Arms (reindexed) and torso/legs connections only — no face/mouth edges
    (0, 1), (1, 3), (3, 5), (5, 7), (7, 9), (5, 9),      # Right arm path (was 11..20)
    (0, 2), (2, 4), (4, 6), (6, 8), (4, 8),              # Left arm path (was 11..19 variants)
    (1, 13), (0, 12), (12, 13),                          # Torso connections (was 12,23,24)
    (12, 14), (14, 16), (16, 18), (18, 20), (16, 20),    # Left leg chain (reindexed)
    (13, 15), (15, 17), (17, 19), (19, 21), (17, 21)     # Right leg chain (reindexed)
]

MEDIAPIPE_22 = {
    "EDGES": MEDIAPIPE_EDGES_22,
    "SHOULDER_IDX": (0, 1),
    "HIP_IDX": (12, 13)
}

MOTIONBERT_EDGES_17 = [
    # Right leg
    (0, 1), (1, 2), (2, 3),

    # Left leg
    (0, 4), (4, 5), (5, 6),

    # Spine
    (0, 7), (7, 8), (8, 9), (9, 10),

    # Left arm
    (8, 11), (11, 12), (12, 13),

    # Right arm
    (8, 14), (14, 15), (15, 16),
]

MOTIONBERT = {
    "EDGES": MOTIONBERT_EDGES_17,
    "HIP_IDX": (1, 4),  # Single root joint at index
    "SHOULDER_IDX": (11, 14),  # Left and right shoulders at indices
    "skeletonType": "mb"
}

NTU_25_EDGES = [
    # ---- Spine & Head ----
    (0, 1),   # SpineBase  -> SpineMid
    (1, 2),   # SpineMid   -> Neck
    (2, 3),   # Neck       -> Head

    # ---- Left Arm ----
    (2, 4),   # Neck       -> LeftShoulder
    (4, 5),   # LeftShoulder -> LeftElbow
    (5, 6),   # LeftElbow  -> LeftWrist
    (6, 7),   # LeftWrist  -> LeftHand

    # ---- Right Arm ----
    (2, 8),   # Neck       -> RightShoulder
    (8, 9),   # RightShoulder -> RightElbow
    (9, 10),  # RightElbow -> RightWrist
    (10, 11), # RightWrist -> RightHand

    # ---- Left Leg ----
    (0, 12),  # SpineBase  -> LeftHip
    (12, 13), # LeftHip   -> LeftKnee
    (13, 14), # LeftKnee  -> LeftAnkle
    (14, 15), # LeftAnkle -> LeftFoot

    # ---- Right Leg ----
    (0, 16),  # SpineBase -> RightHip
    (16, 17), # RightHip  -> RightKnee
    (17, 18), # RightKnee -> RightAnkle
    (18, 19), # RightAnkle -> RightFoot

    # ---- Upper spine refinement ----
    (2, 20),  # Neck -> SpineShoulder
    (20, 4),  # SpineShoulder -> LeftShoulder
    (20, 8),  # SpineShoulder -> RightShoulder

    # ---- Left hand details ----
    (7, 21),  # LeftHand -> LeftHandTip
    (7, 22),  # LeftHand -> LeftThumb

    # ---- Right hand details ----
    (11, 23), # RightHand -> RightHandTip
    (11, 24), # RightHand -> RightThumb
]
