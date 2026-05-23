import os
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import h5py
import numpy as np

def getInstitutionFromPath(path):
    # Assuming path format: /some/path/institution_videoid.h5
    path_parts = path.split('/')
    institution = path_parts[-3]  # Get the last part of the path before the underscore
    return institution

with open('/code/jjiang23/pathml/aim2_balance/processed_files/all_h5_files.txt', 'r') as f:
    h5_files = [line.strip() for line in f.readlines()]

all_classes = set()
all_institutions = set()

video_labels = []
video_institutions = []

# First pass: collect unique values
for path in h5_files:
    with h5py.File(path, 'r') as f:
        labels = f["camera_poses_labels"][:]  # frame-level labels
        vid_path = f.attrs["video_path"]  
        inst = getInstitutionFromPath(vid_path)

        all_classes.update(np.unique(labels))
        all_institutions.add(inst)

# Create index maps
class_to_idx = {c: i for i, c in enumerate(sorted(all_classes))}
inst_to_idx = {s: i for i, s in enumerate(sorted(all_institutions))}

num_classes = len(class_to_idx)
num_insts = len(inst_to_idx)
print(f"Found {num_classes} unique classes: {class_to_idx}")
print(f"Found {num_insts} unique institutions: {inst_to_idx}")
#print how many of video has each phase
phase_counts = {c: 0 for c in sorted(all_classes)}
for path in h5_files:
    with h5py.File(path, 'r') as f:
        labels = f["camera_poses_labels"][:]  # frame-level labels
        for c in np.unique(labels):
            phase_counts[c] += 1
print("Video counts per class:")
for c, count in phase_counts.items():
    print(f"  Class {c}: {count} videos")   

Y = []

for path in h5_files:
    with h5py.File(path, 'r') as f:
        labels = f["camera_poses_labels"][:]  # frame-level labels
        vid_path = f.attrs["video_path"]  
        inst = getInstitutionFromPath(vid_path)

    vec = np.zeros(num_classes + num_insts)

    # mark which classes appear in this video
    for c in np.unique(labels):
        vec[class_to_idx[c]] = 1

    # mark institution
    vec[num_classes + inst_to_idx[inst]] = 1

    Y.append(vec)

Y = np.array(Y)
print(Y.shape, Y)
mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)

splits = []

for fold, (train_idx, val_idx) in enumerate(mskf.split(h5_files, Y)):
    train_files = [h5_files[i] for i in train_idx]
    val_files = [h5_files[i] for i in val_idx]

    splits.append((train_files, val_files))

import json

output = {}
for i, (train, val) in enumerate(splits):
    output[f"fold_{i}"] = {
        "train": train,
        "val": val
    }

with open("splits.json", "w") as f:
    json.dump(output, f, indent=2)