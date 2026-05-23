import os
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import math


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# go back one level to import
import sys
sys.path.append('..')
from data.PoseDataset import PoseDataset
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from data.keypoint_constants import MEDIAPIPE_33, MEDIAPIPE_22, MOTIONBERT

# Training parameters
# WINDOW_SIZE = 256
# STRIDE = 128
WINDOW_SIZE = 120
STRIDE = 60
IN_CH = 3
NUM_CLASSES = 2 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1000
BATCH_SIZE = 32
LR = 0.0005
FEATURE_H5_KEY = "camera_mp_cropped_iou"  # Key in HDF5 files for features
KP_CONFIG = MEDIAPIPE_33  # Keypoint configuration to use (MEDIAPIPE_33, MEDIAPIPE_22, or MOTIONBERT)
SKELETON_TYPE = "mp"  # "mp" for MediaPipe, "mb" for MotionBERT

with open('/code/jjiang23/pathml/aim2_balance/processed_files/all_h5_files.txt', 'r') as f:
    h5_files = [line.strip() for line in f.readlines()]

np.random.shuffle(h5_files)

split = int(0.8 * len(h5_files))
train_files = h5_files[:split]
val_files   = h5_files[split:]

train_ds = PoseDataset(train_files,window_size=WINDOW_SIZE, featureH5Key=FEATURE_H5_KEY, kpConfig=KP_CONFIG,stride=STRIDE )
val_ds   = PoseDataset(val_files,window_size=WINDOW_SIZE,featureH5Key=FEATURE_H5_KEY, kpConfig=KP_CONFIG,stride=STRIDE, augment=False)

#test out MAMP feature extractor
import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MAMP')

from models.encoders.MY_MAMP.encoder import MAMPEncoder

try:
    mamp = MAMPEncoder(
        SKELETON_TYPE,
        '../models/encoders/MAMP/checkpoints/ntu120_xset.pth',
        '../models/encoders/MY_MAMP/pretrain_mamp_t120_layer8+5_mask90.yaml',
        map_location=DEVICE
    )
    print("✓ MAMP encoder loaded successfully")
    print(f"  Config: {mamp.config['model']}")
    print(f"  Model parameters: {sum(p.numel() for p in mamp.model.parameters())}")
except Exception as e:
    print(f"✗ Error loading MAMP encoder: {e}")
    import traceback
    traceback.print_exc()
from scripts.Trainers import MSTCN2_Trainer
from datetime import datetime

current_datetime = datetime.now()

# ========== TRAIN WITH MSTCN2_Trainer ==========
# Create trainer with MAMP encoder + MS_TCN2 segmentor
# MAMP output dimension is 256 (from config)
MAMP_OUTPUT_DIM = 256


trainer = MSTCN2_Trainer(
    encoderModel=mamp,  # Use MAMP_ENCODER wrapper (has __call__)
    num_layers_PG=11,           # PG = Prediction Generation
    num_layers_R=10,            # R = Refinement
    num_R=3,                   # Number of refinement stages
    num_f_maps=64,             # Feature maps in MS_TCN2
    dim=MAMP_OUTPUT_DIM,       # MAMP output dimension (256)
    num_classes=NUM_CLASSES,
    dataset="balance",
    split="train",
    early_stop_patience=25,
    early_stop_min_delta=0.05,
    lambda_smooth=0.005,
)

print(f"✓ Trainer initialized with MAMP encoder (output dim: {MAMP_OUTPUT_DIM}) + MS_TCN2")

# Train the model
save_dir = f"models/mamp_mstcn_{current_datetime.strftime('%Y%m%d_%H%M%S')}"
os.makedirs(save_dir, exist_ok=True)

trainer.train(
    save_dir=save_dir,
    batch_gen=train_ds,
    num_epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LR,
    device=DEVICE,
    val_batch_gen=val_ds
)

print(f"✓ Training completed! Models saved to {save_dir}")