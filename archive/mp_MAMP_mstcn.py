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
import yaml
from datetime import datetime

current_datetime = datetime.now()
with open('/code/jjiang23/pathml/aim2_balanceV2/configs/MP_MAMP_MSTCN.yml', 'r') as f:
    config = yaml.safe_load(f)

d_cfg = config['dataloader_config']
t_cfg = config['trainer_config']

save_dir = config["save_dir"]+current_datetime.strftime('%Y%m%d_%H%M%S')
os.makedirs(save_dir, exist_ok=True)

with open('/code/jjiang23/pathml/aim2_balance/processed_files/all_h5_files.txt', 'r') as f:
    h5_files = [line.strip() for line in f.readlines()]

np.random.shuffle(h5_files)

split = int(d_cfg["split_ratio"] * len(h5_files))
train_files = h5_files[:split]
val_files   = h5_files[split:]

train_ds = PoseDataset(train_files, window_size=d_cfg["window_size"], featureH5Key=d_cfg["h5_key"], stride=d_cfg["stride"], augment=True)
val_ds   = PoseDataset(val_files,  window_size=d_cfg["window_size"], featureH5Key=d_cfg["h5_key"], stride=d_cfg["stride"], augment=False)

# save files to txt for record keeping
with open(os.path.join(save_dir, "train_files.txt"), 'w') as f:
    for file in train_files:
        f.write(file + '\n')
        
with open(os.path.join(save_dir, "val_files.txt"), 'w') as f:
    for file in val_files:
        f.write(file + '\n')

class_weights = train_ds.compute_class_weights(device=t_cfg["device"])
print(f"Class weights: {class_weights}")


#test out MAMP feature extractor
import sys
from pathlib import Path

# Add MAMP to path before importing
sys.path.insert(0, '../models/encoders/MAMP')

from models.encoders.MY_MAMP.encoder import MAMPFeatureEncoder

try:
    mamp = MAMPFeatureEncoder(
        d_cfg["h5_key"],
        '../models/encoders/MAMP/checkpoints/ntu120_xset.pth',
        '../models/encoders/MY_MAMP/pretrain_mamp_t120_layer8+5_mask90.yaml',
        map_location=t_cfg["device"]
    )
    print("✓ MAMP encoder loaded successfully")
    print(f"  Config: {mamp.config['model']}")
    print(f"  Model parameters: {sum(p.numel() for p in mamp.model.parameters())}")
except Exception as e:
    print(f"✗ Error loading MAMP encoder: {e}")
    import traceback
    traceback.print_exc()

from scripts.Trainers import Simple_MSTCN2_Trainer


trainer = Simple_MSTCN2_Trainer(
    encoder=mamp,
    class_weights=class_weights,
    early_stop_patience=t_cfg["early_stop_patience"],
    early_stop_min_delta=t_cfg["early_stop_min_delta"],
)
print(f"✓ Trainer initialized with MAMP encoder (out_dim={mamp.out_dim})")

trainer.train(
    save_dir=save_dir,
    batch_gen=train_ds,
    val_batch_gen=val_ds,        # ← required for early stopping to work
    num_epochs=t_cfg["epochs"],
    batch_size=t_cfg["batch_size"],
    learning_rate=float(t_cfg["lr"]),
    device=t_cfg["device"],
    lambda_smooth=t_cfg["lambda_smooth"],
    time_alignment=t_cfg["time_alignment"]
)
print(f"✓ Training completed! Models saved to {save_dir}")


