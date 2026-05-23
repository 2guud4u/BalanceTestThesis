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
import json 
current_datetime = datetime.now()
with open('/code/jjiang23/pathml/aim2_balanceV2/configs/MP_MSGCN_MSTCN.yml', 'r') as f:
    config = yaml.safe_load(f)

d_cfg = config['dataloader_config']
t_cfg = config['trainer_config']
e_cfg = config['encoder_config']
save_dir = config["save_dir"]+current_datetime.strftime('%Y%m%d_%H%M%S')
os.makedirs(save_dir, exist_ok=True)

#make copy of current config in save_dir for record keeping
with open(os.path.join(save_dir, "config.yml"), 'w') as f:
    yaml.dump(config, f)
with open('/code/jjiang23/pathml/aim2_balanceV2/data/splits.json', 'r') as f:
    splits = json.load(f)

for split_name, split_files in splits.items():
    train_files = split_files["train"]
    val_files = split_files["val"]
    train_ds = PoseDataset(train_files, window_size=d_cfg["window_size"], featureH5Key=d_cfg["h5_key"], stride=d_cfg["stride"], augment=True)
    val_ds   = PoseDataset(val_files,  window_size=d_cfg["window_size"], featureH5Key=d_cfg["h5_key"], stride=d_cfg["stride"], augment=False)
    
    class_weights = train_ds.compute_class_weights(device=t_cfg["device"])
    print(f"Class weights: {class_weights}")


    #test out MAMP feature extractor
    import sys
    from pathlib import Path

    # Add MAMP to path before importing
    sys.path.insert(0, '../models/encoders/MS-GCN')

    from models.encoders.GCN.MS_GCN_ENCODER import MSGCNEncoder

    encoder = MSGCNEncoder(
        e_cfg["graph_args"],
        num_joints=e_cfg["num_joints"],
        in_channels=e_cfg["in_channels"],
        filters=e_cfg["filters"],       
        dil=e_cfg["dil"]
    )

    from scripts.Trainers import Simple_MSTCN2_Trainer


    trainer = Simple_MSTCN2_Trainer(
        encoder=encoder,
        class_weights=class_weights,
        early_stop_patience=t_cfg["early_stop_patience"],
        early_stop_min_delta=t_cfg["early_stop_min_delta"],
        early_stop_monitor=t_cfg["early_stop_monitor"],
    )
    print(f"✓ Trainer initialized with MAMP encoder (out_dim={encoder.out_dim})")

    trainer.train(
        save_dir=save_dir+f"/{split_name}",  # save_dir per split
        batch_gen=train_ds,
        val_batch_gen=val_ds,        # ← required for early stopping to work
        num_epochs=t_cfg["epochs"],
        batch_size=t_cfg["batch_size"],
        learning_rate=float(t_cfg["lr"]),
        device=t_cfg["device"],
        lambda_smooth=t_cfg["lambda_smooth"],
        weight_decay=float(t_cfg.get("weight_decay", 0.0)),
    )
    print(f"✓ Training completed! Models saved to {save_dir}")
    













