import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from models.encoders.Identity import IdentityEncoder


# Map h5_key -> number of joints so we can compute feature_dim = num_joints * 3
_H5KEY_TO_JOINTS = {
    "camera_mp_cropped_iou": 33,
    "world_mp_cropped_iou": 33,
    "motionBert_cropped_iou": 17,
}


def initialize_encoder(d_cfg, e_cfg):
    """
    Build an IdentityEncoder whose out_dim matches the raw feature size.

    feature_dim is resolved in order:
        1. e_cfg["feature_dim"]          — explicit override
        2. d_cfg["num_joints"] * 3       — from data config
        3. inferred from d_cfg["h5_key"] — lookup table
    """
    if e_cfg and e_cfg.get("feature_dim"):
        feature_dim = int(e_cfg["feature_dim"])
    elif d_cfg.get("num_joints"):
        feature_dim = int(d_cfg["num_joints"]) * 3
    elif d_cfg.get("h5_key") in _H5KEY_TO_JOINTS:
        feature_dim = _H5KEY_TO_JOINTS[d_cfg["h5_key"]] * 3
    else:
        raise ValueError(
            "Cannot determine feature_dim for IdentityEncoder. "
            "Set 'feature_dim' in the encoder config, or ensure the data "
            "config contains 'num_joints' or a recognised 'h5_key'."
        )

    encoder = IdentityEncoder(feature_dim)
    print(f"✓ Identity encoder initialized (out_dim={feature_dim}, no learned parameters)")
    return encoder
