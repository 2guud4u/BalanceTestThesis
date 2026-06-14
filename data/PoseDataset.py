
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np

label_map = {
        b"Phase1": 0,
        b"Phase2": 1,
        b"Phase3": 2,
        b"Phase4": 3,
        b"nonphase": 4
}

kpFeaturekeys ={
    "mp_camera": "camera_mp_cropped_iou",
    "mp_world": "world_mp_cropped_iou",
    "motionBert": "motionBert_cropped_iou",
    
    }

def load_video_h5(h5_path, featureH5Key, allPhases=False, joint_indices=None):
    with h5py.File(h5_path, 'r') as f:
        if featureH5Key == "camera_mp_cropped_iou" or featureH5Key == "world_mp_cropped_iou":
            keypoints = f[featureH5Key][:][:,0][:,:,:3]   # (T, J, D)
        if featureH5Key == "motionBert_cropped_iou":
            keypoints = f[featureH5Key][:][:,:,:3]   # (T, J, D)

        labels = f['camera_poses_labels'][:]         # (T,)
        labels = np.array([label_map[lbl] for lbl in labels])
        #remove nans to 0
        keypoints = np.nan_to_num(keypoints)
    assert keypoints.shape[0] == labels.shape[0]
    #turn all nonphases to 1 and all phases to 0
    if not allPhases:
        labels = np.where(labels==4, 1, 0)
    # Optional joint subset selection
    if joint_indices is not None:
        assert max(joint_indices) < keypoints.shape[1], (
            f"joint_indices max={max(joint_indices)} >= num_joints={keypoints.shape[1]} "
            f"for h5_key={featureH5Key}. Check your config."
        )
        keypoints = keypoints[:, joint_indices, :]   # (T, len(joint_indices), D)
    return keypoints, labels

class PoseDataset(Dataset):
    """
    Dataset for pose windows after removing face keypoints.
    Provides both Dataset interface and a simple batch_gen API:
      - has_next()
      - next_batch(batch_size) -> (batch_input, batch_target, mask)
      - reset()
    """
    def __init__(self, meta, featureH5Key, window_size=256, stride=128, augment=True, joint_indices=None):
        self.window_size = window_size
        self.stride = stride
        self.augment = augment
        self.joint_indices = joint_indices
        self.items = []
        self.num_classes = 2

        for h5_path in meta:
            kps, labels = load_video_h5(h5_path, featureH5Key, joint_indices=joint_indices)  # (T, J, D), (T,)
            T = kps.shape[0]
            if T == 0:
                continue

            for s in range(0, max(1, T - 1), self.stride):
                e = s + self.window_size
                self.items.append({
                    'kp': kps[s:e],    # variable-length chunk
                    'lbl': labels[s:e]
                })

        # For Trainer compatibility
        self.list_of_examples = self.items
        self._ptr = 0

        # Count labels across all windows
        all_labels = np.concatenate([item['lbl'] for item in self.items])
        label_counts = np.bincount(all_labels.astype(int))
        label_names = {0: "Phase1", 1: "Phase2", 2: "Phase3", 3: "Phase4", 4: "nonphase"}
        
        print(f"Loaded {len(self.items)} windows from {len(meta)} videos.")
        print("Label distribution:")
        for label_id, count in enumerate(label_counts):
            print(f"  {label_names.get(label_id, f'Label_{label_id}')}: {count}")

    def __len__(self):
        return len(self.items)

    def compute_class_weights(self, extra_datasets=None, device='cpu'):
        """
        Compute class weights from this dataset and optionally additional datasets.
        
        Args:
            extra_datasets: additional PoseDataset or list of PoseDataset objects to include
            device: Device to place weights on (default: 'cpu')
        
        Returns:
            class_weights: Tensor of shape (num_classes,) normalized so sum = num_classes
        """
        datasets = [self]
        if extra_datasets is not None:
            if not isinstance(extra_datasets, list):
                extra_datasets = [extra_datasets]
            datasets += extra_datasets
        
        num_classes = self.num_classes
        class_counts = np.zeros(num_classes)
        
        # Count labels across all datasets
        for dataset in datasets:
            for item in dataset.items:
                labels = item['lbl']
                valid_labels = labels[labels >= 0]
                for c in valid_labels:
                    class_counts[int(c)] += 1
        
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1e-6)
        
        # Weight = 1 / frequency, normalized so total = num_classes
        class_weights = torch.tensor(
            1.0 / class_counts,
            dtype=torch.float32,
            device=device
        )
        class_weights = class_weights / class_weights.sum() * num_classes
        
        return class_weights
    
    def augment_fn(self, kp):
        if np.random.rand() < 0.5:
            kp[:, :, 0] *= -1.0  # horizontal flip
        if np.random.rand() < 0.5:
            kp += np.random.randn(*kp.shape) * 0.01
        return kp

    def pad_chunk(self, kp, lbl):
        """
        Pad/truncate a chunk to window_size.
        kp: (L, J, D)
        lbl: (L,)
        """
        L = kp.shape[0]
        J = kp.shape[1]
        D = kp.shape[2]

        if L >= self.window_size:
            return kp[:self.window_size], lbl[:self.window_size]

        kp_pad = np.zeros((self.window_size, J, D), dtype=kp.dtype)
        kp_pad[:L] = kp

        lbl_pad = np.full((self.window_size,), -100, dtype=lbl.dtype)
        lbl_pad[:L] = lbl

        return kp_pad, lbl_pad

    def __getitem__(self, idx):
        item = self.items[idx]

        kp = item['kp']    # (L, J, D)
        lbl = item['lbl']  # (L,)

        # kp = self.normalize(kp)
        if self.augment:
            kp = self.augment_fn(kp)

        kp, lbl = self.pad_chunk(kp, lbl)

        # flatten joints dynamically
        J = kp.shape[1]
        D = kp.shape[2]
        kp = kp.reshape(self.window_size, J * D)

        return (
            torch.from_numpy(kp).float(),     # (T, J*D)
            torch.from_numpy(lbl).long()      # (T,)
        )

    # -------------------------
    # Simple batch_gen API
    # -------------------------
    def reset(self, shuffle=False):
        """Reset internal pointer. Optionally shuffle examples."""
        if shuffle:
            np.random.shuffle(self.items)
            self.list_of_examples = self.items
        self._ptr = 0

    def has_next(self):
        return self._ptr < len(self.items)

    def next_batch(self, batch_size):
        """
        Return a batch of size up to batch_size.
        Returns:
            batch_input: Tensor (B, T, F)
            batch_target: Tensor (B, T)
            mask: Tensor (B, 1, T)  with 1 for valid frames (label != -100)
        """
        if not self.has_next():
            raise StopIteration

        batch = []
        end = min(self._ptr + batch_size, len(self.items))
        for i in range(self._ptr, end):
            kp, lbl = self.__getitem__(i)  # uses normalization/augment/pad
            batch.append((kp, lbl))
        self._ptr = end

        # stack
        batch_input = torch.stack([b[0] for b in batch], dim=0)   # (B, T, F)
        batch_target = torch.stack([b[1] for b in batch], dim=0)  # (B, T)

        mask = (batch_target != -100).float().unsqueeze(1)  # (B,1,T)

        return batch_input, batch_target, mask
