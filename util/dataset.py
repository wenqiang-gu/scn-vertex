import uproot
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

def voxelize(coords, features, targets=None, resolution=0.5):
    """
    Voxelize 3D point cloud data.

    Args:
        coords:   [N, D] numpy array of raw coordinates (e.g., x, y, z)
        features: [N, C] numpy array of point-wise features
        targets:  [N, T] numpy array of point-wise target values (optional)
        resolution: float, size of one voxel in real units

    Returns:
        voxel_coords:   [M, D] voxel indices (int)
        voxel_features: [M, C] averaged features
        voxel_targets:  [M, T] max target per voxel (if targets provided), else None
    """
    coords = np.asarray(coords)
    features = np.asarray(features)
    assert coords.shape[0] == features.shape[0], "coords and features must have same length"

    if targets is not None:
        targets = np.asarray(targets)
        assert targets.shape[0] == coords.shape[0], "targets must match coords length"

    # Shift coordinates to be non-negative
    coords_shifted = coords - np.min(coords, axis=0)

    # Convert to voxel grid indices
    voxel_indices = (coords_shifted / resolution).astype(int)

    # Accumulate features
    feat_sum = defaultdict(lambda: np.zeros(features.shape[1], dtype=np.float32))
    feat_count = defaultdict(lambda: 0)

    # Accumulate targets by max
    target_dict = dict() if targets is not None else None

    for i in range(voxel_indices.shape[0]):
        key = tuple(voxel_indices[i])

        feat_sum[key] += features[i]
        feat_count[key] += 1

        if targets is not None:
            if key in target_dict:
                target_dict[key] = np.maximum(target_dict[key], targets[i])
            else:
                target_dict[key] = targets[i].copy()

    # Build outputs
    voxel_coords = []
    voxel_features = []
    voxel_targets = [] if targets is not None else None

    for key in feat_sum:
        voxel_coords.append(list(key))
        voxel_features.append(feat_sum[key] / feat_count[key])
        if targets is not None:
            voxel_targets.append(target_dict[key])

    voxel_coords = np.array(voxel_coords, dtype=np.int32)
    voxel_features = np.array(voxel_features, dtype=np.float32)
    voxel_targets = np.array(voxel_targets, dtype=np.float32) if targets is not None else None

    return voxel_coords, voxel_features, voxel_targets

class SparseDataset(Dataset):
    def __init__(self, file_list, num_samples=None):
        self.file_list = []
        self.true_vertex = []
        with open(file_list) as f:
            dataset = csv.reader(f, delimiter=' ')
            for metainfo in dataset:
                if num_samples is not None and len(self.file_list) >= num_samples:
                    break
                try:
                    with uproot.open(metainfo[0]) as file:
                        self.file_list.append(metainfo[0])
                        xyz = (float(metainfo[2]), float(metainfo[3]), float(metainfo[4])) # xyz of true vertex
                        self.true_vertex.append(np.array(xyz))
                except Exception as e:
                    print(f"Error opening file {metainfo[0]}: {e}")
                    return
                
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        def dist2prob(pt, t, sigma = 1.0): # default regularization 1.0
            return np.exp(-0.5 * (np.linalg.norm(pt - t, axis=1) / sigma) ** 2)
        
        with uproot.open(self.file_list[idx]) as file:
            # Load the data from the file
            data = file['T_rec_charge_blob'].arrays(library="np")
            x,y,z,q = data['x'], data['y'], data['z'], data['q']
            coords = np.stack((x,y,z), axis=1) # shape: [N, 3]
            features = q.reshape(-1, 1) # shape: [N, C] C=1
            targets = dist2prob(coords, self.true_vertex[idx]) # shape: (N,), heatmap values
            v_coords, v_feats, v_tgts = voxelize(coords, features, targets, resolution=0.5) # Voxelize the data
            
        return (
            torch.LongTensor(v_coords),     # [N, 3]
            torch.FloatTensor(v_feats),  # [N, C]
            torch.FloatTensor(v_tgts)   # [N, T]    
        )

def sparse_collate_fn(batch):
    coords_batch = []
    feats_batch = []
    targets_batch = []

    for i, (coords, feats, target) in enumerate(batch):
        # Add batch index to coords
        batch_index = torch.full((coords.shape[0], 1), i, dtype=torch.long)
        coords_b = torch.cat([coords, batch_index], dim=1)  # [N, 4] for 3D

        coords_batch.append(coords_b)
        feats_batch.append(feats)
        targets_batch.append(target)

    return (
        torch.cat(coords_batch, dim=0),  # [total_voxels, 4]
        torch.cat(feats_batch, dim=0),   # [total_voxels, C]
        torch.cat(targets_batch)       # [B, ...]
    )