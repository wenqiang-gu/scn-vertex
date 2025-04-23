import uproot
import csv
import numpy as np
import torch
from torch.utils.data import Dataset


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
            
        return (
            torch.LongTensor(coords),     # [N, 3]
            torch.FloatTensor(features),  # [N, C]
            torch.FloatTensor(targets)     
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
        torch.stack(targets_batch)       # [B, ...]
    )