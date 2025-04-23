# SCN-Vertex: SparseConvNet-Based Vertex Finding in LArTPC

**SCN-Vertex** is a 3D vertex finding framework for Liquid Argon Time Projection Chambers (LArTPCs), based on [SparseConvNet](https://github.com/facebookresearch/SparseConvNet). It implements a sparse convolutional neural network to directly localize interaction vertices from 3D ionization charge distributions, achieving high efficiency and sub-voxel resolution.

This repository implements the vertex finding algorithm described in [arXiv:2110.13961](https://arxiv.org/abs/2110.13961), Section 2.5.

## Overview

SCN-Vertex accepts sparse 3D point cloud data (e.g., reconstructed ionization charge) and produces voxel-level probability heatmaps indicating the predicted interaction vertex location.

### Key Features

- Sparse 3D convolution using SparseConvNet
- High-resolution, memory-efficient processing of LArTPC data
- Point cloud input with optional voxelization
- Batch processing support with PyTorch DataLoader
- Modular design for training and evaluation

## Installation

1. Create and activate a Python virtual environment:

```bash
python3 -m venv scn-venv
source scn-venv/bin/activate
```

2. Install required Python packages:

```bash
pip install torch numpy matplotlib tqdm
```

3. Clone and install SparseConvNet:

```bash
git clone https://github.com/facebookresearch/SparseConvNet.git
cd SparseConvNet
python setup.py install
```

## Usage

### Data Format

The input data should be stored in `.npz` format with the following fields:

- `coords`: shape `[N, 3]` — 3D spatial coordinates of each point
- `features`: shape `[N, C]` — per-point features (e.g., charge)
- `target`: shape `[N, 1]` — per-point target values (e.g., probability of vertex)

Use the `voxelize()` utility to preprocess raw point clouds into sparse voxel format.

### Training

Run the training script:

```bash
python train.py
```

Training parameters such as batch size, learning rate, and number of epochs can be adjusted within `train.py`.

## Citation

If you use this code in your work, please cite:

```
@article{MicroBooNE:2021ojx,
    author = "Abratenko, P. and others",
    collaboration = "MicroBooNE",
    title = "{Wire-cell 3D pattern recognition techniques for neutrino event reconstruction in large LArTPCs: algorithm description and quantitative evaluation with MicroBooNE simulation}",
    reportNumber = "FERMILAB-PUB-21-509-ND",
    doi = "10.1088/1748-0221/17/01/P01037",
    journal = "JINST",
    volume = "17",
    number = "01",
    pages = "P01037",
    year = "2022"
}
```

## Acknowledgments

This work is based on the vertex finding framework developed as part of the [Wire-Cell](https://lar.bnl.gov/wire-cell/) and SparseConvNet. Special thanks to the original authors of [SparseConvNet](https://github.com/facebookresearch/SparseConvNet).

