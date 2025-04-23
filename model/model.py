"""
Defines the DeepVtx model architecture using SparseConvNet (SCN).
"""

import torch
import torch.nn as nn
import sparseconvnet as scn
from typing import List, Union

# --- Constants ---
REPS = 2  # U-Net convolution block repetition factor
M = 16    # U-Net base number of features
N_PLANES = [M, 2 * M, 4 * M, 8 * M, 16 * M]  # U-Net features per level

class DeepVtx(nn.Module):
    """
    Deep Vertex (DeepVtx) model using a SparseConvNet U-Net architecture.

    This model is designed for 3D sparse data, potentially for tasks like
    vertex finding or classification in high-energy physics or similar domains.

    Attributes:
        dimension (int): The spatial dimension of the input data (default: 3).
        device (str or torch.device): The device to run the model on ('cuda' or 'cpu').
        spatial_size (int): The spatial size of the input volume. Assumed to be cubic.
                            It's recommended this be a power of 2 for SCN compatibility.
        n_input_features (int): Number of input features per sparse point.
        n_classes (int): Number of output classes for the final linear layer.
        sparseModel (scn.Sequential): The main SparseConvNet U-Net model.
        linear (nn.Linear): Final linear layer for classification.
    """
    def __init__(
        self,
        dimension: int = 3,
        device: Union[str, torch.device] = 'cuda',
        spatial_size: int = 4096,
        n_input_features: int = 1,
        n_classes: int = 1
    ):
        """
        Initializes the DeepVtx model.

        Args:
            dimension (int): Spatial dimension of the input data. Defaults to 3.
            device (Union[str, torch.device]): Device for model parameters ('cuda' or 'cpu').
                                                Defaults to 'cuda'.
            spatial_size (int): Cubic spatial size of the input volume. Defaults to 4096.
                                Needs careful consideration based on SCN requirements.
            n_input_features (int): Number of features for each input point. Defaults to 1.
            n_classes (int): Number of output classes. Defaults to 1.
        """
        super().__init__() # Use super() for cleaner inheritance

        self.dimension = dimension
        self.device = device
        self.spatial_size = spatial_size
        self.n_input_features = n_input_features
        self.n_classes = n_classes

        # Ensure spatial_size is appropriate (optional check, depends on SCN mode)
        # Note: SCN mode 3 (used below) might have specific requirements.
        # if not (spatial_size & (spatial_size - 1) == 0) and spatial_size != 0:
        #     print(f"Warning: spatialSize ({spatial_size}) is not a power of 2, which might be required by SCN.")

        # Define the main SCN U-Net model
        self.sparseModel = scn.Sequential(
            scn.InputLayer(
                dimension=self.dimension,
                spatial_size=torch.LongTensor([self.spatial_size] * self.dimension),
                mode=3 # Mode 3 often used for coordinate-based input
            ),
            scn.SubmanifoldConvolution(
                dimension=self.dimension,
                nIn=self.n_input_features,
                nOut=M,
                filter_size=3, # Standard 3x3x... convolution
                bias=False # Often False when followed by BatchNorm
            ),
            # The core U-Net structure
            scn.UNet(
                dimension=self.dimension,
                reps=REPS,
                nPlanes=N_PLANES,
                residual_blocks=False, # Or True, depending on desired architecture
                downsample=[2, 2] # Downsampling factor at each level
            ),
            scn.BatchNormReLU(M), # BatchNorm and ReLU after U-Net
            scn.OutputLayer(dimension=self.dimension)
        ).to(self.device)

        # Define the final linear layer
        self.linear = nn.Linear(
            in_features=M, # Output features from U-Net's final BatchNormReLU
            out_features=self.n_classes
        ).to(self.device)

    def forward(self, x: Union[List, tuple]) -> torch.Tensor:
        """
        Performs the forward pass of the DeepVtx model.

        Args:
            x (Union[List, tuple]): Input data for SCN, typically a list or tuple containing
                                     [coordinates, features, batch_size (optional)].
                                     Coordinates should be shape (N, dimension+1)
                                     where the last column is the batch index.
                                     Features should be shape (N, n_input_features).

        Returns:
            torch.Tensor: The output tensor after the linear layer and sigmoid activation.
                          Shape will be (N, n_classes), where N is the number of
                          active sites output by the SCN model.
        """
        # Pass input through the SparseConvNet model
        sparse_output_features = self.sparseModel(x)

        # Pass the features from SCN through the final linear layer
        output = self.linear(sparse_output_features)

        # Apply sigmoid activation (common for binary or multi-label classification)
        # Use Softmax for multi-class classification if classes are mutually exclusive
        output = torch.sigmoid(output)

        return output
