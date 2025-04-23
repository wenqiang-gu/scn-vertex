import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Import random_split
from tqdm import tqdm # Import tqdm
import argparse # Import argparse
import pandas as pd # Import pandas

from util.dataset import SparseDataset, sparse_collate_fn
from model.model import DeepVtx as default_model 

# Set a fixed random seed for random_split
generator = torch.Generator().manual_seed(42)

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--val-split-ratio', type=float, default=0.2, help='Ratio of dataset to use for validation (default: 0.2)')
    parser.add_argument('--batch-size', type=int, default=4, help='Input batch size for training (default: 4)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', action='store_true', default=False, help='Enable CUDA training')
    parser.add_argument('--num-samples', type=int, default=200, help='Number of samples to load from the dataset (default: 200)')
    # Add other arguments as needed (e.g., input_size, num_classes if they should be configurable)
    args = parser.parse_args()
    return args

# --- Training and Validation Logic ---
def train(args):
    # --- Device Setup ---
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Use args for parameters
    input_size = 1 # Example: Keep fixed or make configurable - Should match DeepVtx input if not passed
    num_classes = 1 # Example: Keep fixed or make configurable - Should match DeepVtx output if not passed
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    val_split_ratio = args.val_split_ratio

    # --- Dataset and Dataloaders ---
    print("Loading dataset...")
    full_dataset = SparseDataset(file_list='list/nuecc-39k-train.csv', num_samples=args.num_samples) # Adjust path as needed

    # Calculate split sizes
    num_total = len(full_dataset)
    num_val = int(num_total * val_split_ratio)
    num_train = num_total - num_val

    if num_train == 0 or num_val == 0:
        raise ValueError(f"Training or validation set size is zero. Adjust split ratio or dataset size. Train: {num_train}, Val: {num_val}")

    print(f"Splitting dataset: {num_train} train samples, {num_val} validation samples")
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sparse_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sparse_collate_fn) # No need to shuffle validation data

    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    # Pass the selected device to the model
    model = default_model(n_input_features=input_size, n_classes=num_classes, device=device)
    # Model is already moved to device within its __init__ based on the reformatted code

    # Define criterion and optimizer
    criterion = nn.MSELoss() # Or CrossEntropyLoss if classification targets are class indices
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- Metrics Logging ---
    metrics_log = [] # Initialize list to store epoch metrics

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(num_epochs):
        # --- Training Phase ---
        model.train() # Set model to training mode
        train_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        for coords, features, targets in train_progress_bar:
            # Move data to the selected device
            # Note: SCN expects coords as LongTensor, features as FloatTensor
            # The collate_fn should ideally handle tensor conversion and device placement,
            # but we'll do it here explicitly for clarity if needed.
            # Assuming sparse_collate_fn returns tensors:
            coords = coords.to(device)
            features = features.to(device)
            targets = targets.to(device) # Ensure targets are also on the correct device

            optimizer.zero_grad()

            # SCN expects input as a list/tuple: [coords, features]
            outputs = model([coords, features])
            loss = criterion(outputs, targets.float()) # Ensure target dtype matches output for MSELoss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / len(train_loader)
        train_progress_bar.close() # Close the training progress bar

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', leave=False)
        with torch.no_grad(): # Disable gradient calculation for validation
            ibatch = 0
            for coords, features, targets in val_progress_bar:
                ibatch += 1
                # Move data to the selected device
                coords = coords.to(device)
                features = features.to(device)
                targets = targets.to(device)

                outputs = model([coords, features])
                loss = criterion(outputs, targets.float()) # Ensure target dtype matches output
                val_loss += loss.item()

                val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

                ## Optional: Visualization of outputs
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
                import numpy as np

                # Convert tensors to CPU numpy arrays
                coords_np = coords.cpu().numpy()
                outputs_np = outputs.squeeze().cpu().numpy()
                targets_np = targets.squeeze().cpu().numpy()

                # Assuming coords is [N, 4] → [x, y, z, batch_idx]
                x = coords_np[:, 0]
                y = coords_np[:, 1]
                z = coords_np[:, 2]

                # Optional: normalize output to [0, 1] for colormap
                vmin, vmax = outputs_np.min(), outputs_np.max()
                # vmin, vmax = targets_np.min(), targets_np.max()

                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')

                sc = ax.scatter(x, y, z, c=outputs_np, cmap='viridis', vmin=vmin, vmax=vmax, s=1)
                # sc = ax.scatter(x, y, z, c=targets_np, cmap='viridis', vmin=vmin, vmax=vmax, s=1)
                fig.colorbar(sc, ax=ax, label='Model Output')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('3D Scatter Plot of Predicted Vertex Score')

                plt.tight_layout()
                # plt.show() # Use non-interactive backend
                plt.savefig(f"list/output_val_epoch_{epoch}_batch_{ibatch}.png", dpi=300)
                plt.close()

        avg_val_loss = val_loss / len(val_loader)
        # accuracy = 100 * correct / total if total > 0 else 0 # Avoid division by zero
        accuracy = 0 # Placeholder - Accuracy calculation needs review for MSELoss/SCN
        val_progress_bar.close() # Close the validation progress bar

        # Log metrics for the current epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': accuracy
        }
        metrics_log.append(epoch_metrics)

        print(f'Epoch [{epoch+1}/{num_epochs}] completed. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {accuracy:.2f}%')

    print("Finished Training")

    # --- Save Metrics ---
    metrics_df = pd.DataFrame(metrics_log)
    metrics_filename = "training_metrics.csv"
    try:
        metrics_df.to_csv(metrics_filename, index=False)
        print(f"Training metrics saved to {metrics_filename}")
    except Exception as e:
        print(f"Error saving metrics to CSV: {e}")


# --- Main Execution ---
if __name__ == '__main__':
    args = parse_args()
    train(args)
