import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # Import random_split
from tqdm import tqdm # Import tqdm
import argparse # Import argparse
import pandas as pd # Import pandas

# Import DummyDataset from the util folder
from util.dataset import SparseDataset, sparse_collate_fn
from model.model import DeepVtx as default_model 

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--val_split_ratio', type=float, default=0.2, help='Ratio of dataset to use for validation (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=4, help='Input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    # Add other arguments as needed (e.g., input_size, num_classes if they should be configurable)
    args = parser.parse_args()
    return args

# --- Training and Validation Logic ---
def train(args):
    # Use args for parameters
    input_size = 10 # Example: Keep fixed or make configurable
    num_classes = 2 # Example: Keep fixed or make configurable
    batch_size = args.batch_size
    learning_rate = args.lr
    num_epochs = args.epochs
    val_split_ratio = args.val_split_ratio

    # --- Dataset and Dataloaders ---
    print("Loading dataset...")
    full_dataset = SparseDataset(file_list='list/nuecc-39k-train.csv') # Adjust path as needed

    # Calculate split sizes
    num_total = len(full_dataset)
    num_val = int(num_total * val_split_ratio)
    num_train = num_total - num_val

    if num_train == 0 or num_val == 0:
        raise ValueError(f"Training or validation set size is zero. Adjust split ratio or dataset size. Train: {num_train}, Val: {num_val}")

    print(f"Splitting dataset: {num_train} train samples, {num_val} validation samples")
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=sparse_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=sparse_collate_fn) # No need to shuffle validation data

    # --- Model, Loss, Optimizer ---
    print("Initializing model...")
    model = default_model(input_size, num_classes)
    criterion = nn.MSELoss()
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
            optimizer.zero_grad()

            outputs = model([coords, features])
            loss = criterion(outputs, targets)
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
            for coords, features, targets in val_progress_bar:
                outputs = model([coords, features])
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
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
