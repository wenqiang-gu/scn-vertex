import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load the data from the CSV file
try:
    data = pd.read_csv('training_metrics.csv')
except FileNotFoundError:
    print("Error: training_metrics.csv not found in the current directory.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# Check if required columns exist
required_columns = ['epoch', 'train_loss', 'val_loss', 'val_accuracy']
if not all(col in data.columns for col in required_columns):
    print(f"Error: CSV file must contain the columns: {required_columns}")
    exit()

# Create a figure and a set of subplots
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plotting Loss
ax[0].plot(data['epoch'], data['train_loss'], label='Train Loss', marker='o')
ax[0].plot(data['epoch'], data['val_loss'], label='Validation Loss', marker='x')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training and Validation Loss')
ax[0].legend()
ax[0].grid(True)
# Force x-axis ticks to be integers
ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))

# Plotting Accuracy
ax[1].plot(data['epoch'], data['val_accuracy'], label='Validation Accuracy', marker='s', color='green')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy')
ax[1].set_title('Validation Accuracy')
ax[1].legend()
ax[1].grid(True)
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

# Adjust layout and display the plot
plt.tight_layout()
plt.show()

print("Plot displayed.")
