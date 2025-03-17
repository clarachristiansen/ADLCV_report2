import os
import json
import matplotlib.pyplot as plt
import numpy as np

# === SETTINGS ===
folder_path = './loss_history'  # Change to your folder path
file_extension = '.json'

# Optional smoothing (moving average)
def smooth_curve(curve, window_size=10):
    """Simple moving average for smoothing."""
    if len(curve) < window_size:
        return curve  # Can't smooth if the window is bigger than the data
    return np.convolve(curve, np.ones(window_size)/window_size, mode='valid')

# === LOAD JSON FILES ===
loss_curves = {}  # Dictionary: filename (token) -> array of losses

for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        token_name = filename.replace(file_extension, '')
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)
            loss_curves[token_name] = data

# === PLOTTING ===
plt.figure(figsize=(10, 6))

for token, losses in loss_curves.items():
    losses = np.array(losses)
    smoothed_losses = smooth_curve(losses, window_size=10)
    plt.plot(smoothed_losses, label=token, linewidth=2)

plt.title('Textual Inversion Loss Curves')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Token')
plt.tight_layout()

plt.savefig("loss_history.png")
