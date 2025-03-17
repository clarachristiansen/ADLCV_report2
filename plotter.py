import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import matplotlib.ticker as mticker

# === SETTINGS ===
folder_path = './loss_history'  # Change to your folder path
file_extension = '.json'

# Optional smoothing (moving average)
def smooth_curve(curve, window_size=10):
    """Simple moving average for smoothing."""
    if len(curve) < window_size:
        return curve
    return np.convolve(curve, np.ones(window_size)/window_size, mode='valid')

# === LOAD JSON FILES ===
loss_curves = {}

for filename in os.listdir(folder_path):
    if filename.endswith(file_extension):
        token_name = filename.replace(file_extension, '')
        with open(os.path.join(folder_path, filename), 'r') as f:
            data = json.load(f)
            loss_curves[token_name] = data

# === STYLISH PLOTTING ===

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')  # Clean white background with subtle grid
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.titlesize'] = 18
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

colors = plt.cm.viridis(np.linspace(0, 1, len(loss_curves)))

fig, ax = plt.subplots(figsize=(12, 7))

for idx, (token, losses) in enumerate(loss_curves.items()):
    losses = np.array(losses)
    smoothed_losses = smooth_curve(losses, window_size=10)
    ax.plot(smoothed_losses, label=token, linewidth=2.5, color=colors[idx])

# Customizations
ax.set_title('Textual Inversion Loss Curves', fontsize=20, weight='bold')
ax.set_xlabel('Training Step')
ax.set_ylabel('Loss')
ax.grid(True, linestyle='--', alpha=0.4)

# Add subtle ticks
ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

# Remove chart border spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Legend customization
legend = ax.legend(title='Token', title_fontsize=13, frameon=False, loc='upper right')

# Tight layout
fig.tight_layout()

# Save with transparent background
plt.savefig("loss_history_stylish.png", dpi=300, bbox_inches='tight', transparent=True)
plt.show()
