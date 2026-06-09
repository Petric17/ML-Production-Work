import os
import sys
import torch
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.network import DampedPendulumModel

# Load the model
model = DampedPendulumModel()

# Extract the trained weights from the model
w_input_hidden = model.input_layer.weight.detach().numpy() 
w_hidden_output = model.output_layer.weight.detach().numpy() 

# Setup the Canvas
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 7), facecolor='#060b14') 
ax.set_facecolor('#060b14')

# Define Coordinates
x_input, x_hidden, x_output = 0, 1, 2
num_displayed_hidden = 16
y_hidden_nodes = torch.linspace(-1, 1, num_displayed_hidden).numpy()

# Draw Layer 1 (Input -> Hidden)
for i, y_h in enumerate(y_hidden_nodes):
    weight_val = w_input_hidden[i * 4, 0]
    color = '#ff4757' if weight_val > 0 else '#00d2d3'
    raw_alpha = min(abs(weight_val) * 1.5, 1.0)
    alpha = max(raw_alpha, 0.3)
    ax.plot([x_input, x_hidden], [0, y_h], color=color, alpha=alpha, linewidth=1.5, zorder=1)

# Draw Layer 2 (Hidden -> Output)
for i, y_h in enumerate(y_hidden_nodes):
    weight_val = w_hidden_output[0, i * 4]
    color = '#ff4757' if weight_val > 0 else '#00d2d3'
    raw_alpha = min(abs(weight_val) * 1.5, 1.0)
    alpha = max(raw_alpha, 0.3)
    ax.plot([x_hidden, x_output], [y_h, 0], color=color, alpha=alpha, linewidth=1.5, zorder=1)

# Draw Nodes
ax.scatter(x_input, 0, color='#ff6b6b', s=400, edgecolors='white', linewidths=1.5, zorder=2)
ax.scatter([x_hidden]*num_displayed_hidden, y_hidden_nodes, color='#54a0ff', s=150, edgecolors='#2e86de', zorder=2)
ax.scatter(x_output, 0, color='#1dd1a1', s=400, edgecolors='white', linewidths=1.5, zorder=2)

# Labels & Clean Headings
ax.text(x_input, -0.18, "Time (t)", color='white', fontsize=12, ha='center', fontweight='bold')
ax.text(x_hidden, 1.08, "64-Neuron Hidden Layer", color='#54a0ff', fontsize=12, ha='center', fontweight='bold')
ax.text(x_hidden, 1.00, "(Active Damped Physics Cascade)", color='#54a0ff', fontsize=10, ha='center', alpha=0.8)
ax.text(x_output, -0.18, "Damped Angle (θ)", color='white', fontsize=12, ha='center', fontweight='bold')

ax.set_title("Neural Architecture Map: Damped Oscillator Network", 
             color='white', fontsize=15, pad=30, fontweight='bold', y=1.05)

ax.set_ylim(-1.2, 1.2)
ax.axis('off')
plt.tight_layout()

# Save and display
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/damped_network_architecture.png", dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()