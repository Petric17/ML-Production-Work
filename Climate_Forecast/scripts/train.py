import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

# Check your filenames/class names match these imports exactly
from src.dataset_loader import multivariate_sequences
from src.ensemble_network import EnsembleModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Active device: {str(device).upper()}')

# Point out the data path
csv_data_path = os.path.join(project_root, "data", "geocoded_data.csv")
active_features = ['Month', 'Day', 'Temp', 'Pressure', 'Humidity']
target_variable = 'Temp'
seq_length = 30

# Slice the data into 30 day multi-variate sequences
X, y, scaler_X, scaler_y, target_idx = multivariate_sequences(
    data_path=csv_data_path,
    feature_cols=active_features,
    target_col=target_variable,
    seq_length=seq_length
)

# Split the data chronologically
split_index = int(len(X) * 0.85)
X_train, X_val = X[:split_index].to(device), X[split_index:].to(device)
y_train, y_val = y[:split_index].to(device), y[split_index:].to(device)

# --- PATCH 1: Pass the dynamic feature length into the model ---
model = EnsembleModel(num_features=len(active_features)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

batch_size = 128 # Great choice for your hardware experiment!
epochs = 50
physics_lambda = 0.5

print(f"Setup verified. Training model on {len(X_train)} historical sequences...")
print("-----------------------------------------------------")

# Optimization loop 
for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(X_train.size()[0])
    epoch_losses = []
    
    for i in range(0, X_train.size()[0], batch_size):
        optimizer.zero_grad()
        
        indices = permutation[i : i + batch_size]
        batch_x, batch_y = X_train[indices], y_train[indices]
        
        predictions = model(batch_x)
        data_loss = nn.L1Loss()(predictions, batch_y)
        
        # PINN Physics: tracking the dynamic column position of Temperature
        yesterday_temps = batch_x[:, -1, target_idx].unsqueeze(-1)
        true_physics_slope = batch_y - yesterday_temps
        predicted_physics_slope = predictions - yesterday_temps
        physics_loss = torch.mean(torch.abs(predicted_physics_slope - true_physics_slope))
        
        total_loss = data_loss + (physics_lambda * physics_loss)
        
        total_loss.backward()
        optimizer.step()
        epoch_losses.append(total_loss.item())

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/{epochs} | Total PINN Loss (Data + Physics): {np.mean(epoch_losses):.5f}")

print("-----------------------------------------------------")
print(f"Optimization loop complete!")

# Testing the model on unseen future records
model.eval()
with torch.no_grad():
    val_predictions_scaled = model(X_val).cpu().numpy()

# --- PATCH 2 & 3: Direct 1D un-scaling with our dedicated target scaler ---
real_predictions = scaler_y.inverse_transform(val_predictions_scaled).flatten()
real_targets = scaler_y.inverse_transform(y_val.cpu().numpy()).flatten()

# Plot a window of 60 days of predictions vs. actual values
plot_range = 60
plt.figure(figsize=(12, 6))
plt.plot(real_targets[:plot_range], label="True Temperatures", color='#2c3e50', linewidth=2)
plt.plot(real_predictions[:plot_range], label="Ensemble Forecast", color='#e67e22', linewidth=2, linestyle='--')
plt.title("Multi-Variate Ensemble Model Performance on Climate Data")
plt.xlabel("Days into the Future")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True, alpha=0.4)

os.makedirs("outputs", exist_ok=True)
output_img_path = "outputs/climate_forecast.png"
plt.savefig(output_img_path, dpi=300, bbox_inches="tight")

print(f"Success! Forecast visualization saved to {output_img_path}")
print(f"Script completed successfully!")
plt.show()