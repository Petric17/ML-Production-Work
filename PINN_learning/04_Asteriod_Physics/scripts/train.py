import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)
data_path = os.path.join(project_root, "data", "asteroid_data.csv")
df = pd.read_csv(data_path)

from src.network import AsteroidPhysicsModel

X_raw = df[['Est Dia(min)', 'Est Dia(max)', 'Relative Velocity km per sec']].values
y_raw = df['Absolute Magnitude'].values.reshape(-1, 1)

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.2, random_state=42)

#Standardize the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = torch.tensor(scaler_X.fit_transform(X_train), dtype=torch.float32)
y_train_scaled = torch.tensor(scaler_y.fit_transform(y_train), dtype=torch.float32)

X_test_scaled = torch.tensor(scaler_X.transform(X_test), dtype=torch.float32)
y_test_scaled = torch.tensor(scaler_y.transform(y_test), dtype=torch.float32)

#Initialize the model

model = AsteroidPhysicsModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Data processing loop

batch_size = 64
num_epochs = 150

print(f"Starting memory-safe training on {len(X_train_scaled)} asteroid observations...")

for epoch in range(num_epochs):
    model.train()

    #Shuffle the data
    permutation = torch.randperm(X_train_scaled.size()[0])
    epoch_losses = []

    #Stream loop 

    for i in range(0, len(X_train_scaled), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_scaled[indices], y_train_scaled[indices]

        # Forward pass & loss calculation
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass & weights update
        loss.backward()
        optimizer.step()
        
        epoch_losses.append(loss.item())

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Average Training Loss: {np.mean(epoch_losses):.4f}")

print("\nTraining completed safely! Testing AI accuracy on unseen space rocks...")

#Evaluate the Model's Real-World Performance
model.eval()
with torch.no_grad():
    test_predictions_scaled = model(X_test_scaled)
    test_predictions = scaler_y.inverse_transform(test_predictions_scaled.numpy())


#Plot the results - only a small batch of asteroids for visualization

sample_size = 50
plt.figure(figsize=(12, 6))
plt.plot(y_test[:sample_size], label="Real Measured Magnitude", color="black", marker='o', linestyle='none', alpha=0.7)
plt.plot(test_predictions[:sample_size], label="AI Predicted Magnitude", color="orange", marker='x', linestyle='none')
plt.title("Data-Driven Validation: Real Asteroid Metrics vs. AI Predictions")
plt.xlabel("Individual Unseen Asteroids")
plt.ylabel("Absolute Magnitude")
plt.legend()
plt.grid(True)

os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/asteroid_data_prediction.png", dpi=300, bbox_inches="tight")
plt.show()