import sys
import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.network import PendulumModel 
import numpy as np
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, project_root)

#initialize the model from the network.py file

model = PendulumModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

t_physics = torch.linspace(0, 10, 500).view(-1, 1).requires_grad_(True)
t_initial = torch.tensor([[0.0]], requires_grad=True)
true_initial_angle = torch.tensor([[1.0]])


#Optimization loop 
print("Starting training...")
    
for epoch in range(7001):
    optimizer.zero_grad()

    #Calculate the Initial Condition Loss
    predicted_initial_angle = model(t_initial)
    loss_boundary = torch.mean((predicted_initial_angle - true_initial_angle)**2)

    #Calculate the Physics Loss
    predicted_position = model(t_physics)

    # First derivative: d(Position) / d(Time) = Velocity
    predicted_velocity = torch.autograd.grad(
        outputs=predicted_position,
        inputs=t_physics,
        grad_outputs=torch.ones_like(predicted_position),
        create_graph=True
    )[0]

    # Second derivative: d(Velocity) / d(Time) = Acceleration
    predicted_acceleration = torch.autograd.grad(
        outputs=predicted_velocity,
        inputs=t_physics,
        grad_outputs=torch.ones_like(predicted_velocity),
        create_graph=True
    )[0]

    #Physics Equation Rule: Acceleration + Position = 0
    physics_residual = predicted_acceleration + predicted_position
    loss_physics = torch.mean(physics_residual**2)

    #Total Loss: Boundary Loss + Physics Loss
    total_loss = loss_boundary + loss_physics
    total_loss.backward()
    optimizer.step()

    #Print progress every 400 rounds
    if epoch % 400 == 0:
        print(f"Epoch {epoch:4d} | Total Loss: {total_loss.item():.4f} | Physics Loss: {loss_physics.item():.4f}")

print("Training finished successfully!")

#Plot the results

t_test = torch.linspace(0, 10, 500).view(-1, 1)
with torch.no_grad():
    model_predictions = model(t_test).numpy()

#Solution
true_solution = np.cos(t_test.numpy())


#Plot the results

plt.figure(figsize=(10, 5))
plt.plot(t_test.numpy(), true_solution, label="Exact Math Solution (cos(t))", color="black", linestyle="dashed")
plt.plot(t_test.numpy(), model_predictions, label="64-Neuron PINN Prediction", color="red", alpha=0.8)
plt.title("Your PINN Solving Pendulum Motion from Pure Physics")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid(True)

plt.savefig("outputs/pendulum_prediction.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

#Seaborn plot

input_weights = model.input_layer.weight.detach().numpy() #Weight matrix for the input layer

#Plotting 

sns.set_theme(style='white')
plt.figure(figsize=(15, 4))

sns.heatmap(
    input_weights,
    cmap='vlag',
    center=0,
    linewidths=0.5,
    cbar_kws={'orientation': 'horizontal', 'label': 'Connection Weight Intensity'}
)

plt.title('High-Fidelity Neural Weight Map', fontsize=16, fontweight='bold', pad=15)   
plt.xlabel('Input Neuron Index', fontsize=14)
plt.ylabel('Input Feature (Time)', fontsize=14)

plt.tight_layout()
plt.show()
plt.savefig('outputs/neural_weight_map.png', dpi=300, bbox_inches='tight')

print("All plots saved successfully!")