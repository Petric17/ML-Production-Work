import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.network import DampedPendulumModel

model = DampedPendulumModel()

#Time input for the physics and initial conditions
t_physics = torch.linspace(0, 10, 500).view(-1, 1).requires_grad_(True)
t_initial = torch.tensor([[0.0]], requires_grad=True)
true_initial_angle = torch.tensor([[1.0]])

#Damping coefficient
c=0.5

#L-BFGS Optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=0.01,
    line_search_fn="strong_wolfe"
)

print("Starting training...")

#Training loop

for epoch in range(61): 

    def closure():
        optimizer.zero_grad()

        #LOSS 1: Boundary Condition (Start at angle 1.0)
        predicted_initial_angle = model(t_initial)
        loss_boundary = torch.mean((predicted_initial_angle - true_initial_angle)**2)

        #LOSS 2: Physics Residual (Damped Pendulum Equation)
        predicted_position = model(t_physics)

        #First derivative: dtheta/dt = Velocity

        dtheta_dt = torch.autograd.grad(predicted_position, t_physics, grad_outputs=torch.ones_like(predicted_position), create_graph=True)[0]

        #Second derivative: d^2theta/dt^2 = Acceleration

        d2theta_dt2 = torch.autograd.grad(dtheta_dt, t_physics, grad_outputs=torch.ones_like(dtheta_dt), create_graph=True)[0]

        #NEW PHYSICS LAW: Acceleration + (Friction * Velocity) + Position must equal 0
        physics_residual =d2theta_dt2 + (c * dtheta_dt) + predicted_position
        loss_physics = torch.mean(physics_residual**2)

        #Total Loss: Boundary Loss + Physics Loss

        total_loss = loss_boundary + loss_physics
        total_loss.backward()
        return total_loss
    
    optimizer.step(closure)

    current_loss = closure().item()
    print(f"Iteration {epoch:2d} | Current Total Loss: {current_loss:.6f}")

    os.makedirs("outputs", exist_ok=True)

    t_test = torch.linspace(0, 10, 200).view(-1, 1)
    with torch.no_grad():
        model_predictions = model(t_test).numpy()

    true_solution = np.cos(t_test.numpy())

t_np = t_test.numpy()
true_damped_solution = np.exp(-0.5 * c * t_np) * np.cos(t_np * np.sqrt(1 - (c/2)**2))

#Plot the results
plt.figure(figsize=(10, 5))
plt.plot(t_np, true_damped_solution, label="Exact Damped Math Solution", color="black", linestyle="dashed")
plt.plot(t_np, model_predictions, label="PINN Friction Prediction", color="dodgerblue", linewidth=2, alpha=0.8)
plt.title(f"PINN Solving Damped Pendulum Motion (Friction c = {c})")
plt.xlabel("Time (seconds)")
plt.ylabel("Angle (radians)")
plt.legend()
plt.grid(True)

#Save the plot
plt.savefig("outputs/damped_pendulum_prediction.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()