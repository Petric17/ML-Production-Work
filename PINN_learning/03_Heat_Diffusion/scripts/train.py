import os
import sys
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

scripts_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(scripts_dir)
sys.path.insert(0, project_root)

from src.network import HeatDiffusionModel

model = HeatDiffusionModel()

#Space and time input

num_space = 40
num_time = 40 

x = torch.linspace(0, 1, num_space)
t = torch.linspace(0, 1, num_time)

#2D grid coordinates matrix

grid_x, grid_t = torch.meshgrid(x,t, indexing="ij")
x_physics = grid_x.flatten().view(-1, 1).requires_grad_(True)
t_physics = grid_t.flatten().view(-1, 1).requires_grad_(True)
X_physics = torch.cat([x_physics, t_physics], dim=1)

#Boundary Conditions Data

t_boundary = torch.linspace(0, 1, 100).view(-1, 1)
x_left = torch.zeros_like(t_boundary)
x_right = torch.ones_like(t_boundary)

X_left = torch.cat([x_left, t_boundary], dim=1)
X_right = torch.cat([x_right, t_boundary], dim=1)

#Initial Conditions Data

x_initial = torch.linspace(0, 1, 100).view(-1, 1)
t_initial = torch.zeros_like(x_initial)
X_initial = torch.cat([x_initial, t_initial], dim=1)
u_initial = torch.sin(np.pi * x_initial)

#Thermal Diffusivity

alpha_thermal = 0.1

#High-Curvature L-BFGS Optimizer
optimizer = torch.optim.LBFGS(
    model.parameters(),
    lr=0.01,
    line_search_fn="strong_wolfe"
)
print("Training 2D Spatial PINN to solve Heat Diffusion Equation...")

#Training loop

for epoch in range(51):

    def closure():
        optimizer.zero_grad()

        #Initial Conditions Loss
        pred_init = model(X_initial)
        loss_init = torch.mean((pred_init - u_initial)**2)

        #Boundary Conditions Loss
        pred_left = model(X_left)
        pred_right = model(X_right)
        loss_boundary = torch.mean(pred_left ** 2) + torch.mean(pred_right ** 2)
        pred_right = model(X_right)
        loss_boundary = torch.mean(pred_left ** 2) + torch.mean(pred_right ** 2)


        #Partial Differential Equation (PDE) Residual
        pred_u = model(X_physics)

        #First Derivative with respect to both inputs [du/dx, du/dt]

        gradients = torch.autograd.grad(
            pred_u, X_physics,
            grad_outputs=torch.ones_like(pred_u),
            create_graph=True
        )[0]

        du_dx = gradients[:, 0:1]
        du_dt = gradients[:, 1:2]

        #Second Derivative with respect to space

        d2u_dx2 = torch.autograd.grad(
            du_dx, X_physics,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True
        )[0] [:, 0:1]

        #Heat Equation Law
        pde_residual = du_dt - (alpha_thermal * d2u_dx2)
        loss_pde = torch.mean(pde_residual ** 2)

        #Total Loss: Initial + Boundary + PDE
        total_loss = (loss_init * 2.0) + (loss_boundary * 2.0) + loss_pde
        total_loss.backward()
        return total_loss

    optimizer.step(closure)
    current_loss = closure().item()
    print(f"Iteration {epoch:2d} | Total Composite Loss: {current_loss:.6f}")

print("Training finished! Generating 2D Heatmap Animation frames...")

#Plot the results
os.makedirs("outputs", exist_ok=True)

plot_x = np.linspace(0, 1, 100)
plot_t = np.array([0.0, 0.1, 0.3, 0.6, 1.0])
plt.figure(figsize=(10, 6))

for t_val in plot_t:
    X_test = torch.tensor([[x_val, t_val] for x_val in plot_x], dtype=torch.float32)
    with torch.no_grad():
        pred_temp = model(X_test).numpy()
        
    exact_temp = np.exp(-alpha_thermal * (np.pi**2) * t_val) * np.sin(np.pi * plot_x)
    
    p = plt.plot(plot_x, pred_temp, label=f"PINN t={t_val}s", linewidth=2)
    plt.plot(plot_x, exact_temp, linestyle="dashed", color=p[0].get_color(), alpha=0.5)

plt.title("PINN Simulation: Heat Dissipation Down a Metal Rod")
plt.xlabel("Rod Position (meters)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)

plt.savefig("outputs/heat_diffusion_prediction.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()




    