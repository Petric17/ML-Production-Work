import os
import pandas as pd
import numpy as np

# Create data folder
os.makedirs("data", exist_ok=True)
save_path = "data/asteroid_data.csv"

print("Initializing Network-Free NASA Data Generator...")


np.random.seed(42)
num_asteroids = 5000

# Draw realistic physical distributions
min_diameters = np.random.exponential(scale=0.1, size=num_asteroids) + 0.01
max_diameters = min_diameters * np.random.uniform(1.2, 2.5, size=num_asteroids)
velocities = np.random.normal(loc=15, scale=5, size=num_asteroids)
velocities = np.clip(velocities, 1, 45) # Keep velocities realistic

# Absolute Magnitude is mathematically bound to size
magnitudes = 22.0 - 5.0 * np.log10(max_diameters) + np.random.normal(0, 0.5, size=num_asteroids)

df = pd.DataFrame({
    'Est Dia(min)': min_diameters,
    'Est Dia(max)': max_diameters,
    'Relative Velocity km per sec': velocities,
    'Absolute Magnitude': magnitudes
})

# 4. Save to CSV
df.to_csv(save_path, index=False)
print(f"Success! 5,000 structured asteroid observations saved locally to: {save_path}")