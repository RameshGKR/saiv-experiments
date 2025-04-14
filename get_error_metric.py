import pandas as pd
import numpy as np

# === SETTINGS ===
csv_path = "dataset.csv"  # your interim file
steps_per_traj = 300  # N_1 + N_2 + N_3
x_goal = -1.1875
y_goal = 0.7432

# === LOAD CSV ===
data = pd.read_csv(csv_path, header=None)
n_traj = len(data) // steps_per_traj

errors = []

for i in range(n_traj):
    traj_data = data.iloc[i * steps_per_traj: (i + 1) * steps_per_traj]
    x1_final = traj_data.iloc[-1, 0]
    y1_final = traj_data.iloc[-1, 1]
    err = np.sqrt((x1_final - x_goal)**2 + (y1_final - y_goal)**2)
    errors.append(err)
    print(f"Trajectory {i+1:>3} | Final error: {err:.4f} m")

print("\n--- Summary ---")
print(f"Total trajectories: {n_traj}")
print(f"Mean error       : {np.mean(errors):.4f} m")
print(f"Max error        : {np.max(errors):.4f} m")
print(f"Min error        : {np.min(errors):.4f} m")
