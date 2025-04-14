import matplotlib.pyplot as plt
import csv

# truck_trailer_multi_stage_loop_traces_0p1_trial3.csv is with 0.1, 1x1grid, 250steps; N1=100, N2=50 and N3=100

# truck_trailer_multi_stage_loop_traces_0p1_trial4.csv is with 0.1, 1x1grid, 300steps; N1=120, N2=60 and N3=120

filename = "dataset.csv"
#filename = "truck_trailer_multi_stage_loop_traces_0p1_trial3.csv"
data = []

with open(filename, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            data.append([float(row[0]), float(row[1])])  # x1, y1 only

steps_per_traj = 300
n_traj = len(data) // steps_per_traj

x1_goal = -1.1875
y1_goal = 0.74318967

plt.figure(figsize=(12, 6))
for i in range(n_traj):
    traj = data[i*steps_per_traj : (i+1)*steps_per_traj]
    x1 = [row[0] for row in traj]
    y1 = [row[1] for row in traj]
    plt.plot(x1, y1, label=f"Trajectory {i+1}" if i < 5 else "", alpha=0.5)
    plt.plot(x1[-1], y1[-1], 'kx')
    
plt.plot(x1_goal, y1_goal, 'ro', label='Target')
plt.title("Expert Trajectories (Sample Time = 0.1s)")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()
