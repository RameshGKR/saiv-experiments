import os
import pandas as pd

# Base directory containing iteration folders

base_dir = "D:\\Archive\\multi stage relu and tansig\\multi stage relu and tansig\\DSL_truck_trailer_multi_stage_model_run_tansig"


# Iteration folders
iterations = [f"iteration_{i}" for i in range(1, 21)]

# Data storage
test_loss_data = []

# Extract test loss from each iteration
for iteration in iterations:
    file_path = os.path.join(base_dir, iteration, "test_loss.txt")
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            line = file.readline().strip()
            loss_value = float(line.split(":")[-1].strip())  # Extract numerical loss
            test_loss_data.append({"Iteration": iteration, "Test Loss": loss_value})

# Convert to DataFrame
df = pd.DataFrame(test_loss_data)

# Save to Excel file
output_path = "test_loss.xlsx"
df.to_excel(output_path, index=False)

# Display the file to the user
import ace_tools as tools
tools.display_dataframe_to_user(name="Test Loss Summary", dataframe=df)

# Output the Excel file path for reference
output_path
