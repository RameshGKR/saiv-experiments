import pandas as pd
from casadi import vertcat
from concurrent.futures import ProcessPoolExecutor
import time
import os

from truck_trailer_multistage_loop_expert_function import get_expert_action_truck_trailer

def call_mpc_for_row(index_row_tuple):
    i, row = index_row_tuple
    start_time = time.time()
    print(f"[row {i}] computing expert action..")

    input_vector = vertcat(row['input_theta1'], row['input_x1'], row['input_y1'], row['input_theta0'])
    index = int(row['input_index'])

    try:
        output = get_expert_action_truck_trailer(input_vector, index)
        elapsed = time.time() - start_time
        print(f"[row {i}] Done in {elapsed:.2f} seconds")
        return float(output[0]), float(output[1]), elapsed
    except Exception as e:
        print(f"[row {i}] failed: {e}")
        return None, None, None

def process_chunk(chunk_df, chunk_id, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(call_mpc_for_row, [(i, row) for i, row in chunk_df.iterrows()]))

    v0_list, delta0_list, time_list = zip(*results)
    chunk_df['output_v0'] = v0_list
    chunk_df['output_delta0'] = delta0_list
    chunk_df['exec_time_sec'] = time_list

    chunk_filename = f"call_expert_part_{chunk_id}.csv"
    chunk_df.to_csv(chunk_filename, index=False)
    print(f"Saved chunk {chunk_id} to {chunk_filename}")

if __name__ == "__main__":
    df = pd.read_csv("call_expert.csv")
    chunk_size = 100
    total_chunks = (len(df) + chunk_size - 1) // chunk_size

    for chunk_id in range(total_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = min((chunk_id + 1) * chunk_size, len(df))
        chunk_df = df.iloc[start_idx:end_idx].copy()
        print(f"\n Processing chunk {chunk_id} ({start_idx} to {end_idx - 1})")
        process_chunk(chunk_df, chunk_id, num_workers=4)

    print("\n All chunks processed. Ready to merge the parts into a final CSV.")
