from DSL_data_classes import DSL_Trace_Data_Set

if __name__ == "__main__":
    expert_trace_dataset = DSL_Trace_Data_Set()
    expert_trace_dataset.initialize_from_csv('MPC_data\data_set_mpc_example_perfect_paths')
    x=5