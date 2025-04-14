from DSL_data_classes import DSL_Trace_Data_Set, DSL_Data_Set

        
trace_dataset = DSL_Trace_Data_Set()
trace_dataset.initialize_from_csv("test.csv")
trace_dataset.write_dataset_to_csv("test2.csv")