import pandas as pd
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set

dataframe_1 = pd.DataFrame({'a': [1,2,3,4,5], 'b': [2,4,6,8,10],'c': [3,6,9,12,15]})
dataframe_2 = pd.DataFrame({'e': [-1,-2,-3,-4,-5], 'd': [-2,-4,-6,-8,-10]})

dataset = DSL_Data_Set(is_trace=True)
dataset.initialize_with_dataframe(input_dataframe=dataframe_1, output_dataframe=dataframe_2)

trace_dataset = DSL_Trace_Data_Set()
for i in range(5):
    trace_dataset.append_dataset(dataset)

trace_dataset.write_dataset_to_csv("test.csv")

