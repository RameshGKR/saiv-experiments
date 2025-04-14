import os
import math
import numpy as np
import pandas as pd
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy

from imp_NN_training import Train_NN_Policy_parameters, Retrain_NN_Policy_parameters
from imp_validate_datasets import Validation_parameters
from imp_validate_trace_datasets import Validation_trace_parameters
def Simple_Dagger(train_NN, simulate_system_traces, expert_policy, init_policy, Dagger_loops, p, start_point_dataset, trace_length):
    total_dataset = DSL_Data_Set()
    policies_list = [init_policy]

    for idx in range(Dagger_loops):
        beta = math.pow(p, idx)
        loop_policy = beta*expert_policy + (1-beta)*policies_list[idx]

        trace_dataset = simulate_system_traces.simulate_system_traces(loop_policy, start_point_dataset, trace_length)

        loop_dataset = DSL_Data_Set()
        loop_dataset.append_trace_dataset(trace_dataset)
        loop_dataset.output = expert_policy.give_output(loop_dataset.input)

        total_dataset.append_dataset(loop_dataset)

        trained_policy = train_NN.train_policy(total_dataset)
        policies_list.append(trained_policy)
    
    return policies_list

def Simple_NDI(train_NN, simulate_system, expert_policy, start_point_dataset, number_of_paths, max_iteration, trace_length, max_diff):
    
    policy_list =[expert_policy]
    total_traces = DSL_Trace_Data_Set()

    for i in range(max_iteration):
        total_traces = get_start_points(start_point_dataset, number_of_paths, expert_policy, total_traces)

        loop_traces = DSL_Trace_Data_Set()
        training_dataset = DSL_Data_Set()

        for trace in total_traces:
            loop_trace = Simple_ExteND(expert_policy, policy_list[i], trace, trace_length, max_diff, simulate_system)
            training_dataset.append_dataset(loop_trace)
            
            if loop_trace.length>trace.length:
                loop_traces.append_dataset(loop_trace)
            else:
                loop_traces.append_dataset(trace)

        policy = train_NN.train_policy(training_dataset)
        policy_list.append(policy)
        total_traces = loop_traces


def Simple_ExteND(expert_policy, policy, trace, extend_end_index, max_diff, simulate_system):
    extend_trace = DSL_Data_Set(is_trace = True)

    for n in range(extend_end_index):
        if n+1>trace.length:
            loop_data_point = extend_trace.get_iter_datapoint_from_dataset(n-1)
            trace_datapoint = simulate_system.simulate_system(loop_data_point)

            output_trace_datapoint = expert_policy.give_output(trace_datapoint)
            trace_datapoint.set_datapoint_output_with_dataframe(output_trace_datapoint.output_dataframe)
        else:
            trace_datapoint = trace.get_iter_datapoint_from_dataset(n)
        
        extend_trace.append_datapoint(trace_datapoint)

        if Simple_check_output(trace_datapoint, policy, max_diff):
            break
    
    return extend_trace

def Simple_check_output(trace_datapoint, policy, max_diff):
    policy_input_datapoint = DSL_Data_Point()
    policy_input_datapoint.set_datapoint_input_with_dataframe(trace_datapoint.input_dataframe)

    policy_output_datapoint = policy.give_output(policy_input_datapoint)
    
    totalbool = True

    for output_label in trace_datapoint.output_dataframe.columns:
        bool=trace_datapoint.output_dataframe[output_label][0]-max_diff<=policy_output_datapoint.output_dataframe[output_label][0]<=trace_datapoint.output_dataframe[output_label][0]+max_diff
        totalbool=bool and totalbool

    return totalbool

def get_start_points(start_point_dataset, number_of_paths, expert_policy, total_traces):
    start_point_dataset.shuffle()

    for idx in range(number_of_paths):
        start_point = start_point_dataset.get_iter_datapoint_from_dataset(idx)

        k_start_points_dataset = DSL_Data_Set(is_trace=True)
        k_start_points_dataset.append_datapoint(start_point)
        k_start_points_dataset.output = expert_policy.give_output(k_start_points_dataset.input)

        total_traces.append_dataset(k_start_points_dataset)
  
    return total_traces


def Simple_CL(train_NN, retrain_NN, total_dataset, k):
    dataset_list = sort_train_dataset(total_dataset, k)

    training_dataset = DSL_Data_Set()

    for idx, dataset in enumerate(dataset_list):
    
        training_dataset.append_dataset(dataset)

        if idx == 0:
            policy = train_NN.train_policy(training_dataset)
        else:
            policy = retrain_NN.train_policy(training_dataset)  

def sort_train_dataset(total_dataset, amount_of_datasets):
    dataset_list = []
    for _ in range(amount_of_datasets):
        dataset = DSL_Data_Set()
        dataset_list.append(dataset)

    sort_pandaframe = pd.DataFrame(columns=['abs_pos','datapoint'])

    for idx, datapoint in enumerate(total_dataset):
        pos = datapoint.input_dataframe['POS'][0]
        abs_pos = np.abs(pos)

        sort_pandaframe.loc[idx] = [abs_pos, datapoint]

    sort_pandaframe = sort_pandaframe.sort_values(by=['abs_pos'])
    length = sort_pandaframe.shape[0]
    amount = int(np.ceil(length/amount_of_datasets))
    begin_index = 0

    for i in range(amount_of_datasets-1):
        end_index = begin_index + amount
        loop_dataframe = sort_pandaframe.iloc[begin_index:end_index]

        for datapoint in loop_dataframe['datapoint']:
            dataset_list[i].append_datapoint(datapoint)

        begin_index = end_index

    loop_dataframe = sort_pandaframe.iloc[begin_index:]

    for datapoint in loop_dataframe['datapoint']:
        dataset_list[i+1].append_datapoint(datapoint)

    return dataset_list