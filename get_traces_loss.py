import os
import csv
from DSL_data_classes import DSL_Data_Set, DSL_Trace_Data_Set
from imp_NN_training import load_NN_from_weights, give_NN_Truck_Trailer_Multi_Stage_policy, split_datasets, give_split_NN_policy
from DSL_functions import Simulate_system_traces, Validate_trace_datasets
from imp_truck_trailer_multi_stage_loop_DSL import Truck_Trailer_Multi_Stage_Loop_Use_case, standalone_simulate_function, General_Simulation_parameters
from imp_validate_trace_datasets import validate_trace_datasets_function
from simulator import simulator_omega_init
from imp_validate_trace_datasets import Validation_trace_parameters

for i in range(1,21):
	os.makedirs("model_run_tansig_iteration_"+str(i))
	hyperparameterfile_1 = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_1\output_NN_hypertuning\hyperparameterfile"
	modelweights_1 = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_1\output_NN_training\dnn_modelweigths.h5"
	hyperparameterfile_2 = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_2\output_NN_hypertuning\hyperparameterfile"
	modelweights_2 = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_2\output_NN_training\dnn_modelweigths.h5"
	hyperparameterfile_3 = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_3\output_NN_hypertuning\hyperparameterfile"
	modelweights_3 = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\model_3\output_NN_training\dnn_modelweigths.h5"
	datafile_csv = "DSL_truck_trailer_multi_stage_split_model_run_tansig\iteration_"+str(i)+"\dataset.csv"
	expert_trace_dataset_csv = "truck_trailer_multi_stage_loop_traces_index_v1_traces.csv"
	start_point_dataset_csv = "truck_trailer_multi_stage_loop_traces_index_v1_start_points.csv"
	simu = simulator_omega_init()

	general_simulation_parameters = General_Simulation_parameters(function=simu)
	simulate_system_traces = Simulate_system_traces(standalone_simulate_function, general_simulation_parameters)
	validate_trace_datasets = Validate_trace_datasets(validate_trace_datasets_function, False)

	start_point_dataset = DSL_Data_Set()
	start_point_dataset.initialize_from_csv(start_point_dataset_csv)

	expert_trace_dataset = DSL_Trace_Data_Set()
	expert_trace_dataset.initialize_from_csv(expert_trace_dataset_csv)

	use_case = Truck_Trailer_Multi_Stage_Loop_Use_case()
	use_case.set_self_parameters()

	datafile = DSL_Data_Set()
	datafile.initialize_from_csv(datafile_csv)

	dataset_1, dataset_2, dataset_3 = split_datasets(datafile)
	dataset_list = [dataset_1, dataset_2, dataset_3]
	hyperparameterfile_list = [hyperparameterfile_1, hyperparameterfile_2, hyperparameterfile_3]
	modelweights_list = [modelweights_1, modelweights_2, modelweights_3]

	NN_list = []
	for j in range(0,3):
		with open(hyperparameterfile_list[j]) as file_name:
			csvreader = csv.reader(file_name)
			hyperparameters = []
			for row in csvreader:
				hyperparameters.append(row[0])

		total_dataset = dataset_list[j]

		NN = load_NN_from_weights(use_case, hyperparameters, total_dataset.input_dataframe, modelweights_list[j])
		NN_list.append(NN)

	
	policy = give_split_NN_policy(use_case, NN_list[0], NN_list[1], NN_list[2])
	trace_dataset = simulate_system_traces.simulate_system_traces(policy, start_point_dataset.input, 49)

	validation_trace_parameters = Validation_trace_parameters(output_map="model_run_tansig_iteration_"+str(i))
	validate_trace_datasets.validate_trace_datasets(expert_trace_dataset, trace_dataset, validation_trace_parameters)