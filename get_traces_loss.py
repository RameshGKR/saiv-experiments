import os
import csv
from DSL_data_classes import DSL_Data_Set, DSL_Trace_Data_Set
from imp_NN_training import load_NN_from_weights, give_NN_Truck_Trailer_Multi_Stage_policy, split_datasets, give_split_NN_policy
from DSL_functions import Simulate_system_traces, Validate_trace_datasets
from imp_truck_trailer_multi_stage_loop_DSL import Truck_Trailer_Multi_Stage_Loop_Use_case, standalone_simulate_function, General_Simulation_parameters
from imp_validate_trace_datasets import validate_trace_datasets_function
from simulator import simulator_omega_init
from imp_validate_trace_datasets import Validation_trace_parameters

#for i in range(1,21):
for i in range(1,5):
	#os.makedirs("model_run_tansig_iteration_"+str(i))

	os.makedirs("0p1_relu_pruned_"+str(i))
	hyperparameterfile = "DSL_truck_trailer_multi_stage_model_loop_0p1_prune01_slow0_run_relu_5\iteration_"+str(i)+"\output_NN_hypertuning\hyperparameterfile"
	modelweights = "DSL_truck_trailer_multi_stage_model_loop_0p1_prune01_slow0_run_relu_5\iteration_"+str(i)+"\output_NN_training\dnn_modelweigths.h5"
	datafile_csv = "DSL_truck_trailer_multi_stage_model_loop_0p1_prune01_slow0_run_relu_5\iteration_"+str(i)+"\dataset.csv"
	expert_trace_dataset_csv = "truck_trailer_multi_stage_loop_traces_0p1_index_traces.csv"
	start_point_dataset_csv = "truck_trailer_multi_stage_loop_traces_0p1_index_start_points.csv"
	simu = simulator_omega_init()
    
	#os.makedirs("0p1_relu_pruned_with_hybberish_"+str(i))
	#hyperparameterfile = "Dagger_with_hybberish_relu_1\iteration_"+str(i)+"\output_NN_hypertuning\hyperparameterfile"
	#modelweights = "Dagger_with_hybberish_relu_1\iteration_"+str(i)+"\output_NN_training\dnn_modelweigths.h5"
	#datafile_csv = "Dagger_with_hybberish_relu_1\iteration_"+str(i)+"\dagger_with_hybberish_dataset.csv"
	#expert_trace_dataset_csv = "truck_trailer_multi_stage_loop_traces_0p1_index_traces.csv"
	#start_point_dataset_csv = "truck_trailer_multi_stage_loop_traces_0p1_index_start_points.csv"
	#simu = simulator_omega_init()


	general_simulation_parameters = General_Simulation_parameters(function=simu)
	simulate_system_traces = Simulate_system_traces(standalone_simulate_function, general_simulation_parameters)
	validate_trace_datasets = Validate_trace_datasets(validate_trace_datasets_function, False)

	start_point_dataset = DSL_Data_Set()
	start_point_dataset.initialize_from_csv(start_point_dataset_csv)

	expert_trace_dataset = DSL_Trace_Data_Set()
	expert_trace_dataset.initialize_from_csv(expert_trace_dataset_csv)

	use_case = Truck_Trailer_Multi_Stage_Loop_Use_case()
	use_case.set_self_parameters()

	dataset = DSL_Data_Set()
	dataset.initialize_from_csv(datafile_csv)

	NN_list = []

	with open(hyperparameterfile) as file_name:
		csvreader = csv.reader(file_name)
		hyperparameters = []
		for row in csvreader:
			hyperparameters.append(row[0])

	NN = load_NN_from_weights(use_case, hyperparameters, dataset.input_dataframe, modelweights)
	NN_list.append(NN)

	
	policy = give_NN_Truck_Trailer_Multi_Stage_policy(use_case, NN)
	trace_dataset = simulate_system_traces.simulate_system_traces(policy, start_point_dataset.input, 299)

	validation_trace_parameters = Validation_trace_parameters(output_map="0p1_relu_pruned_with_hybberish_"+str(i))
	validate_trace_datasets.validate_trace_datasets(expert_trace_dataset, trace_dataset, validation_trace_parameters)