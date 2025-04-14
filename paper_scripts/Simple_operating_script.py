import math
import sys

from Simple_DSL_NN_training import Train_NN_Policy, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, Retrain_NN_Policy, General_Retrain_NN_Policy_parameters, Retrain_NN_Policy_parameters
from imp_validate_datasets import validate_datasets_function, Validation_parameters
from imp_validate_trace_datasets import validate_trace_datasets_function, Validation_trace_parameters
from Simple_DSL_general_methods import Simple_Dagger, Simple_NDI, Simple_CL
from Use_Case_class import Use_Case
from imp_cart_pole_use_case import Cart_pole_Use_case, standalone_simulate_function, initialize_ocp, General_Simulation_parameters
from paper_scripts.random_tree import Train_Random_Tree_Policy, General_Train_Random_Tree_Policy_Parameters

from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy
from DSL_functions import Train_policy, Simulate_system, Simulate_system_traces, Validate_datasets, Validate_trace_datasets

def run_DSL_operation():
    use_case = Cart_pole_Use_case()
    use_case.set_self_parameters()

    output_map = 'delete_later2'
    [solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)

    general_train_nn_policy_parameters = General_Train_NN_Policy_parameters(do_hypertuning=True, hypertuning_epochs=2, hypertuning_factor=3,
                                                            project_name='hyper_trials', NN_training_epochs=4, hyperparameter_file="hyperparameterfile", use_case=use_case)
    
    general_train_random_tree_policy_parameters = General_Train_Random_Tree_Policy_Parameters(n_estimators=100)
    
    general_retrain_nn_policy_parameters = General_Retrain_NN_Policy_parameters(NN_training_epochs=4, use_case=use_case)
    
    general_simulation_parameters = General_Simulation_parameters(function=Sim_cart_pole_dyn)

    train_NN, retrain_NN, simulate_system, simulate_system_traces, validate_datasets, validate_trace_datasets, expert_policy = give_DSL_functions(train_function = Train_NN_Policy, general_train_policy_parameters = general_train_nn_policy_parameters,
    retrain_function = Retrain_NN_Policy, general_retrain_policy_parameters = general_retrain_nn_policy_parameters, simulation_function = standalone_simulate_function, general_simulation_parameters = general_simulation_parameters, validation_function = validate_datasets_function, general_validation_parameters = False,
    validation_trace_function = validate_trace_datasets_function, general_validation_trace_parameters = False, expert_policy_function = use_case.expert_output_function)
    
    total_dataset = DSL_Data_Set()
    total_dataset.initialize_from_csv('MPC_data\data_set_mpc_example_fixed_no_index.csv')
    init_policy = train_NN.train_policy(total_dataset)

    start_point_dataset = DSL_Data_Set()
    start_point_dataset.initialize_from_csv('MPC_data\data_set_mpc_example_fixed_start_no_index.csv')

    #Simple_Dagger(train_NN, simulate_system_traces, expert_policy, init_policy, 3, 0.5, start_point_dataset, 3)
    #Simple_NDI(train_NN, simulate_system, expert_policy, start_point_dataset, 20, 3, 3, 0.1)
    Simple_CL(train_NN, retrain_NN, total_dataset, 9)

def give_DSL_functions(train_function, general_train_policy_parameters, retrain_function, general_retrain_policy_parameters, simulation_function, general_simulation_parameters, validation_function, general_validation_parameters, validation_trace_function, general_validation_trace_parameters, expert_policy_function):
    train_NN = Train_policy(train_function, general_train_policy_parameters)
    retrain_NN = Train_policy(retrain_function, general_retrain_policy_parameters)

    simulate_system = Simulate_system(simulation_function, general_simulation_parameters)
    simulate_system_traces = Simulate_system_traces(simulation_function, general_simulation_parameters)

    validate_datasets = Validate_datasets(validation_function, general_validation_parameters)
    validate_trace_datasets = Validate_trace_datasets(validation_trace_function, general_validation_trace_parameters)
    expert_policy = Policy(expert_policy_function)
    
    return train_NN, retrain_NN, simulate_system, simulate_system_traces, validate_datasets, validate_trace_datasets, expert_policy

if __name__ == "__main__":
    run_DSL_operation()