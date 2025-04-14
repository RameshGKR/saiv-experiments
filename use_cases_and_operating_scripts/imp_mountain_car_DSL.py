import sys
import os
import csv
from telnetlib import DM
import pandas as pd
from rockit import *
from pylab import *
from casadi import vertcat
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from keras import layers
from casadi import Function
from rockit import casadi_helpers
import keras_tuner as kt
from typing import List
from dataclasses import dataclass
from Use_Case_class import Use_Case
from Use_Case_class import Data_set
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy
from DSL_functions import Train_policy, Simulate_system

class Mountain_Car_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['POS','V']
        self.NN_labels_input = ['POS','V']
        self.labels_output = ['F']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [100]*len(self.perfect_paths)
        self.timesteps = [1]*len(self.perfect_paths)
        self.timestep = 1
        #self.custom_objects = {'restricted_output_mse': restricted_output_mse}

        [solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0)
        self.MPC_function = solve
        self.simulation_function = Sim_cart_pole_dyn
        self.expert_policy = self.expert_output_function
        self.simulate_system = Simulate_system(self.simulate_function)
        
    def give_hypermodel(self, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        hyper_model = Mountain_Car_HyperModel(normalizer)
        return hyper_model

    def give_NN_model(self, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        model = build_and_compile_model(normalizer, hyper_parameters)
        return model
    
    def expert_output_function(self, input_datapoint: DSL_Data_Point) -> DSL_Data_Point:
        MPC_input = vertcat(input_datapoint.input_dataframe['POS'].iloc[0], input_datapoint.input_dataframe['V'].iloc[0])
        try:
            F = self.MPC_function(MPC_input)
            F = casadi_helpers.DM2numpy(F, [2,1])
            output_datapoint = DSL_Data_Point(output = {'F' : F})
        except:
            output_datapoint = DSL_Data_Point()

        return output_datapoint

    def simulate_function(self, datapoint: DSL_Data_Point, Simulation_parameters) -> DSL_Data_Point:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(datapoint.input_dataframe['POS'].iloc[0], datapoint.input_dataframe['V'].iloc[0])
        current_U = vertcat(datapoint.output_dataframe['F'].iloc[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=self.time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        output_datapoint = DSL_Data_Point(input = {'POS': system_result[0], 'V': system_result[1]})

        return output_datapoint
    def give_next_state(self, datapoint: List[float], control_variables: List[float], time_step: float, idx: int) -> List[float]:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(input[0], input[1])
        current_U = vertcat(control_variables[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        return [system_result[0], system_result[1]]

    def sample_k_start_points(self, k: int):
        """This function gives k start points"""
        start_points = []

        for _ in range(k):
            start_pos = uniform(-0.6, -0.4)

            start_point = [start_pos, 0]
            start_points.append(start_point)
        
        end_iterations = [100]*k
        
        return start_points, end_iterations

class Mountain_Car_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization):
        self.normalizer = norm

    def build(self, hp: kt.engine.hyperparameters.HyperParameter) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['sigmoid', 'tanh'])
        hidden_layers = hp.Int('hidden_layers', min_value=4, max_value=4, step=1)

        model = keras.Sequential()
        model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(1))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(hp_learning_rate))
        return model

def build_and_compile_model(normalizer, hyper_parameters: List) -> tf.keras.Sequential:
    """This function gives the NN model"""
    # normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
    # normalizer.adapt(np.array(train_features))
        
    model = keras.Sequential()
    model.add(normalizer)
    for _ in range(int(hyper_parameters[3])):
        model.add(layers.Dense(hyper_parameters[0], activation = hyper_parameters[1]))
    model.add(layers.Dense(1))

    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]))
    return model

@dataclass
class General_Simulation_parameters:
    function:str =False

def standalone_simulate_function(datapoint: DSL_Data_Point, General_Simulation_parameters, Simulation_parameters) -> DSL_Data_Point:
    """This function gives the next state for a state and control variables"""
    current_X = vertcat(datapoint.input_dataframe['POS'].iloc[0], datapoint.input_dataframe['V'].iloc[0])
    current_U = vertcat(datapoint.output_dataframe['F'].iloc[0])

    system_result = General_Simulation_parameters.function(x0=current_X, u=current_U, T=1)["xf"]
    system_result=casadi_helpers.DM2numpy(system_result, [2,1])

    output_datapoint = DSL_Data_Point(input = {'POS': system_result[0], 'V': system_result[1]})

    return output_datapoint

def initialize_ocp(pos_init: float, v_init: float) -> List[Function]:
    """Function that initialize the ocp for the cart pole"""
    Nsim  = 100 # number of simulation steps
    nx    = 2 # the system is composed of 2 states
    nu    = 1 # the system has 1 input
    power = 0.0015
    p_min, p_max = -1.2, 0.6
    v_min, v_max = -0.07, 0.07
    goal_position = 0.45
    N = 100  # number of control intervals

    # Set OCP
    ocp = Ocp(T=100.0)

    # Define states
    p = ocp.state() # position
    v = ocp.state() # velocity

    # Define controls
    F = ocp.control(order=0)

    X_0 = ocp.parameter(2);

    # Specify ODE
    ocp.set_der(p, v)
    ocp.set_der(v, F * power - 0.0025 * cos(3 * p))

    # Objective
    ocp.add_objective(ocp.integral(0.1 * F**2 + (p - goal_position)**2))

    # Path constraints
    ocp.subject_to(-1 <= (F<= 1))
    ocp.subject_to(p_min <= (p <= p_max))
    ocp.subject_to(v_min <= (v <= v_max))

    X = vertcat(p, v)
    ocp.subject_to(ocp.at_t0(X)==X_0)
    ocp.subject_to(ocp.at_tf(X)>=vertcat(goal_position, 0))

    # Pick a solver
    options = {"ipopt": {"print_level": 0}}
    #options["expand"] = True
    options["print_time"] = False
    ocp.solver('ipopt', options)

    # Make it concrete for this ocp
    ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
    ocp.set_value(X_0, vertcat(pos_init, v_init))

    # Get discretised dynamics as CasADi function to simulate the system
    solve_ocp = ocp.to_function('solve_ocp',
                                [X_0],
                                [ocp.sample(F,grid='control')[1][0]])

    Sim_system_dyn = ocp._method.discrete_system(ocp)

    return [solve_ocp, Sim_system_dyn]

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""

    with open("mountain_car_position_perfect_paths_100.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_pos = []
        for row in csvreader:
            MPC_pos.append([float(i) for i in row])
        
    with open("mountain_car_speed_perfect_paths_100.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_theta = []
        for row in csvreader:
            MPC_theta.append([float(i) for i in row])
    
    perfect_paths = []
    
    for idx in range(len(MPC_pos)):
        perfect_path = Data_set(2, 0)
        perfect_path.load_data_from_col_list([MPC_pos[idx], MPC_theta[idx]])
        perfect_paths.append(perfect_path)
   
    return perfect_paths

#-------------------------------------------#
# Functions to disable and restore printing #
#-------------------------------------------#

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

if __name__ == "__main__":
    # input=tf.constant([-100.0, -50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0, 100.0])
    # print(restricted_output(input))
    dataset = DSL_Data_Set()