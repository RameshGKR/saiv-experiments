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
from simulator import simulator_delta_init

class Truck_Trailer_Use_case(Use_Case):
    def set_self_parameters(self):
        """This function will set the self parameters"""
        self.labels_input = ['theta1','x1','y1','theta0']
        self.NN_labels_input = ['theta1','x1','y1','theta0']
        self.labels_output = ['delta0','v0']
        self.hyperparameters = ['units', 'activation_function', 'learning_rate', 'hidden_layers']
        self.perfect_paths = get_perfect_paths()
        self.end_iterations = [100]*len(self.perfect_paths)
        self.timesteps = [0.7]*len(self.perfect_paths)
        self.timestep = 0.7
        #self.custom_objects = {'restricted_output_mse': restricted_output_mse}

        [solve, Sim_cart_pole_dyn] = initialize_ocp(0, 0, 0, 0)
        simu = simulator_delta_init()
        self.MPC_function = solve
        self.simulation_function = simu
        self.expert_policy = self.expert_output_function
        self.simulate_system = Simulate_system(self.simulate_function)
        
    def give_hypermodel(self, train_features: pd.DataFrame) -> kt.HyperModel:
        """This function gives the hyper model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        hyper_model = Truck_Trailer_HyperModel(normalizer, len(self.labels_input))
        return hyper_model

    def give_NN_model(self, hyper_parameters: List, train_features: pd.DataFrame) -> tf.keras.Sequential:
        """This function gives the NN model"""
        normalizer = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1)
        normalizer.adapt(np.array(train_features))

        model = build_and_compile_model(normalizer, hyper_parameters, len(self.labels_input))
        return model
    
    def expert_output_function(self, input_datapoint: DSL_Data_Point) -> DSL_Data_Point:
        MPC_input = vertcat(input_datapoint.input_dataframe['theta1'].iloc[0], input_datapoint.input_dataframe['x1'].iloc[0], input_datapoint.input_dataframe['y1'].iloc[0], input_datapoint.input_dataframe['theta0'].iloc[0])
        try:
            output = self.MPC_function(MPC_input)
            output = casadi_helpers.DM2numpy(output, [2,1])
            output_datapoint = DSL_Data_Point(output = {'delta0' : output[0], 'v0' : output[1]})
        except:
            output_datapoint = DSL_Data_Point(output = {'delta0' : np.nan, 'v0' : np.nan})
            output_datapoint.error_included = True

        return output_datapoint

    def simulate_function(self, datapoint: DSL_Data_Point, Simulation_parameters) -> DSL_Data_Point:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(datapoint.input_dataframe['theta1'].iloc[0], datapoint.input_dataframe['x1'].iloc[0], datapoint.input_dataframe['y1'].iloc[0], datapoint.input_dataframe['theta0'].iloc[0])
        current_U = vertcat(datapoint.output_dataframe['delta0'].iloc[0], datapoint.output_dataframe['v0'].iloc[0])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=self.time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        output_datapoint = DSL_Data_Point(input = {'theta1': system_result[0], 'x1': system_result[1], 'y1': system_result[2], 'theta0': system_result[3]})

        return output_datapoint
    def give_next_state(self, datapoint: List[float], control_variables: List[float], time_step: float, idx: int) -> List[float]:
        """This function gives the next state for a state and control variables"""
        current_X = vertcat(input[0], input[1], input[2], input[3])
        current_U = vertcat(control_variables[0], control_variables[1])

        system_result = self.simulation_function(x0=current_X, u=current_U, T=time_step)["xf"]
        system_result=casadi_helpers.DM2numpy(system_result, [2,1])

        return [system_result[0], system_result[1], system_result[2], system_result[3]]

    # def sample_k_start_points(self, k: int):
    #     """This function gives k start points"""
    #     start_points = []

    #     for _ in range(k):
    #         start_pos = uniform(-0.6, -0.4)

    #         start_point = [start_pos, 0]
    #         start_points.append(start_point)
        
    #     end_iterations = [100]*k
        
    #     return start_points, end_iterations


class Truck_Trailer_HyperModel(kt.HyperModel):
    """This class contains the hypermodel"""
    def __init__(self, norm: tf.keras.layers.experimental.preprocessing.Normalization, amount_of_train_features: int):
        self.normalizer = norm
        self.amount_of_train_features = amount_of_train_features

    def build(self, hp: kt.engine.hyperparameters.HyperParameter) -> tf.keras.Sequential:
        hp_units = hp.Int('units', min_value=16, max_value=96, step=16)
        hp_activation_function = hp.Choice('activation_function', values=['sigmoid', 'tanh'])
        hidden_layers = hp.Int('hidden_layers', min_value=4, max_value=4, step=1)

        model = keras.Sequential()
        inputs = tf.keras.Input(shape=[self.amount_of_train_features,])
        model.add(inputs)
        #model.add(self.normalizer)
        for _ in range(int(hidden_layers)):
            model.add(layers.Dense(hp_units, activation=hp_activation_function))
        model.add(layers.Dense(2))

        hp_learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling="log")
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adam(hp_learning_rate))
        return model


def build_and_compile_model(norm: tf.keras.layers.experimental.preprocessing.Normalization, hyper_parameters: List, amount_of_train_features: int) -> tf.keras.Sequential:
    """This function gives the NN model"""
    model = keras.Sequential()
    inputs = tf.keras.Input(shape=[amount_of_train_features,])
    model.add(inputs)
    #model.add(norm)
    for _ in range(int(hyper_parameters[3])):
        model.add(layers.Dense(hyper_parameters[0], activation = hyper_parameters[1]))
    model.add(layers.Dense(2))

    model.compile(loss='mean_squared_error',
                optimizer=tf.keras.optimizers.Adam(hyper_parameters[2]))
    return model

@dataclass
class General_Simulation_parameters:
    function:str =False

def standalone_simulate_function(datapoint: DSL_Data_Point, General_Simulation_parameters, Simulation_parameters) -> DSL_Data_Point:
    """This function gives the next state for a state and control variables"""
    current_X = vertcat(datapoint.input_dataframe['theta1'].iloc[0], datapoint.input_dataframe['x1'].iloc[0], datapoint.input_dataframe['y1'].iloc[0], datapoint.input_dataframe['theta0'].iloc[0])
    current_U = vertcat(datapoint.output_dataframe['delta0'].iloc[0], datapoint.output_dataframe['v0'].iloc[0])

    system_result = General_Simulation_parameters.function(x0=current_X, u=current_U, T=0.7)["xf"]
    system_result=casadi_helpers.DM2numpy(system_result, [2,1])

    output_datapoint = DSL_Data_Point(input = {'theta1': system_result[0], 'x1': system_result[1], 'y1': system_result[2], 'theta0': system_result[3]})

    return output_datapoint

def initialize_ocp(theta1_init: float, x1_init: float, y1_init: float, theta0_init: float) -> List[Function]:
    """Function that initialize the ocp for the cart pole"""
    T_end=70.0
    ocp = Ocp(T=T_end)
    N = 100
    M = 1
    refine = 2
    Ts = T_end/N

    L0 = 0.3375
    M0 = 0.1
    W0 = 0.2
    L1 = 0.3
    M1 = 0.06
    W1 = 0.2

    x1_t0 = x1_init
    y1_t0 = y1_init
    theta1_t0 = theta1_init
    theta0_t0 = theta0_init
    x0_t0 = x1_t0 + L1*cos(theta1_t0) + M0*cos(theta0_t0)
    y0_t0 = y1_t0 + L1*sin(theta1_t0) + M0*sin(theta0_t0)

    x1_tf = 0.
    y1_tf = 0.
    theta1_tf = 0.
    theta0_tf = 0.

    # Trailer model
    theta1 = ocp.state()
    x1     = ocp.state()
    y1     = ocp.state()

    theta0 = ocp.state()
    x0     = x1 + L1*cos(theta1) + M0*cos(theta0)
    y0     = y1 + L1*sin(theta1) + M0*sin(theta0)

    delta0 = ocp.control(order=1)
    v0     = ocp.control(order=1)

    X_0 = ocp.parameter(4);

    beta01 = theta0 - theta1

    dtheta0 = v0/L0*tan(delta0)
    dtheta1 = v0/L1*sin(beta01) - M0/L1*cos(beta01)*dtheta0
    v1 = v0*cos(beta01) + M0*sin(beta01)*dtheta0

    ocp.set_der(theta1, dtheta1)
    ocp.set_der(x1,     v1*cos(theta1))
    ocp.set_der(y1,     v1*sin(theta1))

    ocp.set_der(theta0, dtheta0)

    X = vertcat(theta1, x1, y1, theta0)#, x0, y0)

    # Initial constraints
    ocp.subject_to(ocp.at_t0(X) == X_0)
    #ocp.subject_to(ocp.at_tf(X) == vertcat(theta1_tf, x1_tf, y1_tf, theta0_tf))

    # Final constraint
    ocp.subject_to(ocp.at_tf(x1) == x1_tf)
    ocp.subject_to(ocp.at_tf(y1) == y1_tf)
    ocp.subject_to(ocp.at_tf(theta1) == theta1_tf)
    ocp.subject_to(ocp.at_tf(beta01) == theta0_tf - theta1_tf)

    # # Initial guess
    # ocp.set_initial(theta0, .1)
    # ocp.set_initial(theta1, 0)
    # ocp.set_initial(v0,    -.2)
    # ocp.set_initial(x1,     np.linspace(x1_t0, x1_tf, N))
    # ocp.set_initial(y1,     np.linspace(y1_t0, y1_tf, N))

    # Path constraints
    ocp.subject_to(-.2 <= (v0 <= .2))
    ocp.subject_to(-1 <= (ocp.der(v0) <= 1))

    ocp.subject_to(-pi/6 <= (delta0 <= pi/6))
    ocp.subject_to(-pi/10 <= (ocp.der(delta0) <= pi/10))

    ocp.subject_to(-pi/2 <= (beta01 <= pi/2))

    # Minimal time
    #ocp.add_objective(ocp.T)
    #ocp.add_objective(ocp.integral(beta01**2))
    #ocp.add_objective(ocp.integral(1.001**ocp.t))
    #ocp.add_objective(ocp.integral((delta0**2+v0**2)))
    ocp.add_objective(ocp.integral(5*((x1-x1_tf)**2+(y1-y1_tf)**2)*((ocp.t/T_end)**2)))
    ocp.add_objective(ocp.integral(0.3*((theta1-theta1_tf)**2+(theta0-theta0_tf)**2)*((ocp.t/T_end)**2)))

    # Pick a solution method
    options = { "expand": True,
                "verbose": False,
                "print_time": False,
                "error_on_fail": True,
                "ipopt": {	#"linear_solver": "ma57",
                            "tol": 1e-8,
                            "print_level": 0}}
    ocp.solver('ipopt',options)

    # Make it concrete for this ocp
    ocp.method(MultipleShooting(N=N,M=M,intg='rk'))

    ocp.set_value(X_0, vertcat(theta1_t0, x1_t0, y1_t0, theta0_t0))#, L1+M0, 0))

    solve_ocp = ocp.to_function('solve_ocp',
                                [X_0],
                                [ocp.sample(delta0,grid='control')[1][0], ocp.sample(v0,grid='control')[1][1]])


    sim_system_dyn = ocp.discrete_system()

    return [solve_ocp, sim_system_dyn]

def get_perfect_paths() -> List[Data_set]:
    """This function gives the perfect paths"""

    with open("Truck_trailer_data\\truck_trailer_theta1_perfect_paths.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_theta1 = []
        for row in csvreader:
            MPC_theta1.append([float(i) for i in row])
        
    with open("Truck_trailer_data\\truck_trailer_x1_perfect_paths.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_x1 = []
        for row in csvreader:
            MPC_x1.append([float(i) for i in row])

    with open("Truck_trailer_data\\truck_trailer_y1_perfect_paths.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_y1 = []
        for row in csvreader:
            MPC_y1.append([float(i) for i in row])
        
    with open("Truck_trailer_data\\truck_trailer_theta0_perfect_paths.csv") as file_name:
        csvreader = csv.reader(file_name)
        MPC_theta0 = []
        for row in csvreader:
            MPC_theta0.append([float(i) for i in row])
    
    perfect_paths = []
    
    for idx in range(len(MPC_theta0)):
        perfect_path = Data_set(4, 0)
        perfect_path.load_data_from_col_list([MPC_theta0[idx], MPC_x1[idx], MPC_y1[idx], MPC_theta1[idx]])
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