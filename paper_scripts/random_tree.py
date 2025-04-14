# Importing the libraries
import numpy as np # for array operations
import pandas as pd # for working with DataFrames
import requests, io # for HTTP requests and I/O commands
import matplotlib.pyplot as plt # for data visualization
import sklearn
from imp_validate_datasets import plot_predict_vs_real, plot_histogram

# scikit-learn modules
from sklearn.model_selection import train_test_split # for splitting the data
from sklearn.metrics import mean_squared_error # for calculating the cost function
from sklearn.ensemble import RandomForestRegressor # for building the model
from DSL_data_classes import DSL_Data_Point
from DSL_policy_case import Policy

from dataclasses import dataclass

@dataclass
class General_Train_Random_Tree_Policy_Parameters:
    n_estimators: int = 100


def Train_Random_Tree_Policy(input_dataset, general_train_random_tree_policy_parameters, train_random_tree_policy_parameters):
    train_dataset, test_dataset = input_dataset.split_dataset([0.8, 0.2])
    model = RandomForestRegressor(n_estimators = general_train_random_tree_policy_parameters.n_estimators, random_state = 0)
    model.fit(train_dataset.input_dataframe, train_dataset.output_dataframe)
    policy = give_RT_policy(model)
    return policy

def give_RT_policy(model):
    def get_control_action_from_NN(input_datapoint):
        """This function gives a control action from the NN"""
        RT_output = model.predict(input_datapoint.input_dataframe)

        RT_output_dict = {}
        RT_output_dict['F'] = RT_output[0]

        output_datapoint = DSL_Data_Point(output = RT_output_dict)

        return output_datapoint
    
    return Policy(get_control_action_from_NN)