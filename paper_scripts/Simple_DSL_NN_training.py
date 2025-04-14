import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h5_to_yaml import h5_to_yml
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from typing import List
from Use_Case_class import Use_Case, give_list_list
from imp_hyper_functions import hypertune, get_hyperparameters_from_file, save_hyperparametersearch_results, save_hyperparameters_to_file
from dataclasses import dataclass
sys.path.insert(0, 'C:\\Users\\Werk\\DSL')
from DSL_data_classes import DSL_Data_Point, DSL_Data_Set, DSL_Trace_Data_Set
from DSL_policy_case import Policy

@dataclass
class General_Train_NN_Policy_parameters:
    do_hypertuning: bool = False  
    hypertuning_epochs: int = 0
    hypertuning_factor: int = 0
    project_name: str = 'hyper_trials'
    NN_training_epochs: int = 0
    hyperparameter_file: str = ""
    use_case: Use_Case = False

@dataclass
class Train_NN_Policy_parameters:
    output_map: str = ""

@dataclass
class General_Retrain_NN_Policy_parameters:
    NN_training_epochs: int = 0
    use_case: Use_Case = False

@dataclass
class Retrain_NN_Policy_parameters:
    hyperparameter_file: str = ""
    saved_weights: str = ""
    output_map: str = ""

def Train_NN_Policy(input_dataset, General_Train_NN_Policy_parameters, train_NN_Policy_parameters):
    train_NN_Policy_parameters = Train_NN_Policy_parameters( output_map = 'delete_later2')
    train_dataset, test_dataset = input_dataset.split_dataset([0.8, 0.2])
    use_case = General_Train_NN_Policy_parameters.use_case

    if General_Train_NN_Policy_parameters.do_hypertuning:
        hyper_model = use_case.give_hypermodel(train_dataset.input_dataframe)
        Hypertune_NN(hyper_model, use_case, train_dataset, General_Train_NN_Policy_parameters, train_NN_Policy_parameters)
    
    hyper_parameters = get_hyperparameters_from_file(train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\'+ General_Train_NN_Policy_parameters.hyperparameter_file)
    dnn_model = use_case.give_NN_model(hyper_parameters, train_dataset.input_dataframe)

    history = fit_model(dnn_model, train_dataset, General_Train_NN_Policy_parameters.NN_training_epochs)
    save_training_results(use_case, dnn_model, history, hyper_parameters, General_Train_NN_Policy_parameters, train_NN_Policy_parameters, "training")

    policy = give_NN_policy(use_case, dnn_model)
    return policy

def Hypertune_NN(hyper_model, use_case, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters):
    output_map_name = Train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\'+ General_Train_NN_Policy_parameters.hyperparameter_file
    try:
        os.makedirs(Train_NN_Policy_parameters.output_map + '\output_NN_hypertuning\\')
    except:
        pass
    best_hps = hypertune(hyper_model, train_dataset, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters)
   
    save_hyperparametersearch_results(best_hps, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, use_case.hyperparameters)
    save_hyperparameters_to_file(use_case, best_hps, output_map_name)

def save_training_results(use_case, model, history, hyper_parameters, General_Train_NN_Policy_parameters, Train_NN_Policy_parameters, output_map_label):
    output_map_name = Train_NN_Policy_parameters.output_map + '\output_NN_'+ output_map_label
    try:
        os.makedirs(output_map_name)
    except:
        pass

    plot_loss(history, output_map_name)
    save_model(use_case, model, output_map_name)
    write_output_file(model, output_map_name, General_Train_NN_Policy_parameters, hyper_parameters, use_case.hyperparameters)

def Retrain_NN_Policy(input_dataset, General_Retrain_NN_Policy_parameters, retrain_NN_Policy_parameters):
    retrain_NN_Policy_parameters = Retrain_NN_Policy_parameters( output_map = 'delete_later2', saved_weights='delete_later2\output_NN_training\dnn_modelweigths.h5', hyperparameter_file='delete_later2\output_NN_hypertuning\hyperparameterfile')
    """Retrain a NN"""
    use_case = General_Retrain_NN_Policy_parameters.use_case
    train_dataset, test_dataset = input_dataset.split_dataset([0.8, 0.2])

    hyper_parameters = get_hyperparameters_from_file(retrain_NN_Policy_parameters.hyperparameter_file)
    saved_model= load_NN_from_weights(use_case, hyper_parameters, train_dataset.input_dataframe, retrain_NN_Policy_parameters.saved_weights)

    history = fit_model(saved_model, train_dataset, General_Retrain_NN_Policy_parameters.NN_training_epochs)
    save_training_results(use_case, saved_model, history, hyper_parameters, General_Retrain_NN_Policy_parameters, retrain_NN_Policy_parameters, "training")

    policy = give_NN_policy(use_case, saved_model)
    return policy


def fit_model(dnn_model: tf.keras.Sequential, train_dataset, Trained_epochs: int) -> keras.callbacks.History:
    """This function fit a model"""
    history = dnn_model.fit(
        train_dataset.input_dataframe, train_dataset.output_dataframe,
        validation_split=0.2,
        verbose=0, epochs=Trained_epochs)
    return history

def plot_loss(history: keras.callbacks.History, map_name: str):
    """This function makes and saves the figure with the loss and validation loss"""
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error  [U1,U2]')
    plt.legend()
    plt.grid(True)
    plt.savefig(map_name + '\Figure_loss_and_val_loss_')
    plt.clf()

def save_model(use_case: Use_Case, dnn_model: tf.keras.Sequential, map_name: str):
    """This function saves the model, the weights and saves it as a yaml."""
    dnn_model.save(map_name+'\dnn_model')
    dnn_model.save_weights(map_name+'\dnn_model' + 'weigths.h5')
    h5_to_yml([map_name+'\dnn_model', map_name + '\dnn_model' +'_yaml'], use_case.custom_objects)

def write_output_file(dnn_model: tf.keras.Sequential, output_map_name, General_Train_NN_Policy_parameters, hyper_parameters: List, hyperparameters_labels: List[str]):
    """This function saves the parameters of NN training and the test results"""
    outputfile= open(output_map_name + '\\' + '_outputfile.txt', "a")
    for idx, hyperparameter_label in enumerate(hyperparameters_labels):
        outputfile.write(f"{hyperparameter_label}: {hyper_parameters[idx]}\n")
    outputfile.write(f"The amount of trained epochs are: {General_Train_NN_Policy_parameters.NN_training_epochs}\n")
    outputfile.write(f"The loss function is: {dnn_model.loss}\n")
    dnn_model.summary(print_fn=lambda x: outputfile.write(x + '\n'))
    outputfile.close()

def load_NN_from_weights(use_case: Use_Case, hyper_parameters: List, train_features: pd.DataFrame, weights_file: str) -> tf.keras.Sequential:
    """This function loads a NN from saved weights"""
    NN = use_case.give_NN_model(hyper_parameters, train_features)
    NN.load_weights(weights_file)
    return NN

def give_NN_policy(use_case: Use_Case, NN: tf.keras.Sequential):
    def get_control_action_from_NN(input_datapoint):
        """This function gives a control action from the NN"""
        NN_output = NN(input_datapoint.input_dataframe)

        NN_output_dict = {}

        if len(use_case.labels_output) == 1:
            NN_output_dict[use_case.labels_output[0]] = NN_output.numpy()[0][0]
        else:
            for idx, NN_output_parameter in enumerate(NN_output):
                NN_output_dict[use_case.labels_output[idx]] = NN_output_parameter.numpy()[0][0]

        output_datapoint = DSL_Data_Point(output = NN_output_dict)

        return output_datapoint
    
    return Policy(get_control_action_from_NN)


if __name__ == "__main__":
  
  """## A DNN regression
  """

  Trained_epochs = 400

  name = 'V_mp_27_' + str(Trained_epochs) + '_epochs'
  map_name = 'dnns_motion_planning_example' + '\\' + name

  os.makedirs(map_name)

  DNN_units = 64
  DNN_activation = 'sigmoid'
  DNN_Learning_rate = 0.0112955831
  #do_a_NN_training('data_set_car_example.csv', map_name, name, Trained_epochs, DNN_units, DNN_activation, DNN_Learning_rate)