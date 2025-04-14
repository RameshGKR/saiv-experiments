import numpy as np
import pandas as pd
from DSL_data_classes import DSL_Data_Set

def find_amounts():
    percentages=[0.2, 0.5, 0.3]
    length = 13

    amounts=[]
    # get the minimum amount for all the percentages
    for percentage in percentages:
        amounts.append(np.floor(percentage*length))

    amounts = [int(x) for x in amounts]
    total_amount = sum(amounts)
    #if there are datapoint left add them in a way to minimize the error between the actual percentage and the asked percentage
    for _ in range(int(length-total_amount)):
        amounts = find_best_new_amounts(amounts, percentages, length)
    
    print(amounts)
    current_percentages=[x/length for x in amounts]
    print(current_percentages)

def find_best_new_amounts(amounts, percentages, length):
    min_error = 100
    best_amounts = amounts

    for index in range(len(amounts)):
        current_amounts = amounts.copy()
        current_amounts[index] += 1

        error = calculate_percentages_error(current_amounts, percentages, length)
        print('dkfjqm')
        print(error)
        if error < min_error:
            print('hey')
            print(min_error)
            min_error = error
            best_amounts = current_amounts

    return best_amounts

def calculate_percentages_error(current_amounts, percentages, length):
    current_amounts_array = np.array(current_amounts)
    percentages_array = np.array(percentages)

    current_percentages_array = current_amounts_array/length
    error = (np.square(current_percentages_array - percentages_array)).mean() # mean square error
    return error



if __name__ == "__main__":

    #find_amounts()
    dataframe_1 = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10], 'b': [2,4,6,8,10,12,14,16,18,20],'c': [3,6,9,12,15,18,21,24,27,30]})
    dataset = DSL_Data_Set()
    dataset.initialize_with_dataframe(input_dataframe=dataframe_1)

    datasets = dataset.split_dataset([0.33,0.33,0.33])
    
    for dataset in datasets:
        print(dataset.input)
