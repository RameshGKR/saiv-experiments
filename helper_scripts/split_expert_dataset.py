import csv

""" with open('data_set_mpc_example_fixed_start.csv') as file:
    csvreader = csv.reader(file)
    row_array = []
    for idx, row in enumerate(csvreader):
        float_row=[]

        for jdx, x in enumerate(row):
            if jdx!=0 and jdx!=1:
                float_row.append(float(x))

        row_array.append(float_row)

with open('data_set_mpc_example_fixed_start_no_index.csv', "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    for row in row_array:
        writer.writerow(row)  """

""" with open('data_set_mpc_example_fixed.csv') as file:
    csvreader = csv.reader(file)
    row_array = []
    for idx, row in enumerate(csvreader):

        float_row =[float(x) for x in row]
        row_array.append(float_row)

with open('data_set_mpc_example_fixed_start.csv') as start_file:
    csvreader = csv.reader(start_file)
    start_row_array = []
    for idx, row in enumerate(csvreader):

        float_row =[float(x) for x in row]
        start_row_array.append(float_row)

total_row_list=[]
for idx, start_row in enumerate(start_row_array):
    total_row_list.append(start_row)
    
    for jdx in range(idx*100,(idx+1)*100):
        total_row_list.append(row_array[jdx])



with open('data_set_mpc_fixed_no_index.csv', "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    for row in total_row_list:
        writer.writerow(row) """


with open('data_set_mpc_example_fixed_no_index.csv') as file:
    csvreader = csv.reader(file)
    row_array = []
    for idx, row in enumerate(csvreader):
        if idx!=0 and (idx)%101==0:
            row_array.append([])

        float_row =[float(x) for x in row]
        row_array.append(float_row)


with open('data_set_mpc_example_fixed_no_index_perfect_paths.csv', "a+") as output:
    writer = csv.writer(output, lineterminator='\n')
    for row in row_array:
        writer.writerow(row)
