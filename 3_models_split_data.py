import csv

with open("truck_trailer_multi_stage_loop_traces_index_v1_dataset.csv") as file_name:
    csvreader = csv.reader(file_name)
    MPC_theta1 = []
    for idx,row in enumerate(csvreader):
        if idx==0:
            with open('truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_1.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:9])

            with open('truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_2.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:9])

            with open('truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_3.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:9])

        if (idx-1)%50<20:
            with open('truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_1.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:9])
        elif (idx-1)%50>19 and (idx-1)%50<30:
            with open('truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_2.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:9])
        else:
            with open('truck_trailer_multi_stage_loop_traces_index_v1_dataset_model_3.csv', "a+") as output_file:
                writer = csv.writer(output_file, lineterminator='\n')
                writer.writerow(row[0:9])
