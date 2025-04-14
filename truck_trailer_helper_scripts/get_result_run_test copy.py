import csv
import numpy as np

with open("truck_trailer_run_test_2.csv") as file_name:
    csvreader = csv.reader(file_name)
    results = []
    begins = []
    i=1
    for row in csvreader:
        if i<0:
            print(i)
        elif (i-1)%101==0:
            print("begin"+str(i))
            begins.append(row[0:4])
        elif (i-1)%101==100:
            print("end"+str(i))
            results.append(row[0:4])
        i=i+1

print(results)
print(begins)
result_array=[]
for result_ar in results:
    loop_result=0
    for result in result_ar:
        loop_result=loop_result+np.abs(float(result))
    result_array.append(loop_result)

print(result_array)


