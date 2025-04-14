import csv

with open("truck_trailer_run_test.csv") as file_name:
    csvreader = csv.reader(file_name)
    results = []
    begins = []
    i=1
    for row in csvreader:
        if i<1289:
            print(i)
        elif (i-1289)%101==0:
            print("begin"+str(i))
            begins.append(row)
        elif (i-1289)%101==100:
            print("end"+str(i))
            results.append(row)
        i=i+1

print(results)
print(begins)
