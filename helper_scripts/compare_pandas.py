import pandas as pd
import numpy as np
 
# data's stored in dictionary
details = {
    'Column1': [1],
    'Column2': [1.2],
    'Column3': [1.5],
    'Column4': [0.8]
}

details2 = {
    'Column1': [1],
    'Column2': [1],
    'Column3': [1],
    'Column4': [1]
}
# creating a Dataframe object
df = pd.DataFrame(details)
df2 = pd.DataFrame(details2)
diff = 0.5
totalbool = True


for output_label in df.columns:
    print(output_label)
    bool=df[output_label][0]-diff<=df2[output_label][0]<=df[output_label][0]+diff
    totalbool=bool and totalbool

print(totalbool)