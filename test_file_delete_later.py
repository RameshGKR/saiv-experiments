import pandas as pd

d = {'col1': [1], 'col2': [3]}
df = pd.DataFrame(data=d)
print('hey')
print(df['col2'][0])