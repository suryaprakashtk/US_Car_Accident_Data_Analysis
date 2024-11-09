import pandas as pd

path = './Dataset/US_Accidents_March23.csv'

df = pd.read_csv(path, nrows=10) 
column_names = df.columns
print(column_names)
stats = df.describe()
print(stats)

x = 5