import os
import numpy as np
import pandas as pd
import skillsnetwork

data = pd.read_csv('iris_data.csv')

# Question 1
print(data.head())
print(data.shape[0])
print(data.columns.tolist())
print(data.dtypes)

# Question 2
data['species'] = data.species.str.replace('Iris-', '')
# or alternatively
# data['species'] = data.species.apply(lambda r: r.replace('Iris-', ''))
print(data.head())

# Qeustion 3_1
print('Number of each species present')
print(data.species.value_counts())

# Question 3_2
df = data.describe()
df.loc['range'] = df.loc['max'] - df.loc['min']
rest_col = ['mean', '25%', '50%', '75%', 'range']
df = df.loc[rest_col]
df.rename({'50%' : 'median'}, inplace=True)
print(df)

# Question 4
print(data.groupby('species').mean())
print(data.groupby('species').mean())







