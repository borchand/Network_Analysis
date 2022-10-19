#read from excel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_excel('AllfileBig.xlsx', sheet_name='in')
df.index = df[0]
df = df.iloc[: , 2:]
df.head()


# remove days in which all are nan
shape = df.shape
df.dropna(inplace=True, axis=0, how='all')
print(f'df is {shape}   Shape after dropping all NA columns: {df.shape}')

