import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_hardness = pd.read_excel('Hardness_processed.xlsx')
df_hardness['ys'] = (df_hardness[df_hardness.columns[1]] - 16.0) / 0.33
df_hardness['time_h'] = df_hardness['averaged x'] / 3600
print(df_hardness[['time_h', 'ys']])

df_tnd = pd.read_excel('TND_processed.xlsx')
df_tnd['time_h'] = df_tnd['averaged x'] / 3600
df_tnd['TND'] = df_tnd[df_tnd.columns[1]]
print(df_tnd[['time_h', 'TND']])

df_mpr = pd.read_excel('MPR_processed.xlsx')
df_mpr['time_h'] = df_mpr['averaged x'] / 3600
df_mpr['MPR'] = df_mpr[df_mpr.columns[1]]
print(df_mpr[['time_h', 'MPR']])
files = ['Hardness.xlsx', 'TND.xlsx', 'MPR.xlsx']
labels = ['Hardness', 'TND', 'MPR']

for file, label in zip(files, labels):
    df = pd.read_excel(file.replace('.xlsx', '_processed.xlsx'))
    plt.figure()
    plt.scatter(df['averaged x'], df[df.columns[1]], alpha=0.6)
    plt.xscale('log')
    if label in ['TND', 'MPR']:
        plt.yscale('log')
    plt.xlabel('Time')
    plt.ylabel(label)
    plt.title(f'{label} vs Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


