"""
validation_data.py

Purpose
-------
This script prepares and visualises the experimental validation datasets used by
the KWN validation framework.

It converts the processed experimental measurements into the quantities required
by the validation scripts, prints the processed values, and generates quick
inspection plots of the experimental data. No modelling, optimisation, or
interpolation is performed.

Main workflow
-------------
1. Hardness processing
   - Loads Hardness_processed.xlsx.
   - Converts Vickers hardness (HV) to yield strength (YS) using:

         YS = (HV - 16.0) / 0.33

   - Converts experimental time from seconds to hours.
   - Prints the processed yield-strength validation data.

2. Total Number Density (TND) processing
   - Loads TND_processed.xlsx.
   - Converts time from seconds to hours.
   - Extracts the experimental total number density values.
   - Prints the processed TND dataset.

3. Mean Particle Radius (MPR) processing
   - Loads MPR_processed.xlsx.
   - Converts time from seconds to hours.
   - Extracts the experimental mean particle radius values.
   - Prints the processed MPR dataset.

4. Experimental data visualisation
   - Generates scatter plots of:
       * Hardness versus time
       * Total Number Density versus time
       * Mean Particle Radius versus time
   - Uses logarithmic time scaling for all datasets.
   - Uses logarithmic y-axis scaling for TND and MPR.
   - Displays the plots for quick visual inspection of the processed
     experimental data.

Role of this file
-----------------
This script is a preprocessing and quality-control utility. It prepares the
experimental validation datasets for use by the KWN validation scripts and
allows the processed measurements to be visually inspected before model
validation.

Unlike the calibration and validation scripts, this file performs no physics
simulation, parameter optimisation, or model fitting. It simply converts,
summarises, and visualises the processed experimental datasets.
"""

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


