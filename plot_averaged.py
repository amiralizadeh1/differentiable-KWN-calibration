"""
plot_averaged.py

Purpose
-------
This script visualises previously processed experimental datasets used for KWN
model validation.

It reads Excel files containing averaged experimental measurements and generates
simple plots of the stored data points. The script performs no data processing,
optimisation, interpolation, or averaging; all averaging has already been
completed before these files are created.

Main workflow
-------------
1. Load processed datasets
   - Reads the following Excel files:
       * TND_point_top_bottom_averaged.xlsx
       * MPR_point_top_bottom_averaged.xlsx
       * Hardness_point_top_bottom_averaged.xlsx

2. Plot experimental points
   - Reads each row of the Excel file as a (Time, Value) pair.
   - Plots the experimental measurements.
   - Labels the axes and figure title.

3. Export figures
   - Saves one PNG figure for each dataset:
       * TND_point_top_bottom_averaged_plot.png
       * MPR_point_top_bottom_averaged_plot.png
       * Hardness_point_top_bottom_averaged_plot.png

Role of this file
-----------------
This file is a lightweight plotting utility used to quickly inspect the
processed experimental datasets before they are used for validation of the KWN
model.

Unlike the calibration and validation scripts, this file performs no numerical
modelling, optimisation, or statistical analysis. It simply visualises
preprocessed experimental data stored in Excel files.
"""

import pandas as pd
import matplotlib.pyplot as plt

# File names
files = ["TND_point_top_bottom_averaged", "MPR_point_top_bottom_averaged", "Hardness_point_top_bottom_averaged"]

for file in files:
    df = pd.read_excel(f"{file}.xlsx")
    
    plt.figure(figsize=(10, 6))
    for i in range(len(df)):
        plt.plot(df.iloc[i, 0], df.iloc[i, 1], marker='o', label=f'Row {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(file)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{file}_plot.png")
    plt.close()
    print(f"Created {file}_plot.png")
