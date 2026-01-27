

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import numpy as np
import os

plt.rcParams.update({
    'font.size': 15,       # Global font size
    'axes.titlesize': 15,  # Font size for titles
    'axes.labelsize': 15,  # Font size for x and y labels
    'xtick.labelsize': 15, # Font size for x-tick labels
    'ytick.labelsize': 15, # Font size for y-tick labels
    'legend.fontsize': 10, # Font size for legend
    'figure.titlesize': 15 # Font size for figure title
})

file_paths = [f"./plots/parameters_mc{i}.xlsx" for i in range(1, 11)]

def format_with_uncertainty(mean_vals, std_vals):
    results = []
    for m, s in zip(mean_vals, std_vals):
        if s == 0 or np.isnan(s):
            results.append(f"{m}")
            continue

        # get order of magnitude
        order = int(np.floor(np.log10(abs(s))))
        # keep 1 significant figure (2 if first digit is 1 or 2 for clarity)
        first_digit = int(s / 10**order)
        sig_figs = 2 if first_digit in [1, 2] else 1

        # round uncertainty
        s_rounded = round(s, -order + (sig_figs - 1))
        # round mean to same decimal place
        decimals = max(-order + (sig_figs - 1), 0)
        m_rounded = round(m, decimals)

        results.append(f"{m_rounded} ± {s_rounded}")
    return results

parameter_rows = []
for file in file_paths:
    if os.path.exists(file):
        df = pd.read_excel(file)
        last_row = df.iloc[-1]  # extract last row
        parameter_rows.append(last_row)
        # print(f"MC run {file_paths.index(file)+1}: {last_row.to_dict()}")

param_df = pd.DataFrame(parameter_rows)
param_df = param_df.drop(columns=["Iteration"])

param_df.columns = ['$P_1$', '$P_2$', '$P_3$', '$P_4$', '$P_5$' ]
print(param_df)
# Calculate mean for each parameter
means = param_df[['$P_1$', '$P_2$', '$P_3$', '$P_4$', '$P_5$']].mean(axis=0)
print("Mean of each parameter:")
print(means)

# Calculate variance for each parameter 
variances = param_df.var(axis=0)
print("Variance of each parameter:")
print(variances)

param_df["Run"] = [f"Run {i+1}" for i in range(len(file_paths))]  

plt.figure(figsize=(10, 5))
parallel_coordinates(param_df, class_column="Run", colormap="viridis")
plt.title("Convergence and Stability Analysis under Shared-Shared Coefficient Reparameterization")
plt.ylabel("Parameter Value")
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('./plots/parameter_parallel_coordinates.png', bbox_inches='tight')





ys_file_paths = [
    "./plots/YS_results_adam_mc1.xlsx",
    "./plots/YS_results_adam_mc2.xlsx",
    "./plots/YS_results_adam_mc3.xlsx", 
    "./plots/YS_results_adam_mc4.xlsx",
    "./plots/YS_results_adam_mc5.xlsx",
    "./plots/YS_results_adam_mc6.xlsx",
    "./plots/YS_results_adam_mc7.xlsx",
    "./plots/YS_results_adam_mc8.xlsx",
    "./plots/YS_results_adam_mc9.xlsx",
    "./plots/YS_results_adam_mc10.xlsx",]

# Read last column (Y_i990) from each file
ys_values = []
for file in ys_file_paths:
    df = pd.read_excel(file)
    ys_values.append(df.iloc[:, -1].values)  # Get last column regardless of name
# Convert to numpy array for easier calculations
ys_array = np.array(ys_values)

# Calculate mean and std across the 10 runs
mean = np.mean(ys_array, axis=0)
std = np.std(ys_array, axis=0)
mean_std = format_with_uncertainty(mean, std)
print(f'mean_std: {mean_std}')

print(f'min_max of std: {min(std)} and {max(std)}')

# Create x-axis values (sample index)
df_time = pd.read_excel(ys_file_paths[0])
x = df_time['Time (h)'].values
# Plot mean and std
plt.figure(figsize=(10, 6))
plt.plot(x, mean, 'b-', linewidth=2, label='Mean')
plt.fill_between(x, mean-std, mean+std, color='red', alpha=0.3, label='±1 Std Dev')
plt.xlabel('Time (h)')
plt.ylabel('Yield Strength (MPa)')
plt.title('Yield Strength Outputs across Monte Carlo Initializations with Reduced Degrees of Freedom')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/ys_mc_runs.png', bbox_inches='tight')
