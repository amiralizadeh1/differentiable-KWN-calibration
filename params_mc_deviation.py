

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
# regularization parameters for 10 MC runs
param1_reg = [0.624, 0.625, 0.625120997428894]
param2_reg = [0.833, 0.834, 0.8344861268997192]
param3_reg = [1.307, 1.308, 1.3085203170776367]
param4_reg = [0.544, 0.545, 0.5454251766204834]
param5_reg = [0.575, 0.576, 0.5759429335594177]

# without regularization parameters for 10 MC runs
param1 = [1.036, 1.049, 1.049, 1.061, 1.072, 1.083, 1.093, 1.102, 1.102, 1.102]
param2 = [0.966, 0.974, 0.981, 0.981, 0.986, 0.987, 0.987, 0.992, 1.004, 1.004]
param3 = [0.992, 1.018, 1.018, 1.025, 1.045, 1.072, 1.081, 1.081, 1.140, 1.172]
param4 = [0.972, 0.983, 0.983, 0.983, 1.018, 1.018, 1.018, 1.036, 1.062, 1.087]
param5 = [1.087, 0.879, 0.879, 0.937, 0.980, 1.002, 1.005, 1.005, 1.078, 1.126]


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

param_df = pd.DataFrame({
    '$P_1$': param1,
    '$P_2$': param2,
    '$P_3$': param3,
    '$P_4$': param4,
    '$P_5$': param5
})

param_df_reg = pd.DataFrame({
    '$P_1$': param1_reg,
    '$P_2$': param2_reg,
    '$P_3$': param3_reg,
    '$P_4$': param4_reg,
    '$P_5$': param5_reg
})

print("Without Regularization:")
means = param_df.mean(axis=0)
print("Mean of each parameter:")
print(means)
stds = param_df.std(axis=0)
print("Std of each parameter:")
print(stds)

print("\nWith Regularization:")
means_reg = param_df_reg.mean(axis=0)
print("Mean of each parameter:")
print(means_reg)
stds_reg = param_df_reg.std(axis=0)
print("Std of each parameter:")
print(stds_reg)

fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(means))

ax.plot(x_pos, means, color='steelblue', linewidth=2, marker='o', label='With Unity Regularisation')
ax.fill_between(x_pos, means - stds, means + stds, color='steelblue', alpha=0.3)

ax.plot(x_pos, means_reg, color='darkorange', linewidth=2, marker='s', label='With Unity and Parameter Rebounding Regularisation')
ax.fill_between(x_pos, means_reg - stds_reg, means_reg + stds_reg, color='darkorange', alpha=0.3)

ax.set_xticks(x_pos)
ax.set_xticklabels(means.index)
ax.set_ylabel("Parameter Value")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./parameter_parallel_coordinates.png', bbox_inches='tight')
plt.show()





