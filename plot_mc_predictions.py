import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('optimized_results/all_mc_final_predictions.xlsx')

# Experimental data
exp_den_time = np.array([1810.1, 3618.8, 14300.9, 29919.6, 59140.7, 270893.8])
exp_hv_time = exp_den_time
target_den_val = np.array([3.24e22, 3.24e22, 2.56e22, 2.15e22, 2.32e22, 1.18e21])
target_rad_val = np.array([32.5, 39.3, 46.5, 49.8, 48.7, 131.0])
exp_hv_val = np.array([75.9, 85.8, 95.1, 95.1, 93.2, 71.7])
hv_top = [77.9, 87.9, 97.0, 96.9, 94.8, 73.8]
hv_bottom = [73.8, 84.0, 93.3, 93.2, 91.2, 69.9]
mpr_top = [42.79616, 50.59788, 59.36818, 64.55315, 63.09573, 168.4404]
mpr_bottom = [22.75228, 27.10552, 32.2917, 34.58144, 34.3192, 91.61857]
tnd_top = [4.14e22, 4.2e22, 3.3e22, 2.77e22, 2.99e22, 1.52e21]
tnd_bottom = [2.24e22, 2.27e22, 1.78e22, 1.53e22, 1.63e22, 1e21]

exp_time_hrs = exp_den_time / 3600.0
hv_err_upper = np.array(hv_top) - exp_hv_val
hv_err_lower = exp_hv_val - np.array(hv_bottom)
tnd_err_upper = np.array(tnd_top) - target_den_val
tnd_err_lower = target_den_val - np.array(tnd_bottom)
mpr_err_upper = (np.array(mpr_top) - target_rad_val) / 10.
mpr_err_lower = (target_rad_val - np.array(mpr_bottom)) / 10.

fig, axes = plt.subplots(3, 1, figsize=(10, 10))
properties = ['pred_hv', 'pred_nd', 'pred_pr']
y_labels = ['Hardness (HV)', 'Number Density (m$^{-3}$)', 'Particle Radius (nm)']

# Plot individual MC runs (commented out)
# for mc_run in df['MC_Run'].unique():
#     data = df[df['MC_Run'] == mc_run]
#     for i, prop in enumerate(properties):
#         plot_data = data[prop] * 1e9 if prop == 'pred_pr' else data[prop]
#         axes[i].plot(data['Time_hours'], plot_data, alpha=0.3, color='blue')

for i, prop in enumerate(properties):
    mean = df.groupby('Time_hours')[prop].mean()
    std = df.groupby('Time_hours')[prop].std()
    if prop == 'pred_pr':
        mean = mean * 1e9
        std = std * 1e9
    axes[i].plot(mean.index, mean.values, 'k-', linewidth=2, label='Mean')
    axes[i].fill_between(mean.index, mean - std, mean + std, alpha=0.2, color='gray', label='±1 Std')
    
    if prop == 'pred_hv':
        axes[i].errorbar(exp_time_hrs, exp_hv_val, yerr=[hv_err_lower, hv_err_upper],
                        fmt='ro', markersize=5, capsize=5, label='Experimental')
    elif prop == 'pred_nd':
        axes[i].errorbar(exp_time_hrs, target_den_val, yerr=[tnd_err_lower, tnd_err_upper],
                        fmt='ro', markersize=5, capsize=5, label='Experimental')
    elif prop == 'pred_pr':
        axes[i].errorbar(exp_time_hrs, target_rad_val/10, yerr=[mpr_err_lower, mpr_err_upper],
                        fmt='ro', markersize=5, capsize=5, label='Experimental')
    
    axes[i].set_ylabel(y_labels[i])
    axes[i].set_xscale('log')
    if prop == 'pred_nd':
        axes[i].set_yscale('log')
    if prop == 'pred_pr':
        axes[i].set_ylim(0, 20)
    axes[i].grid(True)

axes[-1].set_xlabel('Time (hours)')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

plt.tight_layout()
plt.savefig('optimized_results/mc_predictions_plot.png', dpi=300, bbox_inches='tight')
plt.show()
