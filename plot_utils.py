# amazonQ prompts: 1) seperate the plots of YS and put the loss and paramter plots as horizontal subplots with
# subplot lables under them indicated by (a) and (b).

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 

import os
import pandas as pd
import numpy as np
excel_path = './plots/YS_predictions.xlsx'

newFolder = os.path.join('.', 'plots')
if not os.path.exists(newFolder): os.makedirs(newFolder)

plt.rcParams.update({
    'font.size': 15,       # Global font size
    'axes.titlesize': 15,  # Font size for titless
    'axes.labelsize': 15,  # Font size for x and y labels
    'xtick.labelsize': 15, # Font size for x-tick labels
    'ytick.labelsize': 15, # Font size for y-tick labels
    'legend.fontsize': 10, # Font size for legend
    'figure.titlesize': 15 # Font size for figure title
    # 'axes.titlesize': 15,  # Font size for titles
    # 'axes.labelsize': 15,  # Font size for x and y labels
    # 'xtick.labelsize': 15, # Font size for x-tick labels
    # 'ytick.labelsize': 15, # Font size for y-tick labels
    # 'legend.fontsize': 10, # Font size for legend
    # 'figure.titlesize': 15 # Font size for figure title
})
matplotlib.use('Agg')


def nse_score(Yield_t, Yield_interpolated):
    """
    Compute the Nash–Sutcliffe Efficiency (NSE) in percentage form.

    Parameters
    ----------
    Yield_t : array-like
        Predicted yield strength values from the model.
    Yield_interpolated : array-like
        Interpolated experimental yield strength values (ground truth).

    Returns
    -------
    nse_percent : float
        NSE expressed as a percentage (0–100%). Can be <0 if fit is very poor.
    """
    y_pred = np.array(Yield_t, dtype=float)
    y_true = np.array(Yield_interpolated, dtype=float)

    numerator = np.sum((y_true - y_pred)**  2)
    denominator = np.sum((y_true - np.mean(y_true))**  2)

    nse = 1 - (numerator / denominator)
    nse_percent = nse * 100

    return nse_percent

def visual(time, TotalNumberDensity_t, MeanParticleRadius_t, TotalVolFraction_t, Yield_t, loss_t, Yield_interpolated, x, y, N_optimizer, param1_t, param2_t, param3_t, param4_t, param5_t, iteration, optimizer, loss_basic_t=None, mc_run = None):
 

    if(iteration % 10 == 0):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        ax1.set(xscale='linear', xlabel='Time (h)', ylabel='Yield strength (MPa)')
        ax1.scatter(time[1:], Yield_t, color='blue', s=5)
        ax1.scatter(time[1:], Yield_interpolated, color='grey', s=5)
        ax1.scatter(x, y, color='red', s=30, marker='x')
        ax1.legend(['Predicted', 'Interpolated experimental', 'Experimental'])

        ax2.set(xscale='linear', xlabel='Iterations (#)', ylabel='Loss')
        ax2.plot(loss_t, color='blue')
        ax2.legend(['loss'])

        ax3.set(xscale='linear', xlabel='Iterations (#)', ylabel='Parameters')
        ax3.plot(param1_t)
        ax3.plot(param2_t)
        ax3.plot(param3_t)
        ax3.plot(param4_t)
        ax3.plot(param5_t)
        ax3.legend(['$P_1$', '$P_2$', '$P_3$', '$P_4$', '$P_5$'])

        plt.tight_layout()
        path = os.path.join('./plots', f'mc{mc_run}_combined_{optimizer}@{iteration}.png')
        plt.savefig(path)

        loss_excel_path = os.path.join('./plots', f'mc{mc_run}_loss_{optimizer}.xlsx')
        loss_data = {'Iteration': list(range(len(loss_t))), 'loss': loss_t}
        df_loss = pd.DataFrame(loss_data)
        os.makedirs(os.path.dirname(loss_excel_path), exist_ok=True)
        df_loss.to_excel(loss_excel_path, index=False)

        param_excel_path = os.path.join('./plots', f'mc{mc_run}_parameters.xlsx')
        param_data = {
            'Iteration': list(range(len(param1_t))),
            'P1': param1_t, 'P2': param2_t, 'P3': param3_t, 'P4': param4_t, 'P5': param5_t
        }
        df_params = pd.DataFrame(param_data)
        os.makedirs(os.path.dirname(param_excel_path), exist_ok=True)
        df_params.to_excel(param_excel_path, index=False)

    plt.close('all')

def plot_physics_results(time, TotalNumberDensity_t, MeanParticleRadius_t, TotalVolFraction_t, Yield_t, Yield_interpolated, x, y, iteration=0, mc_run=0):
    # Convert yield values to MPa
    y = [1000*i for i in y]
    Yield_t = 1000 * Yield_t
    Yield_interpolated = 1000*Yield_interpolated

    # Plot Total Number Density
    tnd_x = np.array([0.5, 1., 4., 8., 16., 76.])
    tnd_y = np.array([3.247514e+22, 3.247514e+22, 2.565907e+22, 2.154435e+22, 2.324538e+22, 1.181972e+21])
    fig, ax = plt.subplots()
    ax.set(xscale='log', yscale='log', xlabel='Time (h)', ylabel='Total number density')
    ax.scatter(time[1:], TotalNumberDensity_t, color='blue', s=5)
    ax.scatter(tnd_x, tnd_y, color='red', s=30, marker='x')
    ax.legend(['Total number density', 'Experimental data'])
    plt.savefig(f'./plots/mc{mc_run}_TND_physics@{iteration}.png')
    plt.close()

    # Plot Mean Particle Radius
    
    mpr_x = np.array([0.5, 1., 4., 8., 16., 76.])
    mpr_y = np.array([32.538446, 39.358762, 46.533809, 49.833394, 48.708346, 131.025383])*1e-10
    fig, ax = plt.subplots()
    ax.set(xscale='log', yscale='log', xlabel='Time (h)', ylabel='Mean particle radius (m)')
    ax.scatter(time[1:], MeanParticleRadius_t, color='blue', s=5)
    ax.scatter(mpr_x, mpr_y, color='red', s=30, marker='x')
    ax.legend(['Total number density', 'Experimental data'])
    plt.savefig(f'./plots/mc{mc_run}_MPR_physics@{iteration}.png')
    plt.close()

    # Plot Total Volume Fraction
    fig, ax = plt.subplots()
    ax.set(xscale='log', xlabel='Time (h)', ylabel='Total volume fraction')
    ax.scatter(time[1:], TotalVolFraction_t, color='blue', s=5)
    ax.legend(['Total volume fraction'], loc='lower right')
    # ax.set_ylim([0, 0.01])
    plt.savefig(f'./plots/mc{mc_run}_TVF_physics@{iteration}.png')
    plt.close()

    # Plot Yield Strength
    fig, ax = plt.subplots()
    ax.set(xscale='log', yscale='log', xlabel='Time (h)', ylabel='Yield strength (MPa)')
    ax.scatter(time[1:], Yield_t, color='blue', s=5)
    ax.scatter(time[1:], Yield_interpolated, color='grey', s=5)
    ax.scatter(x, y, color='red', s=30, marker='x')
    ax.legend(['Predicted', 'Interpolated experimental', 'Experimental'])
    plt.savefig(f'./plots/mc{mc_run}_YS_physics@{iteration}.png')
    plt.close()


# def visual_comparison():
#     fig, ax = plt.subplots()
#     ax.set(xscale='linear', xlabel='Iterations (#)', ylabel='Loss', title=f'MSE at iteration {iteration}')
#     ax.plot(loss_t, color = 'blue')
#     ax.legend(['loss'])
#     path = os.path.join('./plots', f'loss.png')
#     plt.savefig(path)
#     plt.show()

    