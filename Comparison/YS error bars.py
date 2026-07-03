import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === Experimental Data ===
x_exp = np.array([1, 2, 3, 8, 24, 48, 72, 120, 168])
y_exp = np.array([74, 89, 127, 187, 219, 230, 252, 280, 255])

# === Load Excel Data ===
file_path = "./plots/4-8 tf8/YS_predictions.xlsx"
columns_to_load = ["time (h)", "YS_interpolated (MPa)", "Y_i0", "Y_i200", "Y_i400", "Y_i600"]
df = pd.read_excel(file_path, usecols=columns_to_load)
time = df["time (h)"]

# === Set plot style ===
plt.rcParams.update({
    'font.size': 15,
})

# === Iteration curves to plot separately ===
iteration_columns = {
    "Y_i0": "0",
    "Y_i200": "200",
    "Y_i400": "400",
    "Y_i600": "600"
}

# === Loop through each iteration column and plot ===
for col, iter_num in iteration_columns.items():
    fig, ax = plt.subplots(figsize=(8, 7))

    # Experimental data as red 'x' scatter
    ax.scatter(x_exp, y_exp, label="Experimental [32]", s=40, marker='x', color='red')

    # Interpolated YS curve
    ax.plot(time, df["YS_interpolated (MPa)"], label="YS-interpolated", marker="o", markersize=3, color='grey')

    # Specific iteration YS prediction as scatter (s=5)
    ax.scatter(time, df[col], label=f'YS-predicted @ iteration {iter_num}', color='blue', s=5)


    # Labels and layout
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Yield Strength (MPa)")
    ax.set_title(f"YS Prediction vs Interpolated vs Experiment (Iteration {iter_num})")
    ax.legend()
    ax.grid(False)
    plt.tight_layout()

    # Save plot
    save_path = os.path.join('./plots', f'YS_prediction_iter{iter_num}.png')
    plt.savefig(save_path)
    plt.close()



















file_paths = [
    "./plots/4-1  tf2 0.01LR for p4,p5/YS_predictions.xlsx",
    "./plots/4-2 tf3/YS_predictions.xlsx",
    "./plots/4-4 tf4/YS_predictions.xlsx",
    "./plots/4-5 tf5/YS_predictions.xlsx",
    "./plots/4-8 tf8/YS_predictions.xlsx",
    "./plots/4-10 tf10/YS_predictions.xlsx",
    "./plots/4-11 tf11/YS_predictions.xlsx",
    "./plots/4-12 tf12/YS_predictions.xlsx",
    "./plots/4-13 tf13/YS_predictions.xlsx",
    "./plots/4-14 tf14/YS_predictions.xlsx"
]

plt.rcParams.update({
    'font.size': 15,       # Global font size
    # 'axes.titlesize': 15,  # Font size for titles
    # 'axes.labelsize': 15,  # Font size for x and y labels
    # 'xtick.labelsize': 15, # Font size for x-tick labels
    # 'ytick.labelsize': 15, # Font size for y-tick labels
    # 'legend.fontsize': 10, # Font size for legend
    # 'figure.titlesize': 15 # Font size for figure title
})

path = os.path.join('./plots', f'YS_error.png')

time = pd.read_excel(file_paths[0])["time (h)"]


yield_curves = []
for file in file_paths:
    df = pd.read_excel(file)
    yield_curves.append(df["Y_i990"])


yield_df = pd.concat(yield_curves, axis=1)
yield_df.columns = [f"Run_{i+1}" for i in range(len(yield_curves))]


mean_yield = yield_df.mean(axis=1)
std_yield = yield_df.std(axis=1)

plt.figure(figsize=(10, 6))

plt.plot(time, mean_yield, label="Mean Prediction", linewidth=2)

plt.fill_between(time, mean_yield - std_yield, mean_yield + std_yield,
                 alpha=0.3, label="±1 Std Dev")

for col in yield_df.columns:
    plt.plot(time, yield_df[col], linestyle='--', alpha=0.5)

plt.xlabel("Time (h)")
plt.ylabel("Yield Strength")
plt.title("Yield Strength Predictions with Uncertainty Band")
plt.legend()
# plt.grid(True)
plt.tight_layout()
# plt.show()
plt.savefig(path)















# === Load the loss data ===
loss_file = "./plots/4-8 tf8/loss.xlsx"  # Adjust path if needed
df = pd.read_excel(loss_file, usecols=["Iteration", "loss"])

iteration = df["Iteration"]
loss = df["loss"]

# === Plotting ===
fig, ax = plt.subplots()
ax.set(xscale='linear', xlabel='Iterations (#)', ylabel='Loss', title=f'MSE at iteration {iteration.iloc[-1]}')
ax.plot(iteration, loss, color='blue')
ax.legend(['loss'])

# === Save the figure ===
path = os.path.join('./plots', 'loss.png')
plt.tight_layout()
plt.savefig(path)
