import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
from scipy.interpolate import PchipInterpolator

# ==========================================
# 0. SETUP OUTPUT FOLDER
# ==========================================
output_folder = "plots_myhr"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ==========================================
# 1. USER PROVIDED DATA (TARGETS)
# ==========================================

# Experimental Targets (Converted to TF Tensors for loss calc)
exp_den_time = np.array([
    1810.1, 3618.8, 14300.9, 
    29919.6, 59140.7, 270893.8
])
# Target tensors
target_den_val = tf.constant([
    3.24e22, 3.24e22, 2.56e22, 
    2.15e22, 2.32e22, 1.18e21
], dtype=tf.float32)

target_rad_val = tf.constant([
    32.5, 39.3, 46.5, 
    49.8, 48.7, 131.0
], dtype=tf.float32) # Angstroms

# Hardness Data
exp_hv_time = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 100.0]) * 3600.0 
exp_hv_val  = np.array([55, 80, 100, 115, 110, 85])

# Interpolate hardness data using PCHIP
x_hv_hours = exp_hv_time / 3600.0
x_hv_log = np.log(x_hv_hours)
pchip_hv = PchipInterpolator(x_hv_log, exp_hv_val) 

# ==========================================
# 2. DIFFERENTIABLE PHYSICS MODEL
# ==========================================

class KWNModel(tf.Module):
    def __init__(self):
        # --- TRAINABLE VARIABLE ---
        self.M = tf.Variable(3.0, dtype=tf.float32, name="M")
        
        # --- FIXED CONSTANTS ---
        self.R_gas = 8.314
        self.T = 185.0 + 273.15
        self.b = 2.84e-10
        self.G = 2.7e10
        self.rc = 5.0e-9 
        self.C_total_Mg = 0.55
        self.C_total_Si = 0.82
        self.vm = 7.62e-5
        self.Cp = 59.0
        self.A0 = 30000.0
        self.j0 = 1.8e35
        self.Qd = 130000.
        self.D0 = 2.5e-4
        self.Qs = 45350.
        self.Cs = 6.8
        self.gamma = 0.16
        self.beta = 0.46
        self.k_Mg = 29.0e6
        self.k_Si = 66.3e6
        self.sigma_i = 10.0e6
        self.coarse_factor = 10.

        # Time Settings
        self.n_steps = 500
        self.t_start = 10.0
        self.t_end = 300000.0
        
        # Generate Time Steps (Log Space)
        steps = tf.linspace(tf.math.log(self.t_start)/tf.math.log(10.0), 
                            tf.math.log(self.t_end)/tf.math.log(10.0), self.n_steps)
        self.time_seconds = tf.pow(10.0, steps)
        
        # Calculate dt
        dt = self.time_seconds[1:] - self.time_seconds[:-1]
        self.dt_seconds = tf.concat([[0.0], dt], axis=0)

        # Interpolate hardness at simulation time steps
        time_hours = self.time_seconds.numpy() / 3600.0
        time_hours_log = np.log(time_hours)
        hv_interpolated = pchip_hv(time_hours_log)
        self.target_hv = tf.constant(hv_interpolated, dtype=tf.float32)

    @tf.function
    def __call__(self):
        # 1. Dependent Constants
        D = self.D0 * tf.math.exp(-self.Qd / (self.R_gas * self.T))
        Ce = self.Cs * tf.math.exp(-self.Qs / (self.R_gas * self.T))
        
        # FIX 1: Enforce float32 Pi to prevent float64 casting disconnections
        PI = tf.constant(np.pi, dtype=tf.float32)
        
        # FIX 2: Initialize loop-carried state variables as Tensors
        current_ND = tf.constant(1.0e12, dtype=tf.float32)
        current_PR = tf.constant(0.5e-9, dtype=tf.float32)
        current_C_bar = tf.constant(self.C_total_Mg, dtype=tf.float32)
        
        # TensorArrays to store history
        ND_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        PR_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        HV_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        YS_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        C_hist  = tf.TensorArray(tf.float32, size=self.n_steps)
        Ci_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        is_growing_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        rate_growth_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        rate_coarse_hist = tf.TensorArray(tf.float32, size=self.n_steps)
        dr_dt_hist = tf.TensorArray(tf.float32, size=self.n_steps)

        # 3. Main Loop
        for i in tf.range(self.n_steps):
            dt = self.dt_seconds[i]
            
            # --- A. NUCLEATION ---
            supersat = tf.nn.relu(current_C_bar - Ce) 
            ln_S = tf.math.log((supersat + Ce) / Ce)
            is_supersat = tf.math.sigmoid((current_C_bar - Ce) * 1e5)
            
            barrier = (self.A0 / (self.R_gas * self.T))**3 * (1.0 / (ln_S**2 + 1e-9))
            nucleation_rate = self.j0 * tf.math.exp(-barrier) * tf.math.exp(-self.Qd/(self.R_gas*self.T))

            dN = nucleation_rate * dt * is_supersat
            current_ND = current_ND + dN

            # 1. Calculate Critical Radius (r*)
            safe_ln_S = tf.math.softplus(ln_S) + 1e-6
            r_star = (2 * self.gamma * self.vm) / (self.R_gas * self.T * safe_ln_S)
            
            # 2. Nucleation Radius
            r_nucleated = r_star * 1.05
            
            # 3. Weighted Average
            prev_ND = current_ND - dN
            radius_moment = (prev_ND * current_PR) + (dN * r_nucleated)
            safe_denom = current_ND + 1e-9
            mixed_radius = radius_moment / safe_denom
            
            is_established = tf.math.sigmoid((current_ND - 1.0) * 10.0)
            current_PR = (1.0 - is_established) * r_nucleated + (is_established * mixed_radius)

            # ==========================================
            # B. GROWTH & COARSENING (LSW HYBRID)
            # ==========================================
            Ci = Ce * tf.math.exp((2 * self.gamma * self.vm) / (self.R_gas * self.T * (current_PR + 1e-12)))
            rate_growth = ((current_C_bar - Ci) / (self.Cp - Ci)) * (D / (current_PR + 1e-12))
            
            k_coarse = (8 * self.gamma * self.vm * D * Ce) / (9 * self.R_gas * self.T)
            rate_coarse = (k_coarse / (tf.square(current_PR) + 1e-12)) * self.coarse_factor
            
            driving_force = current_C_bar - Ci
            is_growing = tf.math.sigmoid(driving_force * 1e5) 
            dr_dt = (is_growing * rate_growth) + ((1.0 - is_growing) * rate_coarse)

            current_PR = current_PR + dr_dt * dt
            current_PR = tf.nn.relu(current_PR - 5e-10) + 5e-10

            # ==========================================
            # C. COARSENING & MASS BALANCE (FIXED)
            # ==========================================
            
            # Update Density with Nucleation first (Ensure PI is used)
            vol_frac_current = current_ND * (4.0/3.0) * PI * tf.pow(current_PR, 3.0)
            current_C_bar = self.C_total_Mg - (self.Cp * vol_frac_current)
            
            depletion_threshold = Ce * 5.0  
            is_depleted = tf.math.sigmoid((depletion_threshold - current_C_bar) * 1e5) 
            
            decay_rate = 8e-7 * self.coarse_factor 
            decay_factor = tf.math.exp(-decay_rate * dt)
            N_decayed = current_ND * decay_factor
            
            current_ND = (1.0 - is_depleted) * current_ND + is_depleted * N_decayed
            
            # Use PI and tf.pow to maintain strict tensor operations
            r_mass_balance = tf.pow( (3.0 * vol_frac_current) / (4.0 * PI * (current_ND + 1e-12)), 1.0/3.0)
            current_PR = (1.0 - is_depleted) * current_PR + is_depleted * r_mass_balance
            current_C_bar = tf.maximum(current_C_bar, Ce)

            # --- D. PROPERTIES (STRENGTH) ---
            C_si = self.C_total_Si * (current_C_bar / self.C_total_Mg)
            sigma_ss = self.k_Mg * tf.pow(tf.maximum(current_C_bar, 1e-9), 2/3) + \
                       self.k_Si * tf.pow(tf.maximum(C_si, 1e-9), 2/3)
            
            f = vol_frac_current
            # Ensure PI is used here as well
            sig_cut = self.M * 2 * self.beta * self.G * (self.b / self.rc) * \
                      tf.sqrt(3 * f / (2 * PI)) * (current_PR / self.rc)
            sig_bypass = self.M * 2 * self.beta * self.G * (self.b / current_PR) * \
                         tf.sqrt(3 * f / (2 * PI))
            
            is_bypass = tf.math.sigmoid((current_PR - self.rc) * 1e9)
            sigma_p = (1.0 - is_bypass) * sig_cut + is_bypass * sig_bypass
            
            is_precipitated = tf.math.sigmoid((current_ND - 1e10) * 1e-5)
            sigma_p = sigma_p * is_precipitated
            
            sigma_y = self.sigma_i + sigma_ss + sigma_p
            hv = 0.33 * (sigma_y / 1e6) + 16.0
            
            # Store
            ND_hist = ND_hist.write(i, current_ND)
            PR_hist = PR_hist.write(i, current_PR)
            HV_hist = HV_hist.write(i, hv)
            YS_hist = YS_hist.write(i, sigma_y / 1e6)
            C_hist  = C_hist.write(i, current_C_bar)
            Ci_hist = Ci_hist.write(i, Ci)
            is_growing_hist = is_growing_hist.write(i, is_growing)
            rate_growth_hist = rate_growth_hist.write(i, rate_growth)
            rate_coarse_hist = rate_coarse_hist.write(i, rate_coarse)
            dr_dt_hist = dr_dt_hist.write(i, dr_dt)
            
        return (ND_hist.stack(), PR_hist.stack(), HV_hist.stack(), YS_hist.stack(), 
                C_hist.stack(), Ci_hist.stack(), is_growing_hist.stack(), 
                rate_growth_hist.stack(), rate_coarse_hist.stack(), dr_dt_hist.stack())

# ==========================================
# 3. TRAINING LOOP
# ==========================================

model = KWNModel()
optimizer = tf.optimizers.Adam(learning_rate=0.01) # Slower LR for stability

print("------------------------------------------------")
for var in model.trainable_variables:
    print(f"Initial {var.name}: {var.numpy():.4f}")
print("Starting Training...")
print("------------------------------------------------")

history_loss = []

for epoch in range(2):
    with tf.GradientTape() as tape:
        # Run Simulation
        pred_ND, pred_PR, pred_HV, pred_YS, pred_C_bar, pred_Ci, pred_is_growing, pred_rate_growth, pred_rate_coarse, pred_dr_dt = model()
        
        # Loss based on hardness MSE
        total_loss = tf.reduce_mean(tf.square(pred_HV - model.target_hv))

    # Gradients
    grads = tape.gradient(total_loss, model.trainable_variables)
    for var, grad in zip(model.trainable_variables, grads):
        print(f"Epoch {epoch:03d} | Gradient: {var.name}_grad = {grad.numpy():.6e}")
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    history_loss.append(total_loss.numpy())
    
    # Print variables for each epoch
    ys_np = pred_YS.numpy()
    # Calculate Ce once (it's constant)
    Ce = model.Cs * np.exp(-model.Qs / (model.R_gas * model.T))
    param_str = " | ".join([f"{var.name}: {var.numpy():.5f}" for var in model.trainable_variables])
    print(f"Epoch {epoch:03d} | Loss: {total_loss.numpy():.4f} | {param_str} | Ce: {Ce:.6f}")
    
    # Print all time steps
    print(f"\n  Time Step Details for Epoch {epoch}:")
    print(f"  {'Step':<6} {'Time(h)':<10} {'ND(m^-3)':<15} {'R(Å)':<10} {'HV':<8} {'C_bar':<10} {'Ci':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    time_s = model.time_seconds.numpy()
    nd_np = pred_ND.numpy()
    pr_np = pred_PR.numpy() * 1e10
    hv_np = pred_HV.numpy()
    c_bar_np = pred_C_bar.numpy()
    ci_np = pred_Ci.numpy()
    is_growing_np = pred_is_growing.numpy()
    rate_growth_np = pred_rate_growth.numpy()
    rate_coarse_np = pred_rate_coarse.numpy()
    dr_dt_np = pred_dr_dt.numpy()
    for i in range(0, len(time_s), max(1, len(time_s)//20)):
        print(f"  {i:<6} {time_s[i]/3600:<10.4f} {nd_np[i]:<15.2e} {pr_np[i]:<10.2f} {hv_np[i]:<8.2f} {c_bar_np[i]:<10.4f} {ci_np[i]:<10.4f}")
    print()
    
    # Plot for every epoch
    time_s = model.time_seconds.numpy()
    nd_np = pred_ND.numpy()
    pr_np = pred_PR.numpy() * 1e10
    hv_np = pred_HV.numpy()
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)
    yerr = [target_den_val.numpy() * 0.2, target_den_val.numpy() * 0.2]
    
    param_label = ", ".join([f"{var.name}={var.numpy():.3f}" for var in model.trainable_variables])
    axs[0].plot(time_s, nd_np, 'b-', linewidth=2, label=f'Model ({param_label})')
    axs[0].errorbar(exp_den_time, target_den_val.numpy(), yerr=yerr, fmt='bs', 
                    markersize=8, markerfacecolor='none', markeredgewidth=2, 
                    capsize=5, capthick=2, label='Exp. Data')
    axs[0].set_ylabel(r'Number Density ($m^{-3}$)')
    axs[0].set_title(f'Epoch {epoch} - Loss: {total_loss.numpy():.4f}')
    axs[0].grid(True, which="both", ls="-")
    axs[0].set_yscale('log')
    axs[0].set_ylim(1e19, 1e24)
    axs[0].legend(loc='upper right')
    
    axs[1].plot(time_s, pr_np, 'g-', linewidth=2, label='Model Radius')
    axs[1].plot(exp_den_time, target_rad_val.numpy(), 'go', markersize=8, 
                markerfacecolor='none', markeredgewidth=2, label='Exp. Data')
    axs[1].set_ylabel(r'Mean Radius ($\mathring{A}$)')
    axs[1].grid(True, which="both", ls="-")
    axs[1].legend(loc='upper left')
    
    axs[2].plot(time_s, hv_np, 'r-', linewidth=2, label='Model Hardness')
    axs[2].plot(exp_hv_time, exp_hv_val, 'kd', markersize=8, label='Exp. Data')
    axs[2].set_ylabel('Hardness (HV)')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_xscale('log')
    axs[2].grid(True, which="both", ls="-")
    axs[2].legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"epoch_{epoch:03d}.png"), dpi=300)
    plt.close()

print("------------------------------------------------")
for var in model.trainable_variables:
    print(f"Final Optimized {var.name}: {var.numpy():.5f}")
print("------------------------------------------------")

# ==========================================
# 4. FINAL EXPORT & PLOTTING
# ==========================================

# Run one last time with optimized M (No Tape needed)
final_ND, final_PR, final_HV, final_YS, final_C_bar, final_Ci, final_is_growing, final_rate_growth, final_rate_coarse, final_dr_dt = model()

# Convert to NumPy
time_s = model.time_seconds.numpy()
nd_np = final_ND.numpy()
pr_np = final_PR.numpy() * 1e10 # Angstroms
hv_np = final_HV.numpy()
ys_np = final_YS.numpy()

# -- Export to Excel --
df = pd.DataFrame({
    'Time (s)': time_s,
    'Time (h)': time_s / 3600.0,
    'Number Density (m^-3)': nd_np,
    'Mean Radius (Angstroms)': pr_np,
    'Hardness (HV)': hv_np,
    'Yield Strength (MPa)': ys_np
})

excel_path = os.path.join(output_folder, "myhr_results_corrected.xlsx")
try:
    df.to_excel(excel_path, index=False)
    print(f"Data exported to: {excel_path}")
except ImportError:
    csv_path = os.path.join(output_folder, "myhr_results_corrected.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data exported to: {csv_path}")

# -- Plotting --
fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

# 1. Number Density
# Try to recreate error bars if we can approximate them
# (Since we don't have the original file, we mock 10% error bars for visual fidelity to original request)
yerr = [target_den_val.numpy() * 0.2, target_den_val.numpy() * 0.2] 

param_label = ", ".join([f"{var.name}={var.numpy():.3f}" for var in model.trainable_variables])
axs[0].plot(time_s, nd_np, 'b-', linewidth=2, label=f'Model ({param_label})')
axs[0].errorbar(exp_den_time, target_den_val.numpy(), yerr=yerr, fmt='bs', 
                markersize=8, markerfacecolor='none', markeredgewidth=2, 
                capsize=5, capthick=2, label='Exp. Data')

axs[0].set_ylabel(r'Number Density ($m^{-3}$)')
axs[0].set_title('Evolution of Microstructure (Alloy 4 @ 185°C) - Optimized')
axs[0].grid(True, which="both", ls="-")
axs[0].set_yscale('log')
axs[0].set_ylim(1e19, 1e24) 
axs[0].legend(loc='upper right')

# 2. Mean Radius
axs[1].plot(time_s, pr_np, 'g-', linewidth=2, label='Model Radius')
axs[1].plot(exp_den_time, target_rad_val.numpy(), 'go', markersize=8, 
            markerfacecolor='none', markeredgewidth=2, label='Exp. Data')
axs[1].set_ylabel(r'Mean Radius ($\mathring{A}$)')
axs[1].grid(True, which="both", ls="-")
axs[1].legend(loc='upper left')

# 3. Hardness
axs[2].plot(time_s, hv_np, 'r-', linewidth=2, label='Model Hardness')
axs[2].plot(exp_hv_time, exp_hv_val, 'kd', markersize=8, label='Exp. Data')
axs[2].set_ylabel('Hardness (HV)')
axs[2].set_xlabel('Time (s)')
axs[2].set_xscale('log')
axs[2].grid(True, which="both", ls="-")
axs[2].legend(loc='upper left')

plot_file_path = os.path.join(output_folder, "myhr_simulation_corrected.png")
plt.tight_layout()
plt.savefig(plot_file_path, dpi=300)

print(f"Plot saved to: {plot_file_path}")
plt.show()
