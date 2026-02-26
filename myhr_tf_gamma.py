import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from scipy.interpolate import PchipInterpolator

# ==========================================
# 0. SETUP & DATA
# ==========================================
output_folder = "optimized_results"
if not os.path.exists(output_folder): os.makedirs(output_folder)

# Experimental Targets
exp_den_time = np.array([1810.1, 3618.8, 14300.9, 29919.6, 59140.7, 270893.8])
target_den_val = tf.constant([3.24e22, 3.24e22, 2.56e22, 2.15e22, 2.32e22, 1.18e21], dtype=tf.float32)
target_rad_val = tf.constant([32.5, 39.3, 46.5, 49.8, 48.7, 131.0], dtype=tf.float32) 
exp_hv_time = exp_den_time
exp_hv_val  = np.array([55, 80, 100, 115, 110, 85])

# Interpolation for Hardness (Target)
# 1. Add an anchor point to prevent extrapolation artifacts
# Insert a very small time (e.g., 0.01) and duplicate the first hardness value
x_anchored = np.insert(exp_hv_time, 0, 0.01)
y_anchored = np.insert(exp_hv_val, 0, 20)

# 2. Transform x to Log Scale 
x_log = np.log(x_anchored)

# 3. Create the PCHIP Interpolator using the anchored data
pchip_interpolator = PchipInterpolator(x_log, y_anchored)

# 4. Generate Interpolated Values
query_times_log = np.log(x_anchored)
Yield_interpolated_values = pchip_interpolator(query_times_log)
ip_hv = PchipInterpolator(np.log(x_anchored/3600.0), y_anchored)

# Physical Constants
R_GAS = 8.314
TEMP = 185.0 + 273.15
B_VEC = 2.84e-10
G_MOD = 2.7e10
RC = 5.0e-9 
C_TOT_MG = 0.55
VM = 7.62e-5
CP = 59.0
GAMMA_BASE = 0.16 
N_STEPS = 50
M = 3.1

# Time Grid
time_seconds = tf.pow(10.0, tf.linspace(tf.math.log(360.0)/tf.math.log(10.0), tf.math.log(300000.0)/tf.math.log(10.0), N_STEPS))
dt_seconds = tf.concat([[0.0], time_seconds[1:] - time_seconds[:-1]], axis=0)
target_hv_interp = tf.constant(ip_hv(np.log(time_seconds.numpy()/3600.0)), dtype=tf.float32)

# ==========================================
# 1. PHYSICS MODEL FUNCTION
# ==========================================

def run_physics_simulation(params):
    """
    params: 
    p1: Ci equation (Gamma multiplier)
    p2: Nucleation rate
    p3: Growth rate coefficient
    p4: Growth rate parameter
    p5: Strength model multiplier
    """
    p1, p2, p3, p4, p5 = params
    
    # Dependent Constants
    D = 2.5e-4 * tf.math.exp(-130000. / (R_GAS * TEMP))
    Ce = 6.8 * tf.math.exp(-45350. / (R_GAS * TEMP))
    PI = tf.constant(np.pi, dtype=tf.float32)
    
    # Initial state
    current_ND = tf.constant(1.0e12, dtype=tf.float32)
    current_PR = tf.constant(1.e-9, dtype=tf.float32)
    current_C_bar = tf.constant(C_TOT_MG, dtype=tf.float32)
    
    HV_hist = tf.TensorArray(tf.float32, size=N_STEPS)
    ND_hist = tf.TensorArray(tf.float32, size=N_STEPS)
    PR_hist = tf.TensorArray(tf.float32, size=N_STEPS)
    Ci_hist = tf.TensorArray(tf.float32, size=N_STEPS)
    Cbar_hist = tf.TensorArray(tf.float32, size=N_STEPS)

    for i in tf.range(N_STEPS):
        dt = dt_seconds[i]
        
        # --- A. NUCLEATION ---
        supersat = tf.nn.relu(current_C_bar - Ce)
        ln_S = tf.math.log((supersat + Ce) / Ce)
        barrier = (30000.0 / (R_GAS * TEMP))**3 * (1.0 / (ln_S**2 + 1e-9))
        nucleation_rate =  1.8e35 * tf.math.exp(-barrier * tf.pow(p2, 3)) * tf.math.exp(-130000./(R_GAS*TEMP))
        
        dN = nucleation_rate * dt * tf.math.sigmoid((current_C_bar - Ce) * 1e5)
        current_ND += dN
        
        # --- B. NATURAL GROWTH ---
        # Interfacial concentration modified by p1 (Gamma)
        Ci = Ce * tf.math.exp((2 * (p1 * GAMMA_BASE) * VM) / (R_GAS * TEMP * (current_PR + 1e-12)))
        
        # Growth velocity modified by p3 and p4
        rate_growth = p3 * ((current_C_bar - Ci) / (CP - Ci)) * (D / (current_PR + 1e-12))
        
        # Standard LSW coarsening
        k_coarse = (8 * GAMMA_BASE * VM * D * Ce) / (9 * R_GAS * TEMP)
        rate_coarse = p4 * k_coarse / (tf.square(current_PR) + 1e-12)
        
        is_growing = tf.math.sigmoid((current_C_bar - Ci) * 1e5)
        dr_dt = (is_growing * rate_growth) + ((1.0 - is_growing) * rate_coarse)
        
        # Natural radius update
        current_PR = tf.nn.relu(current_PR + dr_dt * dt - 5e-10) + 5e-10

        # --- C. ARTIFICIAL DECAY & FORCED RADIUS JUMP ---
        # 1. Check volume fraction and current solute state before artificial interventions
        vol_frac_temp = current_ND * (4.0/3.0) * PI * tf.pow(current_PR, 3.0)
        cbar_temp = C_TOT_MG - (CP * vol_frac_temp)
        
        # 2. Setup the depletion threshold trigger
        # Triggers when matrix is highly depleted (near 5x Ce as in Code 2)
        depletion_threshold = Ce * 5.0  
        is_depleted = tf.math.sigmoid((depletion_threshold - cbar_temp) * 1e5) 
        
        # 3. Apply the Artificial Exponential Decay to Number Density
        decay_rate = 8e-7 * 10.0 
        decay_factor = tf.math.exp(-decay_rate * dt)
        N_decayed = current_ND * decay_factor
        
        # Switch ND to the decayed value if depleted
        current_ND = (1.0 - is_depleted) * current_ND + is_depleted * N_decayed
        
        # 4. Force Mass Balance on the Radius
        # Calculate what R must be to maintain the volume fraction with the new dropped ND
        r_mass_balance = tf.pow((3.0 * vol_frac_temp) / (4.0 * PI * (current_ND + 1e-12)), 1.0/3.0)
        
        # Switch PR to the mass balance value if depleted
        current_PR = (1.0 - is_depleted) * current_PR + is_depleted * r_mass_balance

        # 5. Final Mass Balance Check for the step
        vol_frac_final = current_ND * (4.0/3.0) * PI * tf.pow(current_PR, 3.0)
        current_C_bar = tf.maximum(C_TOT_MG - (CP * vol_frac_final), Ce)

        # --- D. STRENGTH ---
        sigma_ss = 29.0e6 * tf.pow(current_C_bar, 2/3) + 66.3e6 * tf.pow(current_C_bar * 1.5, 2/3)
        
        # Strength model modified by p5
        sig_cut = p5 * M * 2 * 0.46 * G_MOD * (B_VEC / RC) * tf.sqrt(3 * vol_frac_final / (2 * PI)) * (current_PR / RC)
        sig_bypass = p5 * M * 2 * 0.46 * G_MOD * (B_VEC / current_PR) * tf.sqrt(3 * vol_frac_final / (2 * PI))
        
        is_bypass = tf.math.sigmoid((current_PR - RC) * 1e9)
        sigma_p = (1.0 - is_bypass) * sig_cut + is_bypass * sig_bypass
        
        sigma_y = 10.0e6 + sigma_ss + sigma_p
        hv = 0.33 * (sigma_y / 1e6) + 16.0
        
        HV_hist = HV_hist.write(i, hv)
        ND_hist = ND_hist.write(i, current_ND)
        PR_hist = PR_hist.write(i, current_PR)
        Ci_hist = Ci_hist.write(i, Ci)
        Cbar_hist = Cbar_hist.write(i, current_C_bar)

    return HV_hist.stack(), ND_hist.stack(), PR_hist.stack(), Ci_hist.stack(), Cbar_hist.stack()

# ==========================================
# 2. MONTE CARLO OPTIMIZATION LOOP
# ==========================================

def save_iteration_plots(mc_run, iteration, time_sec, pred_hv, pred_nd, pred_pr, folder):
    """Save three plots: number density, particle radius, and hardness"""
    mc_folder = os.path.join(folder, f"mc{mc_run}")
    
    time_hrs = time_sec / 3600.0
    
    # Plot 1: Number Density
    nd_folder = os.path.join(mc_folder, "number_density")
    os.makedirs(nd_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.loglog(time_hrs, pred_nd, 'b-', linewidth=2, label='Predicted')
    plt.loglog(exp_den_time/3600.0, target_den_val.numpy(), 'ro', markersize=8, label='Validation Data')
    plt.ylim(1e19, 1e23)
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Number Density (m⁻³)', fontsize=12)
    plt.title(f'Number Density Iteration {iteration}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(nd_folder, f'iter{iteration:03d}.png'), dpi=150)
    plt.close()
    
    # Plot 2: Particle Radius
    pr_folder = os.path.join(mc_folder, "particle_radius")
    os.makedirs(pr_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.semilogx(time_hrs, pred_pr * 1e9, 'g-', linewidth=2, label='Predicted')
    plt.semilogx(exp_den_time/3600.0, target_rad_val.numpy()/10, 'ro', markersize=8, label='Validation Data')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Particle Radius (nm)', fontsize=12)
    plt.title(f'Particle Radius Iteration {iteration}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(pr_folder, f'iter{iteration:03d}.png'), dpi=150)
    plt.close()
    
    # Plot 3: Hardness
    hv_folder = os.path.join(mc_folder, "hardness")
    os.makedirs(hv_folder, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(time_hrs, pred_hv, 'b-', linewidth=2, label='Predicted')
    plt.plot(exp_hv_time/3600.0, exp_hv_val, 'ro', markersize=8, label='Training Data')
    plt.plot(time_hrs, target_hv_interp.numpy(), 'ko', linewidth=1.5, alpha=0.7, label='Interpolation')
    plt.xscale('log')
    plt.xlabel('Time (hours)', fontsize=12)
    plt.ylabel('Hardness (HV)', fontsize=12)
    plt.title(f'Hardness Iteration {iteration}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(hv_folder, f'iter{iteration:03d}.png'), dpi=150)
    plt.close()

N_MC_RUNS = 10
N_ITERATIONS = 500
LR_p1 = 1.0
LR_p2 = 1.
LR_p3 = 1.
LR_p4 = 1.0
LR_p5 = 1.0
LR_ADAM = 0.1

# Gradient clipping values for each parameter
CLIP_p1 = 10.0
CLIP_p2 = 10.
CLIP_p3 = 10.
CLIP_p4 = 10.
CLIP_p5 = 10.

for mc in range(1, N_MC_RUNS + 1):
    print(f"\n{'='*60}")
    print(f"Monte Carlo Run {mc}/{N_MC_RUNS}")
    print(f"{'='*60}")
    tf.random.set_seed(mc)
    
    # Initialize parameters (all trainable)
    p1 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.01), name="p1")
    p2 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.01), name="p2")
    p3 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.01), name="p3")
    p4 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.01), name="p4")
    p5 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.01), name="p5")
    
    params_list = [p1, p2, p3, p4, p5]
    trainable_params = [p1, p2, p3, p4, p5]
    opt = tf.keras.optimizers.Adam(learning_rate=LR_ADAM)
    
    history_loss = []
    history_basic_loss = []
    history_reg_loss = []
    history_p1 = []
    history_p2 = []
    history_p3 = []
    history_p4 = []
    history_p5 = []

    for ii in range(1, N_ITERATIONS+1):
        iter_start = time.time()
        
        with tf.GradientTape() as tape:
            pred_hv, pred_nd, pred_pr, pred_ci, pred_cbar = run_physics_simulation(params_list)
            # Loss: Mean Squared Error + L2 Regularization (Unity constraints)
            basic_loss = tf.reduce_mean(tf.square((pred_hv - target_hv_interp)))
            reg_loss = 1. * (tf.nn.l2_loss(p1 - 1.0) + tf.nn.l2_loss(p2 - 1.0) + tf.nn.l2_loss(p3 - 1.0) + tf.nn.l2_loss(p4 - 1.0) + tf.nn.l2_loss(p5 - 1.0))
            total_loss = basic_loss + reg_loss

        gradients = tape.gradient(total_loss, trainable_params)
        
        # Check for None gradients to ensure stability
        if any(g is None for g in gradients):
            none_params = [trainable_params[i].name for i, g in enumerate(gradients) if g is None]
            print(f"Warning: None gradient detected at iteration {ii} for parameters: {none_params}")
            continue
        
        # Apply gradient clipping
        clip_values = [CLIP_p1, CLIP_p2, CLIP_p3, CLIP_p4, CLIP_p5]
        clipped_gradients = [tf.clip_by_value(grad, -clip_val, clip_val) for grad, clip_val in zip(gradients, clip_values)]
        
        gradients_and_vars = [
            (clipped_gradients[0] * LR_p1, p1),
            (clipped_gradients[1] * LR_p2, p2),
            (clipped_gradients[2] * LR_p3, p3),
            (clipped_gradients[3] * LR_p4, p4),
            (clipped_gradients[4] * LR_p5, p5)
        ]
        opt.apply_gradients(gradients_and_vars)
        
        iter_time = time.time() - iter_start
        time_hrs = time_seconds.numpy() / 3600.0
        
        # Calculate Ce for printing
        Ce = 6.8 * np.exp(-45350. / (R_GAS * TEMP))
        
        # Detailed logging for every iteration
        print(f"\n--- Iteration {ii+1}/{N_ITERATIONS} ---")
        print(f"  Loss (Total): {total_loss.numpy():.6f} | MSE: {basic_loss.numpy():.6f} | Reg: {reg_loss.numpy():.6f}")
        print(f"  Parameters: p1={p1.numpy():.6f}, p2={p2.numpy():.6f}, p3={p3.numpy():.6f}, p4={p4.numpy():.6f}, p5={p5.numpy():.6f}")
        print(f"  Gradients: dp1={gradients[0].numpy():.6e}, dp2={gradients[1].numpy():.6e}, dp3={gradients[2].numpy():.6e}, dp4={gradients[3].numpy():.6e}, dp5={gradients[4].numpy():.6e}")
        print(f"  Time: {iter_time:.3f}s")
        print(f"\n  All Time Steps:")
        for idx in range(len(time_hrs)):
            cbar_minus_ce = pred_cbar.numpy()[idx] - Ce
            print(f"    t={time_hrs[idx]:8.3f}h: ND={pred_nd.numpy()[idx]:.3e} m⁻³, PR={pred_pr.numpy()[idx]*1e9:7.2f} nm, HV={pred_hv.numpy()[idx]:6.2f}, Ci={pred_ci.numpy()[idx]:.4f}, C_bar={pred_cbar.numpy()[idx]:.4f}, C_bar-Ce={cbar_minus_ce:.4f}")
        
        # Save plots at every iteration
        save_iteration_plots(mc, ii, time_seconds.numpy(), pred_hv.numpy(), 
                           pred_nd.numpy(), pred_pr.numpy(), output_folder)
        
        history_loss.append(total_loss.numpy())
        history_basic_loss.append(basic_loss.numpy())
        history_reg_loss.append(reg_loss.numpy())
        history_p1.append(p1.numpy())
        history_p2.append(p2.numpy())
        history_p3.append(p3.numpy())
        history_p4.append(p4.numpy())
        history_p5.append(p5.numpy())

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    iterations = range(1, len(history_loss) + 1)
    ax1.plot(iterations, history_loss, 'b-', linewidth=2, label='Total Loss')
    ax1.plot(iterations, history_basic_loss, 'r--', linewidth=2, label='MSE Loss')
    ax1.plot(iterations, history_reg_loss, 'g--', linewidth=2, label='Reg Loss')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Loss History', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(iterations, history_p1, 'r-', label='$P_1$', linewidth=2)
    ax2.plot(iterations, history_p2, 'g-', label='$P_2$', linewidth=2)
    ax2.plot(iterations, history_p3, 'b-', label='$P_3$', linewidth=2)
    ax2.plot(iterations, history_p4, 'm-', label='$P_4$', linewidth=2)
    ax2.plot(iterations, history_p5, 'c-', label='$P_5$', linewidth=2)
    ax2.set_xlabel('Iteration', fontsize=12)
    ax2.set_ylabel('Parameter Value', fontsize=12)
    ax2.set_title(f'Parameter History', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f'mc{mc}_training_history.png'), dpi=150)
    plt.close()

    # Save training history to Excel
    history_data = {
        'Iteration': range(1, len(history_loss) + 1),
        'Total_Loss': history_loss,
        'Basic_Loss': history_basic_loss,
        'Reg_Loss': history_reg_loss,
        'p1_gamma': history_p1,
        'p2_nucleation': history_p2,
        'p3_growth': history_p3,
        'p4_coarsening': history_p4,
        'p5_strength': history_p5
    }
    df = pd.DataFrame(history_data)
    excel_filename = os.path.join(output_folder, f'mc{mc}_training_history.xlsx')
    df.to_excel(excel_filename, index=False)
    print(f"Training history saved to: {excel_filename}")

    print(f"\n{'='*60}")
    print(f"Finished MC Run {mc}")
    print(f"Final Parameters: p1={p1.numpy():.6f}, p2={p2.numpy():.6f}, p3={p3.numpy():.6f}, p4={p4.numpy():.6f}, p5={p5.numpy():.6f}")
    print(f"Final Loss: {total_loss.numpy():.6f}")
    print(f"{'='*60}")