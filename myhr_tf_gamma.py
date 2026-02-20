import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
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
exp_hv_time = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 100.0]) * 3600.0 
exp_hv_val  = np.array([55, 80, 100, 115, 110, 85])

# Interpolation for Hardness (Target)
pchip_hv = PchipInterpolator(np.log(exp_hv_time/3600.0), exp_hv_val)

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
N_STEPS = 300

# Time Grid
time_seconds = tf.pow(10.0, tf.linspace(tf.math.log(10.0)/tf.math.log(10.0), tf.math.log(300000.0)/tf.math.log(10.0), N_STEPS))
dt_seconds = tf.concat([[0.0], time_seconds[1:] - time_seconds[:-1]], axis=0)
target_hv_interp = tf.constant(pchip_hv(np.log(time_seconds.numpy()/3600.0)), dtype=tf.float32)

# ==========================================
# 1. PHYSICS MODEL FUNCTION
# ==========================================

def run_physics_simulation(params):
    """
    params: 
    p1: Gamma multiplier (General)
    p2: Growth kinetics multiplier
    p3: Coarsening multiplier
    p4: Strength (M-factor) multiplier
    """
    p1, p2, p3, p4 = params
    
    # Dependent Constants
    D = 2.5e-4 * tf.math.exp(-130000. / (R_GAS * TEMP))
    Ce = 6.8 * tf.math.exp(-45350. / (R_GAS * TEMP))
    PI = tf.constant(np.pi, dtype=tf.float32)
    
    # Initial state
    current_ND = tf.constant(1.0e12, dtype=tf.float32)
    current_PR = tf.constant(0.5e-9, dtype=tf.float32)
    current_C_bar = tf.constant(C_TOT_MG, dtype=tf.float32)
    
    HV_hist = tf.TensorArray(tf.float32, size=N_STEPS)

    for i in tf.range(N_STEPS):
        dt = dt_seconds[i]
        
        # --- A. NUCLEATION ---
        supersat = tf.nn.relu(current_C_bar - Ce)
        ln_S = tf.math.log((supersat + Ce) / Ce)
        # Using p1 to modify the energy barrier via Gamma
        barrier = (30000.0 / (R_GAS * TEMP))**3 * (1.0 / (ln_S**2 + 1e-9))
        nucleation_rate = 1.8e35 * tf.math.exp(-barrier * tf.pow(p1, 3)) * tf.math.exp(-130000./(R_GAS*TEMP))
        
        dN = nucleation_rate * dt * tf.math.sigmoid((current_C_bar - Ce) * 1e5)
        current_ND += dN
        
        # --- B. GROWTH & COARSENING ---
        # Interfacial concentration modified by p1 (Gamma)
        Ci = Ce * tf.math.exp((2 * (p1 * GAMMA_BASE) * VM) / (R_GAS * TEMP * (current_PR + 1e-12)))
        
        # Growth velocity modified by p2
        rate_growth = p2 * ((current_C_bar - Ci) / (CP - Ci)) * (D / (current_PR + 1e-12))
        
        # Coarsening velocity modified by p3 and p1 (Gamma)
        k_coarse = (8 * (p1 * GAMMA_BASE) * VM * D * Ce) / (9 * R_GAS * TEMP)
        rate_coarse = p3 * (k_coarse / (tf.square(current_PR) + 1e-12))
        
        is_growing = tf.math.sigmoid((current_C_bar - Ci) * 1e5)
        dr_dt = (is_growing * rate_growth) + ((1.0 - is_growing) * rate_coarse)
        
        current_PR = tf.nn.relu(current_PR + dr_dt * dt - 5e-10) + 5e-10

        # Mass Balance
        vol_frac = current_ND * (4.0/3.0) * PI * tf.pow(current_PR, 3.0)
        current_C_bar = tf.maximum(C_TOT_MG - (CP * vol_frac), Ce)

        # --- C. STRENGTH ---
        sigma_ss = 29.0e6 * tf.pow(current_C_bar, 2/3) + 66.3e6 * tf.pow(current_C_bar * 1.5, 2/3)
        
        # M-factor modified by p4
        sig_cut = p4 * 2 * 0.46 * G_MOD * (B_VEC / RC) * tf.sqrt(3 * vol_frac / (2 * PI)) * (current_PR / RC)
        sig_bypass = p4 * 2 * 0.46 * G_MOD * (B_VEC / current_PR) * tf.sqrt(3 * vol_frac / (2 * PI))
        
        is_bypass = tf.math.sigmoid((current_PR - RC) * 1e9)
        sigma_p = (1.0 - is_bypass) * sig_cut + is_bypass * sig_bypass
        
        sigma_y = 10.0e6 + sigma_ss + sigma_p
        hv = 0.33 * (sigma_y / 1e6) + 16.0
        
        HV_hist = HV_hist.write(i, hv)

    return HV_hist.stack()

# ==========================================
# 2. MONTE CARLO OPTIMIZATION LOOP
# ==========================================

N_MC_RUNS = 2
N_ITERATIONS = 400
LR_ADAM = 0.01

for mc in range(1, N_MC_RUNS + 1):
    print(f"\n=== Monte Carlo Run {mc} ===")
    tf.random.set_seed(mc)
    
    # Initialize parameters near unity (p4/M-factor near 3.0)
    p1 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.1), name="p_gamma")
    p2 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.1), name="p_growth")
    p3 = tf.Variable(tf.random.normal([], mean=1.0, stddev=0.1), name="p_coarse")
    p4 = tf.Variable(tf.random.normal([], mean=3.0, stddev=0.2), name="p_M")
    
    params_list = [p1, p2, p3, p4]
    opt = tf.keras.optimizers.Adam(learning_rate=LR_ADAM)
    
    history_loss = []

    for ii in range(N_ITERATIONS):
        with tf.GradientTape() as tape:
            pred_hv = run_physics_simulation(params_list)
            
            # Loss: Mean Absolute Error + L2 Regularization (Unity constraints)
            basic_loss = tf.reduce_mean(tf.abs(pred_hv - target_hv_interp))
            reg_loss = 0.01 * (tf.nn.l2_loss(p1 - 1.0) + tf.nn.l2_loss(p2 - 1.0) + tf.nn.l2_loss(p3 - 1.0))
            total_loss = basic_loss + reg_loss

        gradients = tape.gradient(total_loss, params_list)
        
        # Check for None gradients to ensure stability
        if any(g is None for g in gradients):
            print(f"Warning: None gradient detected at iteration {ii}")
            continue
            
        opt.apply_gradients(zip(gradients, params_list))
        
        if ii % 50 == 0:
            p_vals = [f"{v.name.split(':')[0]}: {v.numpy():.4f}" for v in params_list]
            print(f"Iter {ii:03d} | Loss: {total_loss.numpy():.4f} | {' | '.join(p_vals)}")

    print(f"Finished MC {mc}. Final p1: {p1.numpy():.4f}, Final p4: {p4.numpy():.4f}")