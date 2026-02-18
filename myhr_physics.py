import numpy as np
import matplotlib.pyplot as plt
import math
import os
import pandas as pd

# ==========================================
# 0. SETUP OUTPUT FOLDER
# ==========================================
output_folder = "plots_myhr_corrected"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# ==========================================
# 1. USER PROVIDED DATA (High Accuracy)
# ==========================================

# -- Number Density (m^-3) --
exp_den_time = np.array([
    1810.1171694409, 3618.84294924468, 14300.9117041412, 
    29919.5858793053, 59140.7079792916, 270893.777674506
])
exp_den_val = np.array([
    3.2475141905607E+22, 3.2475141905607E+22, 2.56590717051335E+22,
    2.15443489821486E+22, 2.3245381353485E+22, 1.18197192102842E+21
])

# -- Mean Radius (Angstroms) --
exp_rad_time = exp_den_time 
exp_rad_val = np.array([
    32.5384457371685, 39.3587620081145, 46.5338086013818,
    49.8333935377973, 48.7083462919317, 131.025382908724
])

# -- Hardness (HV) --
exp_hv_time = np.array([0.1, 0.5, 1.0, 3.0, 10.0, 100.0]) * 3600.0 
exp_hv_val  = np.array([55, 80, 100, 115, 110, 85]) 

# ==========================================
# 2. PARAMETERS & CONSTANTS
# ==========================================

# -- Process --
T_celsius = 185.
T = T_celsius + 273.15 

# -- Time Settings --
t_start_s = 10.0          
t_end_s = 300000.0 
n_steps = 500 # Increased steps for smoother nucleation curve

# -- Physical Constants --
R_gas = 8.314           
pi = np.pi

# -- Alloy 4 Composition --
C_total_Mg = 0.55       
C_total_Si = 0.82       

# -- Precipitate (Mg5Si3) --
vm = 7.62e-5            
Cp = 59.0               
gamma = 0.15           # Slightly lowered surface energy to assist nucleation onset
Qd = 130000.            
D0 = 2.2e-4             
Qs = 45350.             
Cs = 6.8                

# -- Nucleation (CALIBRATED) --
A0 = 27000.0             # Adjusted Barrier Parameter (lowered from 18000 to match scaling)
j0 = 1.8e35         

# Strength Model
M = 3.1                 
beta = 0.46             # INCREASED: Raises the peak hardness to ~115 HV
b = 2.84e-10            
G = 2.7e10              
rc = 5.0e-9             # Critical radius (50 Angstroms). Crossing this triggers the drop.
k_Mg = 29.0e6           
k_Si = 66.3e6           
sigma_i = 10.0e6

# ==========================================
# 3. INITIALIZATION
# ==========================================

time_seconds = np.logspace(np.log10(t_start_s), np.log10(t_end_s), n_steps)
dt_seconds = np.diff(np.concatenate(([0], time_seconds)))

ND_history = []  
PR_history = []  
HV_history = []  
Yield_history = [] 
C_bar_history = []

current_ND = 1.0e12       # Non-zero start to avoid log errors
current_PR = 0.5e-9       
current_C_bar = C_total_Mg 

# ==========================================
# 4. PHYSICS FUNCTIONS
# ==========================================

def calc_diffusion(T_kelvin):
    return D0 * math.exp(-Qd / (R_gas * T_kelvin))

def calc_Ce(T_kelvin):
    return Cs * math.exp(-Qs / (R_gas * T_kelvin))

# --- THE CORRECTED STRENGTH FUNCTION ---
def calc_strength(r, N, C_mg, C_si):
    sigma_ss = k_Mg * (max(C_mg, 0)**(2/3)) + k_Si * (max(C_si, 0)**(2/3)) #discontinuity
    
    if N > 1e10 and r > 0: #discontinuity
        f = N * (4/3) * pi * (r**3)
        if r < rc: #discontinuity
            # Shearing (Rise to peak)
            sigma_p = M * 2 * beta * G * (b / rc) * math.sqrt(3 * f / (2 * pi)) * (r / rc)
        else: #discontinuity
            # Bypassing (Drop after peak)
            sigma_p = M * 2 * beta * G * (b / r) * math.sqrt(3 * f / (2 * pi))
    else: #discontinuity
        sigma_p = 0.0
        
    sigma_y = sigma_i + sigma_ss + sigma_p
    HV = 0.33 * (sigma_y / 1e6) + 16.0
    return HV, sigma_y / 1e6

# ==========================================
# 5. MAIN LOOP
# ==========================================

D = calc_diffusion(T)
Ce = calc_Ce(T)

for i in range(n_steps):
    dt = dt_seconds[i]
    
    # --- A. NUCLEATION ---
    if current_C_bar > Ce: #discontinuity
        # Calculate Driving Force (Supersaturation)
        ln_S = math.log(current_C_bar / Ce)
        
        # Energy Barrier Term
        # Note: A0 must be scaled correctly. If using (A0/RT)^3 form:
        barrier = (A0 / (R_gas * T))**3 * (1.0 / (ln_S**2))
        
        # Rate Calculation
        nucleation_rate = j0 * math.exp(-barrier) * math.exp(- Qd / (R_gas * T))
    else:
        nucleation_rate = 0.0
    
    # Update Density
    dN = nucleation_rate * dt
    current_ND += dN
    
    # Soft Cap to prevent runaway density before coarsening logic is added
    # (Optional, but realistic for single-class models)
    # if current_ND > 5e22: current_ND = 5e22

    # --- B. GROWTH ---
    if current_PR > 0: #discontinuity
        Ci = Ce * math.exp((2 * gamma * vm) / (R_gas * T * current_PR))
        
        if current_C_bar > Ci: #discontinuity
            # Growth
            dr_dt = ((current_C_bar - Ci) / (Cp - Ci)) * (D / current_PR)
        else: #discontinuity
            # Dissolution / Coarsening approximation for single class
            # When matrix is depleted, small particles dissolve. 
            # In single class, this is harder to model, so we assume slow Ostwald ripening
            # coarsening rate ~ k / r^2
            k_coarse = (8 * gamma * vm * D * Ce) / (9 * R_gas * T) # LSW theory
            dr_dt = k_coarse / (current_PR**2)

        current_PR += dr_dt * dt
        if current_PR < 5e-10: current_PR = 5e-10
    
    # --- C. MASS BALANCE & COARSENING DENSITY DROP ---
    vol_frac = current_ND * (4.0/3.0) * pi * (current_PR**3)
    current_C_bar = C_total_Mg - (Cp * vol_frac)
    
    # Simple Coarsening Logic for Number Density
    # If Volume Fraction is constant (Mass balance limit reached), N must drop as r increases
    # N(t) * r(t)^3 = constant
    if current_C_bar <= Ce * 1.05 and dr_dt > 0: #discontinuity
        # We are in coarsening regime
        # Recalculate N based on available volume fraction
        max_vol_frac = (C_total_Mg - Ce) / Cp
        current_ND = max_vol_frac / ((4.0/3.0) * pi * (current_PR**3))

    if current_C_bar < Ce: current_C_bar = Ce #discontinuity

    # --- D. PROPERTIES ---
    current_C_Si = C_total_Si * (current_C_bar / C_total_Mg)
    hv_val, yield_val = calc_strength(current_PR, current_ND, current_C_bar, current_C_Si)
    
    ND_history.append(current_ND)
    PR_history.append(current_PR * 1e10) # Angstroms
    HV_history.append(hv_val)
    Yield_history.append(yield_val)
    C_bar_history.append(current_C_bar)

# ==========================================
# 6. EXPORT
# ==========================================

df = pd.DataFrame({
    'Time (s)': time_seconds,
    'Time (h)': time_seconds / 3600.0,
    'Number Density (m^-3)': ND_history,
    'Mean Radius (Angstroms)': PR_history,
    'Hardness (HV)': HV_history,
    'Yield Strength (MPa)': Yield_history
})

excel_path = os.path.join(output_folder, "myhr_results_corrected.xlsx")
try:
    df.to_excel(excel_path, index=False)
    print(f"Data exported to: {excel_path}")
except:
    csv_path = os.path.join(output_folder, "myhr_results_corrected.csv")
    df.to_csv(csv_path, index=False)
    print(f"Data exported to: {csv_path}")

# ==========================================
# 7. PLOTTING
# ==========================================

fig, axs = plt.subplots(3, 1, figsize=(10, 16), sharex=True)

# --- Plot 1: Number Density ---
# Load TND_processed data for error bars
# Load TND_processed data for error bars
tnd_file_path = "TND_processed.xlsx"  # Assuming the file is in the current directory

tnd_df = pd.read_excel(tnd_file_path)
# Extract high bars (indices where i%3==0) and low bars (indices where i%3==1)
high_bars = []
low_bars = []
exp_den_val = []
for i in range(len(tnd_df)):
    if i % 3 == 0:
        # Use experimental density values for indices where i%3==0
        exp_den_val.append(tnd_df.iloc[i]['y'])
    if i % 3 == 1:
        high_bars.append(tnd_df.iloc[i]['y'])
    elif i % 3 == 2:
        low_bars.append(tnd_df.iloc[i]['y'])

# Calculate error bars (difference from experimental values)
yerr_upper = np.array(high_bars) - exp_den_val
yerr_lower = exp_den_val - np.array(low_bars)
yerr = [yerr_lower, yerr_upper]

axs[0].plot(time_seconds, ND_history, 'b-', linewidth=2, label='Model Density')
axs[0].errorbar(exp_den_time, exp_den_val, yerr=yerr, fmt='bs', markersize=8, markerfacecolor='none', markeredgewidth=2, capsize=5, capthick=2, label='Exp. Data')

axs[0].set_ylabel(r'Number Density ($m^{-3}$)')
axs[0].set_title('Evolution of Microstructure (Alloy 4 @ 185°C)')
axs[0].grid(True, which="both", ls="-")
axs[0].set_yscale('log')
axs[0].set_ylim(1e19, 1e24) 
axs[0].legend(loc='upper right')

# --- Plot 2: Mean Radius (Angstroms) ---
axs[1].plot(time_seconds, PR_history, 'g-', linewidth=2, label='Model Radius')
axs[1].plot(exp_rad_time, exp_rad_val, 'go', markersize=8, markerfacecolor='none', markeredgewidth=2, label='Exp. Data')
axs[1].set_ylabel(r'Mean Radius ($\mathring{A}$)')
axs[1].grid(True, which="both", ls="-")
axs[1].legend(loc='upper left')

# --- Plot 3: Hardness ---
axs[2].plot(time_seconds, HV_history, 'r-', linewidth=2, label='Model Hardness')
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