# comparing adam and powell performance with respect to iterations and time.

import time
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import re
import tensorflow as tf
import numpy as np
import json
# import kawin.Thermodynamics
from kawin.thermo import MulticomponentThermodynamics
from kawin.thermo import MulticomponentThermodynamics
from plot_utils import visual
phases = ['FCC_A1', 'MG5SI6_B_DP']
therm = MulticomponentThermodynamics('AlMgSi.tdb', ['AL', 'MG', 'SI'], phases, drivingForceMethod='sampling') #'approximate', 'sampling'
from scipy.optimize import minimize

plt.rcParams.update({
    'font.size': 15,       # Global font size
    'axes.titlesize': 15,  # Font size for titles
    'axes.labelsize': 15,  # Font size for x and y labels
    'xtick.labelsize': 15, # Font size for x-tick labels
    'ytick.labelsize': 15, # Font size for y-tick labels
    'legend.fontsize': 15, # Font size for legend
    'figure.titlesize': 15 # Font size for figure title
})

newFolder = os.path.join('D:\PhD\Implementation', 'AI')
if not os.path.exists(newFolder): os.makedirs(newFolder)

#composition and process parameters
T = 150.+273
FinalTime = 170. #(h)

#physical constanst
K = 1.380649e-23 #+ (j.K-1)
Delta = 7.
R = 8.31  #(j.K-1.mol-1)
pi = tf.constant(np.pi, dtype=tf.float32)
e = tf.constant(tf.math.exp(1.0), dtype=tf.float32)
S1 = 5.
S2 = 6.
xp_Mg = 5./11.
xp_Si = 6./11.
v = 1.92e-29 #+ #average volume of atoms in MgxSiy precipitates[1]
a = 2.86e-10 #+ interatomic distance[software] 
N0 = 5.8e28  #+ Number of sites [s] 5.8e28 #1.5e27
vm = 3.95e-5  #+ ppt molar volum
epsilon =  1.
Gmod = 2.7e10  #+ #Gmod is the sheer modulus of aluminium matrix (GPa) (N.m-2)
b_dis = 2.84e-10 #+ should be in (m)

#fitting
q = 2. #stress exponent [s]
DELTA_DIS =  0.5 #[1]
k_Mg = 21.1e6   #[1] 21.1e6 Pa/wt%. [4] 29.0e6
k_Si = 10.5e6  #[1] 10.5e6  Pa/wt% [4] 66.3
Sigma0 = 55000000  #+
c02 = 500 #+ #[s] #(c02-Sigma) may give a negative value, which is invalid for the elongation equation.
c03 = 10 #+ #[s]

#modelling parameter
seed_value = 16
tf.random.set_seed(seed_value)
N_optimizer = 2000
dt = 1. #(h)
LR_param1 = 0.1
LR_param2 = 0.001
LR_param3 = 0.1
LR_param4 = 0.01
LR_param5 = 0.1
LR_paramrpc = 0.0000001
LR_SGD = 0.01
LR_Adam = 0.001
ftol_adam = 0.0000001 #adam: 0.001 in 15min, , 0.0001 for 18 min still no convergence, 0.00001 for good view #0.000001 40mins for best view 
ftol_powell= 0.0001 # 0.1 for 40 mins


param1_init = tf.random.normal([], mean=1.0, stddev=0.1)
initial_p2 = tf.math.log(tf.random.normal([], mean=1.0, stddev=0.01))
param2_init = initial_p2
param3_init = tf.random.normal([], mean=1.0, stddev=0.1)
param4_init = tf.random.normal([], mean=1.0, stddev=0.1)
param5_init = tf.random.normal([], mean=1.0, stddev=0.1)

time_steps = int(FinalTime/dt)
time_ = np.arange(0, time_steps * dt, dt)

Fk_coeff_1 = 2.*DELTA_DIS*Gmod*b_dis**2. #Delata_dis is a parameter depending on te shape and nature of dislocations
Fk_coeff_2 = math.sqrt(2)*DELTA_DIS*Gmod*b_dis
Gamma_dis = 2./math.sqrt(3)*DELTA_DIS*Gmod*b_dis**2. #calculated from rp=rpc condition #Gamma is the line tension of dislocations 
Gamma_b_coeff = math.sqrt(2.*math.sqrt(3)*Gamma_dis)*b_dis

# #data processsing
x = np.array([1, 2, 3, 8, 24, 48, 72, 120, 168])
y = np.array([74, 89, 127, 187, 219, 230, 252, 280, 255])
y = y/1.e3 #normalisation
coefficients = np.polyfit(x, y, 2)
Yield_interpolated = np.polyval(coefficients, time_[1:])
Yield_interpolated = tf.Variable(Yield_interpolated, trainable=False, dtype=np.float32)


def D_Mg():
    D0_Mg = 2.2e-4  #[2] 6.23e-6 (m2.s-1)  # [3] 2.2×10−4
    # D0_Mg = 2.2e-4
    Q_Mg = 115000  # 115000[2] (j.mol-1) activation energy [3] 130000
    # D_Mg_ = D0_Mg * math.exp(-Q_Mg/(R*T))
    D_Mg_ = D0_Mg * (e** (-Q_Mg/(R*T)))
    return D_Mg_ #(m2.s-1)
D_Mg_ = D_Mg()

def D_Si():
    # T should be in Kelvin
    D0_Si = 2.2e-4  #[2] 2.48e-4 (m2.s-1) # [3] 2.2×10−4  
    Q_Si = 115000  # 137000 [2] (j.mol-1) activation energy [3] 130000
    # D_Si_ = D0_Si * math.exp(-Q_Si/(R*T))
    D_Si_ = D0_Si * (e** (-Q_Si/(R*T)))
    return D_Si_
D_Si_ = D_Si()

def dGvol(xm_Mg, xm_Si):
    dG_ = therm._getDrivingForceApprox([xm_Mg, xm_Si], T, precPhase='MG5SI6_B_DP', returnComp=True)[0]
    # dG = -dG_.item()
    dG = -dG_
    dGvol_ = dG/vm
    return dGvol_ #j.m-3

def Rs(dGvol_, Gamma_, param1_):
    Rs_ =  param1_*(-2.* Gamma_/dGvol_) * (2.*Delta/(3.*Delta-1.)) + sigmoid(10*dGvol_)
    return Rs_ #(m)

def Z():
    Z_ = 1.
    return Z_

def Beta():
    # x = D_Mg()
    # y = D_Si()
    # Beta_ = (4. * pi* Delta * Rs()**2.) * ((xp_Mg/(xm_Mg*D_Mg()) + xp_Si/(xm_Si*D_Si()))-1.) / a**4.
    Beta_ = 1.
    return Beta_

def sigmoid(x):
    # Use a stable implementation to avoid overflow
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)


def DeltaGsnorm(dGvol_, Gamma_):
    shape_factor = (4.*Delta**3.)/(3.*Delta-1.)**2.
    DeltaGsnorm__ =   (16.*pi*Gamma_**3.)/(3.*dGvol_**2.) * (1.e20)**sigmoid(dGvol_)  *  shape_factor  /  (K*T)
    return DeltaGsnorm__

def dNdT(Z__, Beta__, DeltaGsnorm__, param2_):
    param2_exp = tf.exp(param2_)
    # dNdT__ =  N0 * Z__ * Beta__ * e**(- (param2_exp) * DeltaGsnorm__) #(s-1) 
    dNdT__ = N0 * Z__ * Beta__ * e**(- param2_exp * DeltaGsnorm__)/(1 + (param2_exp * DeltaGsnorm__))
    return dNdT__

def dNdT_nograd(Z__, Beta__, DeltaGsnorm__, param2_):
    # dNdT__ =  N0 * Z__ * Beta__ * e**(- (param2_exp) * DeltaGsnorm__) #(s-1) 
    # dNdT__ = N0 * Z__ * Beta__ * e**(- param2 * DeltaGsnorm__)
    dNdT__ = N0 * Z__ * Beta__ * e**(- param2 * DeltaGsnorm__)/(1 + (param2 * DeltaGsnorm__))
    return dNdT__

def Rp(Rs__, Gamma_, param3_):
    Rp__ =param3_* ( Rs__ + (K*T/(pi*Delta*Gamma_)**0.5)/2. )
    return Rp__

def CalculateNucleation(KWNcounter, dNdT_, dt, Rp_, ND, PR): #after converting to Decimal, this function doesn't need to know KWNcounter
    dt = tf.Variable(float(dt), trainable = False, dtype=np.float32)
    ND = tf.tensor_scatter_nd_update(ND, [[KWNcounter-1]], [dNdT_*dt*3600.], name=None)
    PR = tf.tensor_scatter_nd_update(PR, [[KWNcounter-1]], [Rp_], name=None)
    return ND, PR

def CalculateVelocity(radius, xm_Mg_, xm_Si_, Rs__, Gamma_, coars_coeff_, param4_, param5_):

    d2Gx_Mg = (R * T) / D_Mg_
    a11 = (xp_Mg-xm_Mg_)**2. / xm_Mg_ * d2Gx_Mg
    a12 = xp_Mg * d2Gx_Mg

    d2Gx_Si = (R * T) / D_Si_
    a21 = (xp_Si-xm_Si_)**2. / xm_Si_ * d2Gx_Si
    a22 = xp_Si * d2Gx_Si

    b5 = 1./(a11 + a21)
    b10 = 1./(a12 + a22)
    ksi = 1.0
    b6 = 4.*Gamma_*Delta*vm/(ksi*radius*(3.*Delta-1.)*R * T)
    b7 = 1./Rs__ - 1./radius 

    vel_growth = param4_*b5*b6*b7
    vel_coarse = param5_*b10*b6*coars_coeff_/radius
    vel_total = vel_growth + vel_coarse
    return vel_total, vel_growth, vel_coarse

def CalculateGrowth(KWNcounter_, dt_, xm_Mg_, xm_Si_, Rs__, ND, PR, Gamma_, coars_coeff_, param4_, param5_):
    
    for i in range(1): 
        vel_total, vel_growth, vel_coarse = CalculateVelocity(PR[i], xm_Mg_, xm_Si_, Rs__, Gamma_, coars_coeff_, param4_, param5_)
        rad_growth = PR[i] + vel_growth*dt_*3600. #the radius that is only resultant from growth (not coarsening)
        rad_total = PR[i] + vel_total*dt_*3600.

        PR = tf.tensor_scatter_nd_update(PR, [[i]], [rad_total])
        ND = tf.tensor_scatter_nd_update(ND, [[i]], [ND[i]*(rad_growth/rad_total)**3.]) 

    return ND, PR

def Update(KWNcounter, xm_Mg0_, xm_Si0_, ND, PR, VF, TVF_t, TND_t, MPR_t):
    TVF = 0.
    TND = 0.
    MPR = 0.
    for i in range(1):

        VF = tf.tensor_scatter_nd_update(VF, [[i]], [(2.*Delta-2./3.)*pi*PR[i]**3.*ND[i]]) 

        TVF = TVF + (2.*Delta-2./3.)*pi*PR[i]**3.*ND[i] 
        TND = TND + ND[i] 
        MPR = MPR + ND[i]*PR[i] 
    
    TND_t = tf.tensor_scatter_nd_update(TND_t, [[KWNcounter-1]], [TND])
    MPR_t = tf.tensor_scatter_nd_update(MPR_t, [[KWNcounter-1]], [MPR/(TND+1.)])
    TVF_t = tf.tensor_scatter_nd_update(TVF_t, [[KWNcounter-1]], [TVF])

    xm_Mg = (xm_Mg0_-epsilon*TVF*xp_Mg)/(1.-epsilon*TVF)
    xm_Si = (xm_Si0_-epsilon*TVF*xp_Si)/(1.-epsilon*TVF)
    xm_Mg = tf.where(xm_Mg > 0, xm_Mg, tf.zeros_like(xm_Mg))
    xm_Si = tf.where(xm_Si > 0, xm_Si, tf.zeros_like(xm_Si))
    xm_Al = 1. - xm_Mg - xm_Si

    return xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t, MPR_t

def Strength(KWNcounter, ND, PR, rpc_, xm_Mg, xm_Si, Mppt, Yield_t_, Tau_c_t, Sigma_ppt_t): #plan B: simply TVF
    summation_nlp_1 = 0. #SM
    summation_nlp_2 = 0. #SM
    TND_weak = 1. #formerly summation_n #to avoid float devision, set to 1. #SM
    summation_nf = 0. #SM

    for i in range(1):
        #SM
        if (PR[i] > rpc_): # strong particles
            summation_nlp_2 = summation_nlp_2 + 2.*Delta*PR[i]*ND[i]

        else: # weak particles
            Fk = Fk_coeff_1*PR[i]/rpc_ #(N)
            summation_nlp_1 = summation_nlp_1 + ND[i]*2.*Delta*PR[i]
            TND_weak = TND_weak + ND[i]
            summation_nf = summation_nf + ND[i]*2.*DELTA_DIS*Gmod*(b_dis**2.)*PR[i]/rpc_

    
    #SM
    Tau_c_Strong = Fk_coeff_2*tf.sqrt(summation_nlp_2)
    Tau_c_Weak = (tf.sqrt(summation_nlp_1)/Gamma_b_coeff) * (summation_nf/TND_weak)**1.5  
    Tau_c_t = tf.tensor_scatter_nd_update(Tau_c_t, [[KWNcounter-1]], [[Tau_c_Weak, Tau_c_Strong]])
    Sigma_ppt = Mppt*tf.sqrt(tf.pow(Tau_c_Strong, q) + tf.pow(Tau_c_Weak, q))
    Sigma_ss =  k_Mg*xm_Mg + k_Si*xm_Si
    # Sigma_ppt_t.append(Sigma_ppt)
    Sigma_ppt_t = tf.tensor_scatter_nd_update(Sigma_ppt_t, [[KWNcounter-1]], [Sigma_ppt])

    Sigma = Sigma0 + Sigma_ppt + Sigma_ss
    Yield_t_ = tf.tensor_scatter_nd_update(Yield_t_, [[KWNcounter-1]], [Sigma /1.e9])

    return Yield_t_

    # loss_function = tf.keras.losses.MeanSquaredError()
def loss_function(var1, var2):
    return tf.reduce_mean(tf.abs(var1 - var2))


















############################################       ADAM   ####################

opt = tf.keras.optimizers.Adam(learning_rate=LR_Adam)
loss_t = []
def physics_adam():

    rpc = paramrpc* 1.4e-9 #3. (nm) [s] #2.4e-9 (m) [1] 5e-9 [4]

    ND = tf.Variable(tf.zeros([time_steps]), trainable=False)
    PR = tf.Variable(tf.zeros([time_steps]), trainable=False)
    VF = tf.Variable(tf.zeros([time_steps]), trainable=False)
    TVF_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
    Sigma_ppt_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False, dtype=np.float32)
    MPR_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
    TND_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
    Yield_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False) # because the 0th time step doesn't have any associated YS prediction
    Tau_c_t = tf.Variable(tf.zeros([time_steps-1, 2]), trainable=False, dtype=tf.float32) #weak and strong

    Gamma =  0.039
    Mppt = 1.0   #[1] 3.1
    coars_coeff = 2.5e-3 #1e-3 #the smaller, the later the peak

    wtp_Mg = 0.5
    wtp_Si = 0.43
    mw_Mg = 24.305  #(g/mol)
    mw_Si = 28.09   #(g/mol)
    mw_Al = 26.98   #(g/mol)
    xm_denom = wtp_Mg/mw_Mg + wtp_Si/mw_Si + (100.-wtp_Mg-wtp_Si)/mw_Al
    xm_Mg0 = wtp_Mg/mw_Mg/xm_denom
    xm_Si0 = wtp_Si/mw_Si/xm_denom
    xm_Al0 = (100.-wtp_Mg-wtp_Si)/mw_Al/xm_denom
    xm_Mg = xm_Mg0
    xm_Si = xm_Si0
    xm_Al = xm_Al0
    
    # print(f'CalculateNucleation:')
    dGvol_ = dGvol(xm_Mg, xm_Si)
    Z_ = Z()
    Beta_ = Beta()
    Rs_ = Rs(dGvol_, Gamma, param1)
    DeltaGsnorm_ = DeltaGsnorm(dGvol_, Gamma)
    Rp_ = Rp(Rs_, Gamma, param3)
    dNdT_ = dNdT(Z_, Beta_, DeltaGsnorm_, param2)
    KWNcounter = 1
    ND, PR = CalculateNucleation(KWNcounter, dNdT_, dt, Rp_, ND, PR)

    # print(f'Delivered by CalculateNucleation:')
    # print(f'dGvol_: {dGvol_}')
    # print(f'Rs_: {Rs_}')
    # print(f'DeltaGsnorm_: {DeltaGsnorm_}')
    # print(f'Rp_: {Rp_}')
    # print(f'dNdT_: {dNdT_}')
    # print(f'ND: {ND}')
    # print(f'PR: {PR}')
    # print()


    for KWNcounter in range(1, time_steps):
        
        t = time_[KWNcounter]
        
        # print(f'==============================counter: {KWNcounter}======time: {t}==========')

        ND, PR = CalculateGrowth(KWNcounter, dt, xm_Mg, xm_Si, Rs_, ND, PR, Gamma, coars_coeff, param4, param5)

        # xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t,  MPR_t, Yield_t, Tau_c_t, Sigma_ppt_t, TND, MPR, TVF = CalculateVF(KWNcounter, xm_Mg0, xm_Si0, ND, PR, VF, Yield_t, TVF_t, TND_t, MPR_t, Mppt, Tau_c_t, Sigma_ppt_t, rpc)
        xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t, MPR_t = Update(KWNcounter, xm_Mg0, xm_Si0, ND, PR, VF, TVF_t, TND_t, MPR_t)
        Yield_t = Strength(KWNcounter, ND, PR, rpc, xm_Mg, xm_Si, Mppt, Yield_t, Tau_c_t, Sigma_ppt_t)

    lambda_l2 = 1e-4  # Adjust this factor to control the strength of regularization
    l2_reg = (
        tf.nn.l2_loss(param1- 1.0) +
        tf.nn.l2_loss(param2) +
        tf.nn.l2_loss(param3- 1.0) +
        tf.nn.l2_loss(param4- 1.0) +
        tf.nn.l2_loss(param5- 1.0)
    )

    loss_ = loss_function(Yield_t, Yield_interpolated) + lambda_l2 * l2_reg
    
    error_percentage = tf.reduce_mean(tf.abs((Yield_t - Yield_interpolated) / Yield_interpolated)) * 100
    
    loss_t.append(loss_.numpy())

    visual(time_, TND_t, MPR_t, TVF_t, Yield_t, loss_t, Yield_interpolated, x, y, N_optimizer, param1_t, param2_t, param3_t, param4_t, param5_t, ii, optimizer='adam')
    print(f'adam iteration {ii} finished. loss {loss_} e% {error_percentage}')
    print('\n')
    return loss_


param1 = tf.Variable(param1_init, trainable = True, dtype=np.float32) # 1.0
param2 = tf.Variable(param2_init, trainable = True, dtype=np.float32) # 1.12 0.1133
param3 = tf.Variable(param3_init, trainable = True, dtype=np.float32) # 1.05
param4 = tf.Variable(param4_init, trainable = True, dtype=np.float32) # 1.1
param5 = tf.Variable(param5_init, trainable = True, dtype=np.float32) # 1.07
paramrpc = tf.Variable(0.85, trainable = False, dtype=np.float32) # 0.85

param1_t = []
param2_t = []
param3_t = []
param4_t = []
param5_t = []
paramrpc_t = []

grad1_t = []
grad2_t = []
grad3_t = []
grad4_t = []
grad5_t = []

param1_t.append(param1.numpy())
param2_t.append(np.exp(initial_p2.numpy()))
param3_t.append(param3.numpy())
param4_t.append(param4.numpy())
param5_t.append(param5.numpy())
paramrpc_t.append(paramrpc.numpy())



tic = time.time()

for ii in range(N_optimizer):

    if ii==0:
        print("adam param1 @0:", param1)
        print("adam param2 @0:", param2)
        print("adam param3 @0:", param3)
        print("adam param4 @0:", param4)
        print("adam param5 @0:", param5)
        print("adam paramrpc @0:", paramrpc)

    with tf.GradientTape() as tape:
        loss_value = physics_adam()
    gradients = tape.gradient(loss_value, [param1, param2, param3, param4, param5]) #paramrpc

    # Adam
    gradients_and_vars = [
    (gradients[0] * LR_param1, param1),
    (gradients[1] * LR_param2, param2),
    (gradients[2] * LR_param3, param3),
    (gradients[3] * LR_param4, param4),
    (gradients[4] * LR_param5, param5)
    ]
    opt.apply_gradients(gradients_and_vars)

    # Commented out manual gradient descent with specific learning rates
    # param1.assign_sub(LR_param1 * gradients[0])
    # param2.assign_sub(LR_param2 * gradients[1])
    # param3.assign_sub(LR_param3 * gradients[2])
    # param4.assign_sub(LR_param4 * gradients[3])
    # param5.assign_sub(LR_param5 * gradients[4])
    # paramrpc.assign_sub(LR_paramrpc * gradients[5])

    param1_t.append(param1.numpy())
    param2_t.append(np.exp(param2.numpy()))
    param3_t.append(param3.numpy())
    param4_t.append(param4.numpy())
    param5_t.append(param5.numpy())
    paramrpc_t.append(paramrpc.numpy())

    grad1_t.append(gradients[0].numpy())
    grad2_t.append(gradients[1].numpy())
    grad3_t.append(gradients[2].numpy())
    grad4_t.append(gradients[3].numpy())
    grad5_t.append(gradients[4].numpy())

    print("param1:", param1_t[-1])
    print("param2:", param2_t[-1])
    print("param3:", param3_t[-1])
    print("param4:", param4_t[-1])
    print("param5:", param5_t[-1])
    
    # Check ftol convergence criteria
    if ii > 0:
        current_loss = loss_t[-1]
        previous_loss = loss_t[-2]
        loss_diff = abs(current_loss - previous_loss)
        tolerance = ftol_adam * max(abs(current_loss), abs(previous_loss), 1.0)
        print(f"At iteration {ii}: loss_diff={loss_diff:.6f} , tolerance={tolerance:.6f}")
        if loss_diff <= tolerance:
            print(f"ADAM Convergence reached.")
            break

toc = time.time()
adam_time = (toc-tic)/60.
adam_iterations = ii + 1
print(f'execution time of ADAM is {adam_time} mins')
print()

# Save Adam final convergence results before Powell starts
adam_final_params = {
    'param1': param1.numpy(),
    'param2': param2.numpy(), 
    'param3': param3.numpy(),
    'param4': param4.numpy(),
    'param5': param5.numpy()
}

# #===========================================    Powell  ========================================

# #modelling parameter
# N_optimizer_gd = 10
# loss_t = []
# nan_count = 0

# def physics_powell():

#     if not hasattr(physics_powell, 'counter'):
#         physics_powell.counter = 0

#     rpc = paramrpc* 1.4e-9 #3. (nm) [s] #2.4e-9 (m) [1] 5e-9 [4]

#     ND = tf.Variable(tf.zeros([time_steps]), trainable=False)
#     PR = tf.Variable(tf.zeros([time_steps]), trainable=False)
#     VF = tf.Variable(tf.zeros([time_steps]), trainable=False)
#     TVF_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
#     Sigma_ppt_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False, dtype=np.float32)
#     MPR_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
#     TND_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
#     Yield_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False) # because the 0th time step doesn't have any associated YS prediction
#     Tau_c_t = tf.Variable(tf.zeros([time_steps-1, 2]), trainable=False, dtype=tf.float32) #weak and strong

#     Gamma =  0.039
#     Mppt = 1.0   #[1] 3.1
#     coars_coeff = 2.5e-3 #1e-3 #the smaller, the later the peak

#     wtp_Mg = 0.5
#     wtp_Si = 0.43
#     mw_Mg = 24.305  #(g/mol)
#     mw_Si = 28.09   #(g/mol)
#     mw_Al = 26.98   #(g/mol)
#     xm_denom = wtp_Mg/mw_Mg + wtp_Si/mw_Si + (100.-wtp_Mg-wtp_Si)/mw_Al
#     xm_Mg0 = wtp_Mg/mw_Mg/xm_denom
#     xm_Si0 = wtp_Si/mw_Si/xm_denom
#     xm_Al0 = (100.-wtp_Mg-wtp_Si)/mw_Al/xm_denom
#     xm_Mg = xm_Mg0
#     xm_Si = xm_Si0
#     xm_Al = xm_Al0
    
#     # print(f'CalculateNucleation:')
#     dGvol_ = dGvol(xm_Mg, xm_Si)
#     Z_ = Z()
#     Beta_ = Beta()
#     Rs_ = Rs(dGvol_, Gamma, param1)
#     DeltaGsnorm_ = DeltaGsnorm(dGvol_, Gamma)
#     Rp_ = Rp(Rs_, Gamma, param3)
#     dNdT_ = dNdT_nograd(Z_, Beta_, DeltaGsnorm_, param2)
#     KWNcounter = 1
#     ND, PR = CalculateNucleation(KWNcounter, dNdT_, dt, Rp_, ND, PR)

#     # print("parameters:")
#     # print("param1:", param1_t[-1])
#     # print("param2:", param2_t[-1]) 
#     # print("param3:", param3_t[-1])
#     # print("param4:", param4_t[-1])
#     # print("param5:", param5_t[-1])

#     # print(f'Delivered by CalculateNucleation:')
#     # print(f'dGvol_: {dGvol_}')
#     # print(f'Rs_: {Rs_}')
#     # print(f'DeltaGsnorm_: {DeltaGsnorm_}')
#     # print(f'Rp_: {Rp_}')
#     # print(f'dNdT_: {dNdT_}')
#     # print(f'ND: {ND}')
#     # print(f'PR: {PR}')
#     # print()

#     for KWNcounter in range(1, time_steps):
        
#         t = time_[KWNcounter]
        
#         # print(f'==============================counter: {KWNcounter}======time: {t}==========')

#         ND, PR = CalculateGrowth(KWNcounter, dt, xm_Mg, xm_Si, Rs_, ND, PR, Gamma, coars_coeff, param4, param5)

#         # xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t,  MPR_t, Yield_t, Tau_c_t, Sigma_ppt_t, TND, MPR, TVF = CalculateVF(KWNcounter, xm_Mg0, xm_Si0, ND, PR, VF, Yield_t, TVF_t, TND_t, MPR_t, Mppt, Tau_c_t, Sigma_ppt_t, rpc)
#         xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t, MPR_t = Update(KWNcounter, xm_Mg0, xm_Si0, ND, PR, VF, TVF_t, TND_t, MPR_t)
#         Yield_t = Strength(KWNcounter, ND, PR, rpc, xm_Mg, xm_Si, Mppt, Yield_t, Tau_c_t, Sigma_ppt_t)

#     lambda_l2 = 1e-4 # 1e-6  # Adjust this factor to control the strength of regularization
#     l2_reg = (
#         tf.nn.l2_loss(param1- 1.0) +
#         tf.nn.l2_loss(param2) +
#         tf.nn.l2_loss(param3- 1.0) +
#         tf.nn.l2_loss(param4- 1.0) +
#         tf.nn.l2_loss(param5- 1.0)
#     )

#     loss_ = loss_function(Yield_t, Yield_interpolated) + lambda_l2 * l2_reg
    
#     # Count NaN values
#     global nan_count
#     if tf.math.is_nan(loss_):
#         nan_count += 1
    
#     loss_t.append(loss_.numpy())

#     # Store history
#     param1_t.append(param1.numpy())
#     param2_t.append(param2.numpy())
#     param3_t.append(param3.numpy())
#     param4_t.append(param4.numpy())
#     param5_t.append(param5.numpy())
#     paramrpc_t.append(paramrpc.numpy())
    
#     print(f"powell physics_counter {physics_powell.counter} with loss value {loss_} ")
#     print()

#     visual(time_, TND_t, MPR_t, TVF_t, Yield_t, loss_t, Yield_interpolated, x, y, N_optimizer_gd, param1_t, param2_t, param3_t, param4_t, param5_t, physics_powell.counter, optimizer='powell')
#     physics_powell.counter += 1
#     return loss_

# paramrpc = tf.Variable(0.85, trainable = False, dtype=np.float32) 
# param1 = tf.Variable(param1_init, trainable = True, dtype=np.float32)
# param2 = tf.Variable(tf.exp(param2_init), trainable = True, dtype=np.float32)
# param3 = tf.Variable(param3_init, trainable = True, dtype=np.float32)
# param4 = tf.Variable(param4_init, trainable = True, dtype=np.float32)
# param5 = tf.Variable(param5_init, trainable = True, dtype=np.float32)

# param1_t = []
# param2_t = []
# param3_t = []
# param4_t = []
# param5_t = []
# paramrpc_t = []

# param1_t.append(param1.numpy())
# param2_t.append(param2.numpy())
# param3_t.append(param3.numpy())
# param4_t.append(param4.numpy())
# param5_t.append(param5.numpy())
# paramrpc_t.append(paramrpc.numpy())

# tic = time.time()

# def objective_function(params):
#     # Unpack parameters
#     param1.assign(params[0])
#     param2.assign(params[1]) 
#     param3.assign(params[2])
#     param4.assign(params[3])
#     param5.assign(params[4])
    
#     loss_value = physics_powell()
#     return loss_value.numpy()

# # Initial parameter values
# initial_params = [param1.numpy(), param2.numpy(), param3.numpy(), param4.numpy(), param5.numpy()]

# # Create bounds for each parameter separately
# bounds = [
#     (0.0, 3.0),     # param1 bounds
#     (0.0, 3.0),     # param2 bounds 
#     (0.0, 3.0),     # param3 bounds
#     (0.0, 3.0),     # param4 bounds
#     (0.0, 3.0)      # param5 bounds
# ]
# # Set flag to track main optimization calls
# objective_function.is_main_call = True

# # Custom callback to track iterations
# def callback(xk):
#     objective_function.is_main_call = True
#     return False

# # Add iteration counter
# minimize_count = 0

# def callback(xk):
#     global minimize_count
#     minimize_count += 1
#     print(f"minimize_count {minimize_count} of {N_optimizer_gd}")
#     objective_function.is_main_call = True
#     return False

# result = minimize(objective_function, initial_params, method='Powell',
#                  bounds=bounds,
#                  callback=callback, 
#                  options={'maxiter': N_optimizer_gd,  # default: None
#                          'disp': True,            # default: False
#                          'ftol': ftol_powell,           # default: 1e-4, Function tolerance - smaller value for more precision
#                          'xtol': 1e-12,           # default: 1e-4, Parameter tolerance
#                          'maxfev': 10000,        # default: None, Maximum function evaluations
#                          'return_all': True       # default: False, Return optimization path
#                          })

# print("Parameters: ", result.x)

# print("Number of iterations for powell: ", result.nit)

# toc = time.time()
# powell_time = (toc-tic)/60.
# powell_iterations = result.nit
# print(f'execution time of Powell is {powell_time} mins')

# results_dir = './results'
# os.makedirs(results_dir, exist_ok=True)

# with open(os.path.join(results_dir, f'results_seed_{seed_value}.txt'), 'w') as f:
#     f.write(f"Random Seed: {seed_value}\n")
#     f.write(f"ADAM - Execution Time: {adam_time:.2f} mins, Function Evaluations: {adam_iterations}\n")
#     f.write(f"ADAM Final Parameters: param1={adam_final_params['param1']:.6f}, param2={np.exp(adam_final_params['param2']):.6f}, param3={adam_final_params['param3']:.6f}, param4={adam_final_params['param4']:.6f}, param5={adam_final_params['param5']:.6f}\n")
#     f.write(f"Powell - Execution Time: {powell_time:.2f} mins, Function Evaluations: {physics_powell.counter}, Iterations: {result.nit}, NaN Count: {nan_count}\n")
#     f.write(f"Powell Final Parameters: param1={result.x[0]:.6f}, param2={result.x[1]:.6f}, param3={result.x[2]:.6f}, param4={result.x[3]:.6f}, param5={result.x[4]:.6f}\n")





# # #===========================================    Nelder-Mead  ========================================

# #modelling parameter
# N_optimizer_nm = 1000
# loss_t = []
# nan_count = 0

# def physics_nm():

#     if not hasattr(physics_nm, 'counter'):
#         physics_nm.counter = 0

#     rpc = paramrpc* 1.4e-9 #3. (nm) [s] #2.4e-9 (m) [1] 5e-9 [4]

#     ND = tf.Variable(tf.zeros([time_steps]), trainable=False)
#     PR = tf.Variable(tf.zeros([time_steps]), trainable=False)
#     VF = tf.Variable(tf.zeros([time_steps]), trainable=False)
#     TVF_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
#     Sigma_ppt_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False, dtype=np.float32)
#     MPR_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
#     TND_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False)
#     Yield_t = tf.Variable(tf.zeros([time_steps-1]), trainable=False) # because the 0th time step doesn't have any associated YS prediction
#     Tau_c_t = tf.Variable(tf.zeros([time_steps-1, 2]), trainable=False, dtype=tf.float32) #weak and strong

#     Gamma =  0.039
#     Mppt = 1.0   #[1] 3.1
#     coars_coeff = 2.5e-3 #1e-3 #the smaller, the later the peak

#     wtp_Mg = 0.5
#     wtp_Si = 0.43
#     mw_Mg = 24.305  #(g/mol)
#     mw_Si = 28.09   #(g/mol)
#     mw_Al = 26.98   #(g/mol)
#     xm_denom = wtp_Mg/mw_Mg + wtp_Si/mw_Si + (100.-wtp_Mg-wtp_Si)/mw_Al
#     xm_Mg0 = wtp_Mg/mw_Mg/xm_denom
#     xm_Si0 = wtp_Si/mw_Si/xm_denom
#     xm_Al0 = (100.-wtp_Mg-wtp_Si)/mw_Al/xm_denom
#     xm_Mg = xm_Mg0
#     xm_Si = xm_Si0
#     xm_Al = xm_Al0
    
#     # print(f'CalculateNucleation:')
#     dGvol_ = dGvol(xm_Mg, xm_Si)
#     Z_ = Z()
#     Beta_ = Beta()
#     Rs_ = Rs(dGvol_, Gamma, param1)
#     DeltaGsnorm_ = DeltaGsnorm(dGvol_, Gamma)
#     Rp_ = Rp(Rs_, Gamma, param3)
#     dNdT_ = dNdT_nograd(Z_, Beta_, DeltaGsnorm_, param2)
#     KWNcounter = 1
#     ND, PR = CalculateNucleation(KWNcounter, dNdT_, dt, Rp_, ND, PR)

#     for KWNcounter in range(1, time_steps):
        
#         t = time_[KWNcounter]
        
#         # print(f'==============================counter: {KWNcounter}======time: {t}==========')

#         ND, PR = CalculateGrowth(KWNcounter, dt, xm_Mg, xm_Si, Rs_, ND, PR, Gamma, coars_coeff, param4, param5)

#         # xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t,  MPR_t, Yield_t, Tau_c_t, Sigma_ppt_t, TND, MPR, TVF = CalculateVF(KWNcounter, xm_Mg0, xm_Si0, ND, PR, VF, Yield_t, TVF_t, TND_t, MPR_t, Mppt, Tau_c_t, Sigma_ppt_t, rpc)
#         xm_Mg, xm_Si, ND, PR, VF, TVF_t, TND_t, MPR_t = Update(KWNcounter, xm_Mg0, xm_Si0, ND, PR, VF, TVF_t, TND_t, MPR_t)
#         Yield_t = Strength(KWNcounter, ND, PR, rpc, xm_Mg, xm_Si, Mppt, Yield_t, Tau_c_t, Sigma_ppt_t)

#     lambda_l2 = 1e-4 # 1e-6  # Adjust this factor to control the strength of regularization
#     l2_reg = (
#         tf.nn.l2_loss(param1- 1.0) +
#         tf.nn.l2_loss(param2) +
#         tf.nn.l2_loss(param3- 1.0) +
#         tf.nn.l2_loss(param4- 1.0) +
#         tf.nn.l2_loss(param5- 1.0)
#     )

#     # loss_ = loss_function(Yield_t, Yield_interpolated) + lambda_l2 * l2_reg
#     loss_ = loss_function(Yield_t, Yield_interpolated) 
    
#     # Count NaN values
#     global nan_count
#     if tf.math.is_nan(loss_):
#         nan_count += 1
    
#     loss_t.append(loss_.numpy())

#     # Store history
#     param1_t.append(param1.numpy())
#     param2_t.append(param2.numpy())
#     param3_t.append(param3.numpy())
#     param4_t.append(param4.numpy())
#     param5_t.append(param5.numpy())
#     paramrpc_t.append(paramrpc.numpy())
    
#     print(f"NM physics_counter {physics_nm.counter} with loss value {loss_} ")
#     print()

#     visual(time_, TND_t, MPR_t, TVF_t, Yield_t, loss_t, Yield_interpolated, x, y, N_optimizer_nm, param1_t, param2_t, param3_t, param4_t, param5_t, physics_nm.counter, optimizer='nm')
#     physics_nm.counter += 1
#     return loss_

# paramrpc = tf.Variable(0.85, trainable = False, dtype=np.float32) 
# param1 = tf.Variable(param1_init, trainable = True, dtype=np.float32)
# param2 = tf.Variable(tf.exp(param2_init), trainable = True, dtype=np.float32)
# param3 = tf.Variable(param3_init, trainable = True, dtype=np.float32)
# param4 = tf.Variable(param4_init, trainable = True, dtype=np.float32)
# param5 = tf.Variable(param5_init, trainable = True, dtype=np.float32)

# param1_t = []
# param2_t = []
# param3_t = []
# param4_t = []
# param5_t = []
# paramrpc_t = []

# param1_t.append(param1.numpy())
# param2_t.append(param2.numpy())
# param3_t.append(param3.numpy())
# param4_t.append(param4.numpy())
# param5_t.append(param5.numpy())
# paramrpc_t.append(paramrpc.numpy())

# tic = time.time()

# def objective_function(params):
#     # Unpack parameters
#     param1.assign(params[0])
#     param2.assign(params[1]) 
#     param3.assign(params[2])
#     param4.assign(params[3])
#     param5.assign(params[4])
    
#     loss_value = physics_nm()
#     return loss_value.numpy()

# # Initial parameter values
# initial_params = [param1.numpy(), param2.numpy(), param3.numpy(), param4.numpy(), param5.numpy()]

# # Create initial simplex with X% offsets
# n_params = len(initial_params)
# initial_simplex = np.zeros((n_params + 1, n_params))
# initial_simplex[0] = initial_params  # First vertex is the initial point

# for i in range(n_params):
#     initial_simplex[i + 1] = initial_params.copy()
#     initial_simplex[i + 1][i] = initial_params[i] * 1.9

# # Create bounds for each parameter separately
# bounds = [
#     (0.0, 3.0),     # param1 bounds
#     (0.0, 3.0),     # param2 bounds 
#     (0.0, 3.0),     # param3 bounds
#     (0.0, 3.0),     # param4 bounds
#     (0.0, 3.0)      # param5 bounds
# ]
# # Set flag to track main optimization calls
# objective_function.is_main_call = True

# # Custom callback to track iterations
# def callback(xk):
#     objective_function.is_main_call = True
#     return False

# # Add iteration counter
# minimize_count = 0

# def callback(xk):
#     global minimize_count
#     minimize_count += 1
#     print(f"minimize_count {minimize_count} of {N_optimizer_nm}")
#     objective_function.is_main_call = True
#     return False

# result = minimize(objective_function, initial_params, method='Nelder-Mead',
#                 #  bounds= [],
#                  callback=callback, 
#                 options={
#                 "xatol": 1e-10,          # default absolute tolerance in x
#                 "fatol": 1e-4,          # default absolute tolerance in f(x)
#                 "maxiter": N_optimizer_nm,        #
#                 "maxfev": None,         # defaults internally to 200 * len(x0)
#                 "disp": False,          # no messages
#                 "return_all": False,    # don’t store history
#                 "initial_simplex": initial_simplex # custom simplex with x% offsets
#                          })

# print("Parameters: ", result.x)

# print("Number of iterations for NM: ", result.nit)


# toc = time.time()
# nm_time = (toc-tic)/60.
# nm_iterations = result.nit
# print(f'execution time of NM is {nm_time} mins')
# print(f'Nelder-Mead optimization message: {result.message}')

# results_dir = './results'
# os.makedirs(results_dir, exist_ok=True)

# with open(os.path.join(results_dir, f'results_seed_{seed_value}.txt'), 'w') as f:
#     f.write(f"Random Seed: {seed_value}\n")
#     # f.write(f"ADAM - Execution Time: {adam_time:.2f} mins, Function Evaluations: {adam_iterations}\n")
#     # f.write(f"ADAM Final Parameters: param1={adam_final_params['param1']:.6f}, param2={np.exp(adam_final_params['param2']):.6f}, param3={adam_final_params['param3']:.6f}, param4={adam_final_params['param4']:.6f}, param5={adam_final_params['param5']:.6f}\n")
#     # f.write(f"Powell - Execution Time: {powell_time:.2f} mins, Function Evaluations: {physics_powell.counter}, Iterations: {result.nit}, NaN Count: {nan_count}\n")
#     # f.write(f"Powell Final Parameters: param1={result.x[0]:.6f}, param2={result.x[1]:.6f}, param3={result.x[2]:.6f}, param4={result.x[3]:.6f}, param5={result.x[4]:.6f}\n")
#     f.write(f"NM - Execution Time: {nm_time:.2f} mins, Function Evaluations: {physics_nm.counter}, Iterations: {result.nit}, NaN Count: {nan_count}\n")
#     f.write(f"NM Final Parameters: param1={result.x[0]:.6f}, param2={result.x[1]:.6f}, param3={result.x[2]:.6f}, param4={result.x[3]:.6f}, param5={result.x[4]:.6f}\n")
