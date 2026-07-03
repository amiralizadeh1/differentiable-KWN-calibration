# [references] 
# [1] Chen et al., MSE A 2017
# [2] Cinkilic et al. metals 2020
# [3] Bardel et al. Acta Materialia 2013
# [4] Myhr 2001

import time
import numpy as np
import matplotlib.pyplot as plt
import os
import math
# import tensorflow as tf
# import kawin.Thermodynamics
from kawin.Thermodynamics import MulticomponentThermodynamics
phases = ['FCC_A1', 'MG5SI6_B_DP']
therm = MulticomponentThermodynamics('D:/PhD/Implementation/hello/AlMgSi.tdb', ['AL', 'MG', 'SI'], phases, drivingForceMethod='sampling') #'approximate', 'sampling'

param = 1.0

tic = time.time()

class KWN:
    # tic = time.time()
    K = 1.380649e-23 #+ (j.K-1)
    Delta = 7.
    R = 8.31  #(j.K-1.mol-1)
    pi = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
    e = 2.7182818284590452353602874713527
   
    def __init__(self, wtp_Mg, wtp_Si, Gamma, T, FinalTime):
        self.wtp_Mg = wtp_Mg
        self.wtp_Si = wtp_Si
        self.S1 = 5.
        self.S2 = 6.
        self.xp_Mg = 5./11.
        self.xp_Si = 6./11.
        self.v = 1.92e-29 #+ #average volume of atoms in MgxSiy precipitates[1]
        self.a = 2.86e-10 #+ interatomic distance[software] 
        self.N0 = 5.8e28  #+ Number of sites [s] 5.8e28 #@
        self.vm = 3.95e-5  #+ ppt molar volum
        self.epsilon = 1.
        self.Gmod = 2.7e10  #+ #Gmod is the sheer modulus of aluminium matrix (GPa) (N.m-2)
        self.b_dis = 2.84e-10  #+ should be in (m) magnitude of the burger's vector
        self.Mppt = 1.1 #[1]
        self.q = 2. #stress exponent [s]
        self.T = T+273
        self.Gamma = Gamma
        #choose smallest value possible without getting 'no more Mg left' error.
        # 130: 0.038 and more works 
        # 150: 0.038 and more works (this showes funny results esp with volume fraction graph)
        # 170: 0.037 and more works
        # 200: 0.032 and more works
        # 250: 0.018 and more works

        self.coars_coeff = 2.5e-3 # 8.e-3 Coarsening kinetic factor [s]? #2.25e-1 in [olds]   #the smaller, the later the peak

        self.FinalTime = FinalTime

        mw_Mg = 24.305  #(g/mol)
        mw_Si = 28.09   #(g/mol)
        mw_Al = 26.98   #(g/mol)
        xm_denom = wtp_Mg/mw_Mg + wtp_Si/mw_Si + (100.-wtp_Mg-wtp_Si)/mw_Al

        self.xm_Mg0 = wtp_Mg/mw_Mg/xm_denom
        self.xm_Si0 = wtp_Si/mw_Si/xm_denom
        self.xm_Al0 = (100.-wtp_Mg-wtp_Si)/mw_Al/xm_denom
        self.xm_Mg = self.xm_Mg0
        self.xm_Si = self.xm_Si0
        self.xm_Al = self.xm_Al0

        #initialse_KWN
        self.counter = 0
        self.KWNcounter = 1
        self.NumberDensity = []
        self.ParticleRadius = []
        self.VolFraction = [] 
        self.TotalVolFraction_t = []
        self.MeanParticleRadius_t = []
        self.TotalNumberDensity_t = []
        self.Yield = []

        self.Tau_c_Weak_t = []
        self.Tau_c_Strong_t = []
        self.sssFraction_t = []
        self.Mg_t = []
        self.Si_t = []

        self.Delta_dis = 0.5  #[1] 0.25
        self.Fk_coeff_1 = 2.*self.Delta_dis*self.Gmod*self.b_dis**2. #Delata_dis is a parameter depending on te shape and nature of dislocations
        self.Fk_coeff_2 = math.sqrt(2)*self.Delta_dis*self.Gmod*self.b_dis
        self.Gamma_dis = 2./math.sqrt(3)*self.Delta_dis*self.Gmod*self.b_dis**2. #calculated from rp=rpc condition #Gamma is the line tension of dislocations 
        self.Gamma_b_coeff = math.sqrt(2.*math.sqrt(3)*self.Gamma_dis)*self.b_dis

        self.k_Mg = 21.1e6   #[1] 21.1e6 Pa/wt%. [4] 29.0e6
        self.k_Si = 10.5e6  #[1] 10.5e6  Pa/wt% [4] 66.3
        self.Sigma0 = 55000000  #+
        self.c02 = 500 #+ #[s] #(c02-Sigma) may give a negative value, which is invalid for the elongation equation.
        self.c03 = 10 #+ #[s]
        self.rpc = 0.85 * 1.4e-9 #3. (nm) [s] #2.4e-9 (m) [1] 5e-9 [4]

        print(f'KWN initialized, FinalTime: {self.FinalTime}')

    def D_Mg(self):
        self.D0_Mg = 2.2e-4  #[2] 6.23e-6 (m2.s-1)  # [3] 2.2×10−4
        # D0_Mg = 2.2e-4
        self.Q_Mg = 115000  # 115000[2] (j.mol-1) activation energy [3] 130000
        # self.D_Mg_ = self.D0_Mg * math.exp(-self.Q_Mg/(self.R*self.T))
        self.D_Mg_ = self.D0_Mg * self.e**(-self.Q_Mg/(self.R*self.T))
        return self.D_Mg_ #(m2.s-1)
    
    def D_Si(self):
        # T should be in Kelvin
        self.D0_Si = 2.2e-4  #[2] 2.48e-4 (m2.s-1) # [3] 2.2×10−4  
        self.Q_Si = 115000  # 137000 [2] (j.mol-1) activation energy [3] 130000
        # self.D_Si_ = self.D0_Si * math.exp(-self.Q_Si/(self.R*self.T))
        self.D_Si_ = self.D0_Si * self.e**(-self.Q_Si/(self.R*self.T))
        return self.D_Si_

    def dGvol(self):
        #drivingForceMethod: 'approximate', 'sampling', or 'curvature'
        # temperature: T(K), composition: weight%
        # xcomp: Initial composition of parent matrix phase in atomic fraction Use float for binary system and array of solutes for multicomponent systems
        dG = therm._getDrivingForceApprox([self.xm_Mg, self.xm_Si], self.T, precPhase='MG5SI6_B_DP', returnComp=True)[0]
        dG = -dG.item()
        self.dGvol_ = dG/self.vm
        return self.dGvol_ #j.m-3
        
    def Rs(self):
        self.dGvol()
        if (self.dGvol_ < 0):
            self.Rs_ = (-2.*self.Gamma/self.dGvol_ ) * (2.*self.Delta/(3.*self.Delta-1.))
        else:
            raise Exception('positive driving force')
        return self.Rs_ #(m)
    
    def Z(self):
        self.Z_ = 1.
        return self.Z_
    
    def Beta(self):
        self.Beta_ = 1.
        return self.Beta_
    
    def DeltaGsnorm(self):
        self.dGvol()
        if (self.dGvol_<0.):
            shape_factor = (4.*self.Delta**3.)/(3.*self.Delta-1.)**2.
            DeltaGs = (16.*self.pi*self.Gamma**3.)/(3.*self.dGvol_**2.)

            self.DeltaGsnorm_ =   DeltaGs   *  shape_factor  /  (self.K*self.T)
        else:
            raise Exception('positive driving force')
        return self.DeltaGsnorm_
    
    def dNdT(self):
        self.expFactor = self.e**(-self.DeltaGsnorm())
        self.dNdT_ = self.N0 * self.Z() * self.Beta() * self.expFactor #(s-1)
        # self.dNdT_ = self.N0 * self.expFactor #(s-1)
        return self.dNdT_
    
    def Rp(self):
        self.Rp_ = self.Rs() + math.sqrt(self.K*self.T/(self.pi*self.Delta*self.Gamma))/2.
        return self.Rp_  
    
    def CalculateNucleation(self): #after converting to Decimal, this function doesn't need to know KWNcounter
        self.NumberDensity.append(self.dNdT()*self.dt*3600.) #(s-1.h.s.h-1)
        self.ParticleRadius.append(self.Rp())

    def CalculateVelocity(self, radius):

        d2Gx_Mg = (self.R * self.T) / self.D_Mg()
        a11 = (self.xp_Mg-self.xm_Mg)**2. / self.xm_Mg * d2Gx_Mg
        a12 = self.xp_Mg * d2Gx_Mg

        d2Gx_Si = (self.R * self.T) / self.D_Si()
        a21 = (self.xp_Si-self.xm_Si)**2. / self.xm_Si * d2Gx_Si
        a22 = self.xp_Si * d2Gx_Si

        b5 = 1./(a11 + a21)
        b10 = 1./(a12 + a22)
        ksi = 1.25 # the smaller, the earlier the peak
        b6 = 4.*self.Gamma*self.Delta*self.vm/(ksi*radius*(3.*self.Delta-1.)*self.R * self.T)

        b7 = 1./self.Rs() - 1./radius

        self.vel_growth = b5*b6*b7
        self.vel_coarse = b10*b6*self.coars_coeff/radius
        self.vel_total = self.vel_growth + self.vel_coarse

    def CalculateGrowth(self):

        for i in range(self.KWNcounter+1): #? KWNcounter starts at 0, before NumberDensity gets called in this function, it was popped by Nuc
            if ((self.ParticleRadius[i]>0.) & (self.NumberDensity[i]>0.)): #how's it possible for particles to have negative radius or negative NumberDensity?

                self.CalculateVelocity(self.ParticleRadius[i])
                # print()
                # print(f' vel_growth is {self.vel_growth} for {i}')
                # print(f' vel_coarse is {self.vel_coarse} for {i}')
                # print(f' vel_total is {self.vel_total} for {i}')
                self.rad_growth = self.ParticleRadius[i] + self.vel_growth*self.dt*3600. #the radius that is only resultant from growth (not coarsening)
                self.rad_total = self.ParticleRadius[i] + self.vel_total*self.dt*3600.

                self.ParticleRadius[i] = self.rad_total
                self.NumberDensity[i] = self.NumberDensity[i]*(self.rad_growth/self.rad_total)**3.

    # def CalculateVolFraction(self):
    #     for i in range(self.KWNcounter):
    #         self.VolFraction[i] =(2.*self.Delta-2./3.)*self.pi*self.ParticleRadius[i]**3.*self.NumberDensity[i]
            
    #     self.VolFraction.append(   (2.*self.Delta-2./3.)*self.pi*self.ParticleRadius[-1]**3.*self.NumberDensity[-1]   )              

    def UpdateVariables1(self):
        self.TotalVolFraction = 0.
        self.TotalNumberDensity = 0.
        self.MeanParticleRadius = 0.
        summation_nlp_1 = 0.
        summation_nf = 0.
        self.summation_nlp_2 = 0.
        TotalNumberDensity_weak = 1. #formerly summation_n #to avoid float devision, set to 1.
        for i in range(self.KWNcounter+1):

            self.VolFraction.append(0.)
            self.VolFraction[i] = (2.*self.Delta-2./3.)*self.pi*self.ParticleRadius[i]**3.*self.NumberDensity[i] # updates all the volume fraction for all the previous classes   
            self.TotalVolFraction = self.TotalVolFraction + (2.*self.Delta-2./3.)*self.pi*self.ParticleRadius[i]**3.*self.NumberDensity[i]
            self.TotalNumberDensity = self.TotalNumberDensity + self.NumberDensity[i]
            self.MeanParticleRadius = self.MeanParticleRadius + self.NumberDensity[i]*self.ParticleRadius[i]
            # self.summation_nlp_2 = self.summation_nlp_2 + self.NumberDensity[i]*2.*self.Delta*self.ParticleRadius[i]


            if (self.ParticleRadius[i] > self.rpc): # strong
                self.summation_nlp_2 = self.summation_nlp_2 + 2.*self.Delta*self.ParticleRadius[i]*self.NumberDensity[i]

            else: # weak
                Fk = self.Fk_coeff_1*self.ParticleRadius[i]/self.rpc #(N)
                summation_nlp_1 = summation_nlp_1 + self.NumberDensity[i]*2.*self.Delta*self.ParticleRadius[i]
                TotalNumberDensity_weak = TotalNumberDensity_weak + self.NumberDensity[i]
                summation_nf = summation_nf + self.NumberDensity[i]*Fk

        self.VolFraction.append((2.*self.Delta-2./3.)*self.pi*self.ParticleRadius[-1]**3.*self.NumberDensity[-1])
        

        self.TotalVolFraction_t.append(self.TotalVolFraction)

        self.MeanParticleRadius = self.MeanParticleRadius/self.TotalNumberDensity
        self.TotalNumberDensity_t.append(self.TotalNumberDensity)
        self.MeanParticleRadius_t.append(self.MeanParticleRadius)

        if self.xm_Mg0 <= self.epsilon*self.TotalVolFraction*self.xp_Mg: raise Exception('No more Mg left in the matrix')
        if self.xm_Si0 <= self.epsilon*self.TotalVolFraction*self.xp_Si: raise Exception('No more Si left in the matrix')
        self.xm_Mg = (self.xm_Mg0-self.epsilon*self.TotalVolFraction*self.xp_Mg)/(1.-self.epsilon*self.TotalVolFraction)
        self.xm_Si = (self.xm_Si0-self.epsilon*self.TotalVolFraction*self.xp_Si)/(1.-self.epsilon*self.TotalVolFraction)
        self.xm_Al = 1. - self.xm_Mg - self.xm_Si

        self.Tau_c_Strong = self.Fk_coeff_2*math.sqrt(self.summation_nlp_2)
        self.Tau_c_Weak = (math.sqrt(summation_nlp_1)/self.Gamma_b_coeff) * (summation_nf/TotalNumberDensity_weak)**1.5 
        # print(f'Tau_c_Weak, Tau_c_Strong: {Tau_c_Weak/(Tau_c_Weak+Tau_c_Strong)}, {Tau_c_Strong/(Tau_c_Weak+Tau_c_Strong)}')

        self.Sigma_ppt = self.Mppt*( self.Tau_c_Strong**self.q + self.Tau_c_Weak**self.q)**(1./self.q)
        self.Sigma_ss =  self.k_Mg*self.xm_Mg + self.k_Si*self.xm_Si
        # self.Sigma = self.Sigma0 + self.Sigma_ss + self.Sigma_ppt
        # self.sssFraction_t.append((self.Sigma_ss/self.Sigma)*100.)
        self.Sigma = self.Sigma0  + self.Sigma_ppt
        self.Yield.append(self.Sigma)
        self.Tau_c_Strong_t.append(self.Tau_c_Strong)
        self.Tau_c_Weak_t.append(self.Tau_c_Weak)
        self.Mg_t.append(self.xm_Mg)
        self.Si_t.append(self.xm_Si)

   

    def visual(self, ND, PR, data, name=None):
        runtime = time.time() - tic
        if name is None:
            name = f'{int(runtime)}_{int(self.T)}_{int(self.FinalTime)}_{self.wtp_Si}_{self.wtp_Si}'

        newFolder = os.path.join('D:/PhD/Implementation', 'experimental')
        if not os.path.exists(newFolder): os.makedirs(newFolder)



        plt.xscale('log')
        # plt.xscale('linear')
        # plt.plot(np.arange(0, kwn.FinalTime, self.dt), [a/10e6 for a in kwn.Yield])
        # plt.plot(t_l, [a/10e6 for a in kwn.Yield], 'bo', markersize=3)
        plt.plot(t_l, [a/1e6 for a in kwn.Yield])
        plt.scatter(data[0], data[1], color = 'red')
        plt.xlabel(f'time (h)')
        # plt.xlim((0,3))
        plt.ylabel('Yield strength (Mpa)')
        plt.title('Yield Strength vs time')
        plt.legend(['model output', 'experimental'])
        print(os.getcwd())        
        path = os.path.join(newFolder, f'Y_t log {name}.png')
        plt.savefig(path)
        plt.show()

        # # plt.xscale('log')
        # plt.xscale('linear')
        # plt.plot(t_l, kwn.Mg_t)
        # plt.plot(t_l, kwn.Si_t)
        # plt.xlabel(f'time (h)')
        # # plt.xlim((0,3))
        # plt.ylabel('Elements (molar fraction)')
        # plt.title('Concetration vs time')
        # plt.legend(['Mg', 'Si'])
        # print(os.getcwd())        
        # path = os.path.join(newFolder, f'Mg_Si {name}.png')
        # plt.savefig(path)
        # plt.show()

        # plt.xscale('log')
        # plt.xscale('linear')
        # plt.plot(t_l, kwn.sssFraction_t)
        # # plt.scatter(data[0], data[1], color = 'red')
        # plt.xlabel(f'time (h)')
        # # plt.xlim((0,3))
        # plt.ylabel('Strength_solidSolution/Strength_total (%)')
        # plt.title('The effect of SS hardening vs. time')
        # plt.legend(['The effect'])
        # # print(os.getcwd())        
        # path = os.path.join(newFolder, f'sssFraction_t {name}.png')
        # plt.savefig(path)
        # plt.show()

        plt.xscale('linear')
        # plt.xscale('linear')
        # plt.plot(np.arange(0, kwn.FinalTime, self.dt), [a/10e6 for a in kwn.Yield])
        # plt.plot(t_l, [a/10e6 for a in kwn.Yield], 'bo', markersize=3)
        plt.plot(t_l, [a/1e6 for a in kwn.Yield])
        plt.scatter(data[0], data[1], color = 'red')
        plt.xlabel(f'time (h)')
        # plt.xlim((0,3))
        plt.ylabel('Yield strength (Mpa)')
        plt.title('Yield Strength vs time')
        plt.legend(['model output', 'experimental'])
        print()
        print(os.getcwd())        
        path = os.path.join(newFolder, f'Y_t linear {name}.png')
        plt.savefig(path)
        plt.show()

        plt.xscale('log')
        # plt.xscale('linear')
        # plt.plot(np.arange(0, kwn.FinalTime, self.dt), self.TotalVolFraction_t)
        plt.plot(t_l, self.TotalVolFraction_t)
        plt.xlabel(f'time (h)')
        # plt.xlim((0,3))
        plt.ylabel('total volume fraction')
        plt.title('total volume fraction vs time')
        print()
        print(os.getcwd())        
        path = os.path.join(newFolder, f'VF_t {name}.png')
        plt.savefig(path)
        plt.show()

        plt.xscale('log')
        # plt.xscale('linear')
        # plt.plot(np.arange(0, kwn.FinalTime, self.dt), self.MeanParticleRadius_t)
        plt.plot(t_l, self.MeanParticleRadius_t)
        plt.xlabel(f'time (h)')
        # plt.xlim((0,3))
        plt.ylabel('Mean Particle Radius')
        plt.title('Mean Particle Radius vs. time')
        print()
        print(os.getcwd())        
        path = os.path.join(newFolder, f'PR_t {name}.png')
        plt.savefig(path)
        plt.show()

        plt.xscale('log')
        # plt.xscale('linear')
        # plt.plot(np.arange(0, kwn.FinalTime, self.dt), self.MeanParticleRadius_t)
        plt.plot(t_l, self.TotalNumberDensity_t)
        plt.xlabel(f'time (h)')
        # plt.xlim((0,3))
        plt.ylabel('TotalNumberDensity_t')
        plt.title('Total Number Density')
        print()
        print(os.getcwd())        
        path = os.path.join(newFolder, f'ND_t {name}.png')
        plt.savefig(path)
        plt.show()

        with open(os.path.join(newFolder, f'{name}.txt'), 'w') as file:
            file.write(f'Model inputs:\n')
            file.write(f'time, dt: {self.FinalTime}, {self.dt}\n')
            file.write(f'temp: {self.T}\n')
            file.write(f'wtp_Mg, wtp_Si: {self.wtp_Mg}, {self.wtp_Si}\n')
            file.write(f'Gamma: {self.Gamma}\n')
            file.write(f'coars_coeff: {self.coars_coeff}\n')
            file.write(f'Delta: {self.Delta}\n')
            file.write(f'D0_Mg: {self.D0_Mg} Q_Mg: {self.Q_Mg} D0_Si: {self.D0_Si} Q_Si: {self.Q_Si}\n')
            file.write(f'\n')
            file.write(f'Model outputs:')
            file.write(f'run_time: {runtime}\n')
            file.write(f'ND: {ND}\n')
            file.write(f'PR: {PR}\n')
            file.write(f'Sigma: {self.Sigma}\n') #[2]
    

# #################################################################### Loop-wise KWN ###########################
kwn = KWN(0.5, 0.43, 0.039, 150., 190.)
t = 0.
t_l = []
# for kwn.KWNcounter in range(int(kwn.FinalTime/kwn.dt)): #excludes FinalTime from the simulation
for kwn.KWNcounter in range(500):    
    if(t > kwn.FinalTime ): break
    kwn.dt = (kwn.KWNcounter+1)**3/10000
    t = t + kwn.dt 
    t_l.append(t)
    print(f'==============================counter: {kwn.KWNcounter}======time: {t}=============================')

    print('molar concentrations:')
    print(kwn.xm_Mg)
    print(kwn.xm_Si)
    print(kwn.xm_Al)

    print(f'CalculateNucleation:')
    kwn.CalculateNucleation()

    print(f'dNdT: {kwn.dNdT_}')
    print(f'Rp: {kwn.Rp_}')
    print(f'Z: {kwn.Z_}')
    print(f'Beta: {kwn.Beta_}')
    print(f'DeltaGsNorm: {kwn.DeltaGsnorm_}')
    print(f'expFactor: {kwn.expFactor}')
    print(f'Rs: {kwn.Rs_}')
    print(f'dGvol: {kwn.dGvol_}')
    # print()
    # print(kwn.NumberDensity)
    # print()
    # print(kwn.ParticleRadius)
    # print()


    print(f'CalculateGrowth:')
    kwn.CalculateGrowth()
    # print(f'D_Si: {kwn.D_Si_}')
    # print(f'D_Mg: {kwn.D_Mg_}')
    # print()
    # print(kwn.NumberDensity)
    # print()
    # print(kwn.ParticleRadius)
    # print()

    # print(f'CalculateVolFraction:')
    # kwn.CalculateVolFraction()
    # print()
    # # print(f'kwn.TotalVolFraction: {kwn.TotalVolFraction}')
    # print()

    print(f'UpdateVariables:')
    kwn.UpdateVariables1()
    # print(f'kwn.MeanParticleRadius: {kwn.MeanParticleRadius}')
    # print(f'kwn.TotalVolFraction: {kwn.TotalVolFraction}')
    # print(f'kwn.TotalNumberDensity: {kwn.TotalNumberDensity}')
    # print()


    print(f'Sigma0: {kwn.Sigma0}')
    print(f'Sigma_ss: {kwn.Sigma_ss}')
    print(f'Sigma_ppt: {kwn.Sigma_ppt}')
    print(f'Sigma: {kwn.Sigma}')

    print('concentrations:')
    print(kwn.xm_Mg)
    print(kwn.xm_Si)
    print(kwn.xm_Al)
    print()
    print('concentration difference:')
    print(kwn.xm_Mg - kwn.xm_Mg0)
    print(kwn.xm_Si - kwn.xm_Si0)
    print(kwn.xm_Al - kwn.xm_Al0)
    print()
    print(f'Ln(dNdt): {math.log(kwn.N0)} + {math.log(kwn.Z_)} + {math.log(kwn.Beta_)} - {kwn.DeltaGsnorm_}')
    print('\n\n\n\n')

data = [[1, 2, 3, 8, 24, 48, 72, 120, 168, 336], [74, 89, 127, 187, 219, 230, 252, 280, 255, 188]]

kwn.visual(kwn.NumberDensity, kwn.ParticleRadius, data, name=11)

# print(kwn.Yield)
