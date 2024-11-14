# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 21:23:28 2024

@author: Weijie Gu
"""

import numpy as np
import colorednoise as cn
import emd
from confEMD.emd_period_energy import emd_period_energy
from confEMD.emd_noise_conf import emd_noise_conf
import matplotlib.pyplot as plt


L = 30 
N = 300
dt = L/N
t = dt * np.arange(N)
cl_index_arr = [0, 0.5, 1, 1.5, 2]
num_cl_index = len(cl_index_arr)

def generate_data():
    n_iter = 2000
    
    config = emd.sift.get_config('sift')
    config['imf_opts/sd_thresh'] = 1e-4
    
    modal_period_total = []
    modal_energy_total = []
    mode_n_total = []
    
    for i in range (num_cl_index):
        modal_period = np.array([])
        modal_energy = np.array([])
        mode_n = np.array([])
        print (i)
        
        for j in range (n_iter):
            x = cn.powerlaw_psd_gaussian(cl_index_arr[i], N)
            x /= np.std(x) #normalise std to unity
            x *= np.sqrt(1/N) #normalise total energy to unity
            
            #Calculate EMD modes
            modes = emd.sift.sift(x, **config)
            num_modes = len(modes[0,:]) 
            num_modes -= 1 #-1 to exclude the last mode
            
            # period_arr = np.empty(num_modes, dtype='object')
            # global_ws = np.empty(num_modes, dtype='object')
            # best_fit = np.empty(num_modes, dtype='object')
            # period = np.zeros(num_modes) 
            
            for k in range (num_modes): 
                s = modes[:,k]
                emd_period_energy_result = emd_period_energy(s, t)
           
                modal_period = np.append(modal_period, emd_period_energy_result.dominant_period)
                modal_energy = np.append(modal_energy, emd_period_energy_result.energy)
                mode_n = np.append(mode_n, k+1)
        
                
                # period[k] = emd_period_energy_result.dominant_period
                # period_arr[k] = emd_period_energy_result.period
                # global_ws[k] = emd_period_energy_result.global_ws
                # best_fit[k] = emd_period_energy_result.best_fit
                
            # plt.figure(figsize=(16, 9))
            # for k in range (num_modes): 
            #     plt.subplot(3,3,k+1)
            #     plt.plot(period_arr[k], global_ws[k])
            #     plt.plot(period_arr[k], best_fit[k])
            #     plt.xlabel('Period')
            #     plt.ylabel('Power')
            # plt.tight_layout()
            # plt.show()
            
        modal_period_total.append(modal_period)
        modal_energy_total.append(modal_energy)
        mode_n_total.append(mode_n)
        
    for i in range (num_cl_index):
        np.savetxt('energy_period_relation/Data/modal_period_'+str(i+1), modal_period_total[i])
        np.savetxt('energy_period_relation/Data/modal_energy_'+str(i+1), modal_energy_total[i])
        np.savetxt('energy_period_relation/Data/mode_n_'+str(i+1), mode_n_total[i])
    
    
def generate_conf():
    for i in range (num_cl_index):
        modal_period = np.loadtxt('energy_period_relation/Data/modal_period_'+str(i+1))
        
        #Calculate confidence limits
        conf = emd_noise_conf(t, alpha=cl_index_arr[i], period1=np.min(modal_period), period2=np.max(modal_period), num_samples=2000, signal_energy=1, fap=0.01)
        np.savetxt('energy_period_relation/Data/conf_up_'+str(i+1), conf.up)
        np.savetxt('energy_period_relation/Data/conf_down_'+str(i+1), conf.down)
        np.savetxt('energy_period_relation/Data/conf_period_'+str(i+1), conf.period)
        np.savetxt('energy_period_relation/Data/conf_mean_energy_'+str(i+1), conf.mean_energy)
        np.savetxt('energy_period_relation/Data/mean_period_pt_'+str(i+1), conf.mean_period_pt)
        np.savetxt('energy_period_relation/Data/mean_energy_pt_'+str(i+1), conf.mean_energy_pt)

#%%
plt.figure(figsize=(10, 10))
for i in range (num_cl_index):
    modal_period = np.loadtxt('energy_period_relation/Data/modal_period_'+str(i+1))
    modal_energy = np.loadtxt('energy_period_relation/Data/modal_energy_'+str(i+1))
    mode_n = np.loadtxt('energy_period_relation/Data/mode_n_'+str(i+1))
    
    conf_up = np.loadtxt('energy_period_relation/Data/conf_up_'+str(i+1))
    conf_down = np.loadtxt('energy_period_relation/Data/conf_down_'+str(i+1))
    conf_period = np.loadtxt('energy_period_relation/Data/conf_period_'+str(i+1))
    conf_mean_energy = np.loadtxt('energy_period_relation/Data/conf_mean_energy_'+str(i+1))
    mean_period_pt = np.loadtxt('energy_period_relation/Data/mean_period_pt_'+str(i+1))
    mean_energy_pt = np.loadtxt('energy_period_relation/Data/mean_energy_pt_'+str(i+1))
    
    params, cov = np.polyfit(np.log(conf_period), np.log(conf_mean_energy), 1, cov=True)
    sample_period = np.linspace(np.min(modal_period), np.max(modal_period), 10)
    expectation = np.exp(params[1])*sample_period**params[0]
    
    cl_index = params[0]+1
    cl_index_err = np.sqrt(cov[0,0])
    print ('alpha = ', cl_index, '+/-', cl_index_err)
    
    high_cutoff = 0.4*N*dt
    
    #Plot energy-period distributions of IMFs
    plt.subplot(3,2,i+1)
    plt.scatter(modal_period, modal_energy, s=0.1, color='orange')
    plt.scatter(mean_period_pt, mean_energy_pt, marker='+', color='red')
    plt.plot(sample_period, expectation, color='blue')   
    plt.plot(conf_period, conf_up, color='green')
    plt.plot(conf_period, conf_down, color='green')
    plt.axvline(x = np.min(mean_period_pt), color = 'black', linestyle='dashed')
    plt.axvline(x = high_cutoff, color = 'black', linestyle='dashed')
    plt.title(r'$\alpha$=' + str(cl_index_arr[i]))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Period')
    plt.ylabel('Energy')
    
plt.tight_layout()
plt.show()









