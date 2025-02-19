###############################################################################
# This example illustrates the period-energy relation for different types of 
# noise
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import emd
from scope.emd import emd_noise_conf

L = 30 
N = 300
dt = L/N
t = dt * np.arange(N)
cl_index_arr = [0, 1, 1.5, 2]
num_cl_index = len(cl_index_arr)
n_iter = 1000

config = emd.sift.get_config('sift')
config['imf_opts/sd_thresh'] = 1e-4


plt.figure(figsize=(10, 10))

for i in range (num_cl_index):
    
    conf = emd_noise_conf(t, alpha=cl_index_arr[i], period_min=2*dt,
                          period_max=L,
                          num_samples=n_iter, signal_energy=1, fap=0.01)
    
    modal_period = conf['period_all']
    modal_energy = conf['energy_all']
    mode_n = conf['mode_n_all']
    
    conf_up = conf['up']
    conf_down = conf['down']
    conf_period = conf['period']
    conf_mean_energy = conf['mean_energy']
    mean_period_pt = conf['mean_period_pt']
    mean_energy_pt = conf['mean_energy_pt']
    
    high_cutoff = 0.4*N*dt
    
    #Plot energy-period distributions of IMFs
    plt.subplot(3,2,i+1)
    plt.scatter(modal_period, modal_energy, s=0.1, color='orange')
    plt.scatter(mean_period_pt, mean_energy_pt, marker='+', color='red')
    plt.plot(conf_period, conf_mean_energy, color='blue')   
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









