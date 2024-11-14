# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:55:48 2024

@author: Weijie Gu
"""

import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt

from confEMD.fit_fourier import fit_fourier
from confEMD.emd_modes import emd_modes
from confEMD.emd_trend import emd_trend 
from confEMD.emd_energy_spectrum import emd_energy_spectrum

#Plot EMD modes
def plot_modes(modes):
    num_modes = len(modes[0,:])
    for i in range (num_modes):
        plt.subplot(num_modes,1,i+1)
        plt.plot(t, modes[:,i])
        plt.ylabel('Mode ' + str(i+1))
        plt.ylim(np.min(modes), np.max(modes))
        
    plt.xlabel('Time')
    plt.show()
    
#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 

t = dt * np.arange(N)
start_time = np.max(t) * 0.05 
decay_time = np.max(t) * 0.3

trend = np.exp(-t/decay_time)
signal = 0.1 * np.sin(2*np.pi*(1/6)*t) #freq=5/30
noise = 0.06 * cn.powerlaw_psd_gaussian(0, t.size) + 0.1 * cn.powerlaw_psd_gaussian(2, t.size)

x = trend + signal + noise

plt.plot(t, x)
plt.plot(t, trend)
plt.plot(t, trend+signal)
plt.xlabel('Time')
plt.ylabel('Power')
plt.title('Input Signal')
plt.show()

x -= np.mean(x) #set mean to zero
#%%
#-----------------------------------------------------------------------------
#Calculate EMD modes and trend

modes = emd_modes(x, sift=1e-4)

#plot modes
plot_modes(modes)

#Calculate trend (rough)
trend_emd = emd_trend(modes, t)
plt.plot(t, trend_emd)
plt.title('trend of signal')
plt.show()

#subtract this trend from the signal 
x = x - trend_emd

#plot detrended signal
plt.plot(t, x)
plt.title('Detrended signal')
plt.show()

#%%
#Estimate noise parameters from FFT of the detrended signal
fit_fft = fit_fourier(x,dt,fap=0.05)

#Convert frequency to period
period = 1/fit_fft['frequency']

#coloured noise index
alpha = fit_fft['pl_index']

alpha_stderr = fit_fft['pl_index_stderr']

plt.loglog(period, fit_fft['power'], linewidth=2)
plt.loglog(period, fit_fft['expectation_continuous'], 
           label=rf'$\alpha$ = {alpha:.2f} +/- {alpha_stderr:.2f} ', color='blue', linewidth=3)
plt.loglog(period, fit_fft['confidence_limit'], label='95%',
           color='red', linewidth=3)

plt.title('FFT Spectrum')
plt.xlabel('Period [a.u.]')
plt.ylabel('Fourier Magnitude [a.u.]')
plt.legend()
plt.show()
#%%
#Calculate EMD power spectrum
sp = emd_energy_spectrum(modes, t, plot_fitting=True)

cutoff_period = 0.4 * len(x) * dt #show cutoff period

#plot EMD spectrum
plt.errorbar(sp['period'], sp['energy'], xerr=sp['period_err'], label='EMD Spectrum', fmt='.', color='green', ms=15, capsize=5, mew=2)
plt.axvline(x = cutoff_period, color = 'black', linestyle='dashed') #cutoff period = 0.4*(mode length)

plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1e1)
plt.xlim(0.1, 60)
plt.title('EMD Spectrum')
plt.xlabel('Period [a.u.]')
plt.ylabel('EMD Modal Energy [a.u.]')
plt.legend()
plt.grid()
plt.show()


#%%
#-----------------------------------------------------------------------------
from confEMD.emd_noise_conf import emd_noise_conf
#Confidence limits for coloured noise
conf_c = emd_noise_conf(t, alpha = alpha, period1 = 2*dt, period2 = N*dt, num_samples = 100, 
                        signal_energy = fit_fft['color_energy'], fap = 0.05)
#Confidence limits for white noise
conf_w = emd_noise_conf(t, alpha = 0, period1 = 2*dt, period2 = N*dt, num_samples = 100, 
                        signal_energy = fit_fft['white_energy'], fap = 0.05)

#Upper confidence limit for the combined noises
conf_up = conf_c['up'] + conf_w['up']

#Lower confidence limit for the combined noises
conf_down = conf_c['down'] + conf_w['down']

conf_period = conf_c['period']

conf_mean = conf_c['mean_energy'] + conf_w['mean_energy']

#%%
#Plot EMD spectrum with confidence limits
plt.errorbar(sp['period'], sp['energy'], xerr=sp['period_err'], label='EMD Spectrum', fmt='.', color='green', ms=15, capsize=5, mew=2)
plt.axvline(x = cutoff_period, color='black', linestyle='dashed')
plt.plot(conf_period, conf_up, color='red', label='95%')
plt.plot(conf_period, conf_down, color='red')
plt.plot(conf_period, conf_mean, color='blue')


plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-3, 1e1)
plt.xlim(0.1, 60)
plt.title('EMD Spectrum')
plt.xlabel('Period [a.u.]')
plt.ylabel('EMD Modal Energy [a.u.]')
plt.legend()
plt.grid()
plt.show()
