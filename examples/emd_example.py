# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 14:55:48 2024

@author: Weijie Gu
"""

import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt

from scope.fourier import fit_fourier
from scope.emd import emd_modes, emd_trend, emd_energy_spectrum, emd_noise_conf
from scope.utils import plot_modes, plot_signal, plot_fft_spectrum, plot_emd_spectrum


# #Generate time series
# L = 30 #length of time series
# N = 300 #number of data points 
# dt = L / N 

# t = dt * np.arange(N)
# # start_time = np.max(t) * 0.05 
# decay_time = np.max(t) * 0.3

# trend = np.exp(-t/decay_time)
# signal = 0.1 * np.sin(2*np.pi*(1/6)*t) #freq=5/30
# noise = 0.06 * cn.powerlaw_psd_gaussian(0, t.size) + 0.1 * cn.powerlaw_psd_gaussian(2, t.size)

# x = trend + signal + noise

# plt.plot(t, x)
# plt.plot(t, trend)
# plt.plot(t, trend+signal)
# plt.xlabel('Time')
# plt.ylabel('Amplitude')
# plt.title('Input Signal')
# plt.show()

# np.savetxt('examples/example data', (t, trend, signal, x))


#%%############################################################################
#Load example data file
t, trend, signal, x = np.loadtxt('examples/example data')

plt.plot(t, x)
plt.plot(t, trend)
plt.plot(t, trend+signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Input Signal')
plt.show()

#%%############################################################################

dt = t[1]-t[0]
N = len(x) #number of data points 
x -= np.mean(x) #set mean to zero

#Calculate EMD modes and trend
modes = emd_modes(x, sd_thresh=1e-4)

#plot modes
plot_modes(t, modes)

#Calculate trend (rough)
modes = emd_trend(modes, t)
trend_emd = modes[:, -1]
plot_signal(t, trend_emd, 'Trend of the signal')

#subtract this trend from the signal 
x = x - trend_emd

#plot detrended signal
plot_signal(t, x, 'Detrended signal')

#plot updated modes
plot_modes(t, modes)

#%%############################################################################

#Estimate noise parameters from FFT of the detrended signal
fit_fft = fit_fourier(x, dt, fap=0.05)

alpha = fit_fft['pl_index']

#Convert frequency to period
period = 1 / fit_fft['frequency']

# plot the results of fit_fourier
plot_fft_spectrum(fit_fft)

#%%############################################################################

#Calculate EMD power spectrum
emd_sp = emd_energy_spectrum(modes, t, plot_fitting=True)

cutoff_period = 0.4 * len(x) * dt #show cutoff period


plot_emd_spectrum(emd_sp, cutoff_period)

#%%############################################################################

# false alarm probability
fap = 0.05

#Confidence limits for coloured noise
conf_c = emd_noise_conf(t, alpha=alpha, period_min=2*dt, 
                        period_max=N*dt, num_samples=100, 
                        signal_energy=fit_fft['color_energy'], fap=fap)
#Confidence limits for white noise
if fit_fft['white_energy'] > 0: # check if there is only colored noise model
    conf_w = emd_noise_conf(t, alpha=0, period_min=2*dt,
                            period_max=N*dt, num_samples = 100, 
                            signal_energy=fit_fft['white_energy'], fap=fap)

#Upper confidence limit for the combined noises
conf_up = conf_c['up'] + conf_w['up']

#Lower confidence limit for the combined noises
conf_down = conf_c['down'] + conf_w['down']

conf_period = conf_c['period']

conf_mean = conf_c['mean_energy'] + conf_w['mean_energy']

# plot emd spectrum
plot_emd_spectrum(emd_sp, cutoff_period, conf_period, conf_up, conf_down, conf_mean, fap)

