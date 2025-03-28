###############################################################################
# This file is a hands-on example of the processing of a noisy synthetic time 
# series using scope
###############################################################################

import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt

from scope.fourier import fit_fourier
from scope.emd import emd_modes, emd_trend, emd_energy_spectrum, emd_noise_conf
from scope.utils import plot_modes, plot_signal, plot_fft_spectrum, plot_emd_spectrum

use_example_data = True # set to False for a new generated time series

if use_example_data:
    #Load example data file
    t, trend, signal, x = np.loadtxt('example_data.txt')
else:
    #Generate time series
    L = 30 #length of time series
    N = 300 #number of data points 
    dt = L / N 

    t = dt * np.arange(N)
    decay_time = np.max(t) * 0.3

    trend = np.exp(-t/decay_time)
    signal = 0.1 * np.sin(2*np.pi*(1/3)*t) #freq=5/30
    noise = 0.06 * cn.powerlaw_psd_gaussian(0, t.size) + 0.1 * cn.powerlaw_psd_gaussian(2, t.size)

    x = trend + signal + noise

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
modes = emd_modes(x, sd_thresh=0.0001)

#plot modes
plot_modes(t, modes)

#Calculate trend (rough)
modes = emd_trend(modes, t)
trend_emd = modes[:, -1]

fig, ax = plt.subplots(1)
ax.plot(t, x, alpha=0.8)
plot_signal(t, trend_emd, 'Empirical trend of the signal', ax=ax)

#subtract this trend from the signal 
x = x - trend_emd

#plot detrended signal
plot_signal(t, x, 'Detrended signal')
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
emd_sp = emd_energy_spectrum(modes, t, plot_fitting=False)

cutoff_period = 0.4 * len(x) * dt #show cutoff period


plot_emd_spectrum(emd_sp, cutoff_period)

#%%############################################################################

# false alarm probability
fap = 0.05

#Confidence limits for coloured noise
conf_c = emd_noise_conf(t, alpha=alpha, period_min=2*dt, 
                        period_max=N*dt, num_samples=300, 
                        signal_energy=fit_fft['color_energy'], fap=fap)
#Confidence limits for white noise
if fit_fft['white_energy'] > 0: # check if there is only colored noise model
    conf_w = emd_noise_conf(t, alpha=0, period_min=2*dt,
                            period_max=N*dt, num_samples = 300, 
                            signal_energy=fit_fft['white_energy'], fap=fap)
else:
    size = len(conf_c['up'])
    conf_w = {}
    conf_w['up'] = np.zeros(size)
    conf_w['down'] = np.zeros(size)
    conf_w['mean_energy'] = np.zeros(size)

#Upper confidence limit for the combined noises
conf_up = conf_c['up'] + conf_w['up']

#Lower confidence limit for the combined noises
conf_down = conf_c['down'] + conf_w['down']

conf_period = conf_c['period']

conf_mean = conf_c['mean_energy'] + conf_w['mean_energy']

# plot emd spectrum
plot_emd_spectrum(emd_sp, cutoff_period, conf_period, conf_up, conf_down, conf_mean, fap)

#%%############################################################################
# Plot a significant EMD mode together with the initial signal
fig, ax = plt.subplots(1)
ax.plot(t, modes[:,-3], alpha=0.8, label='EMD mode')
ax.plot(t, signal, alpha=0.8, label='Initial signal')
plt.ylim(-0.2,0.2)
plt.xlabel('Time')
plt.ylabel('Signal')
ax.legend()
