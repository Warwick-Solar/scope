###############################################################################
# This file is an adapted version of the hands-on example of the processing of 
# a noisy synthetic time series using the emd_modes function, to instead use 
# the more versatile tune_emd function. It also includes the option of being 
# used to obtain a non-stationary injected signal.
###############################################################################

import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt

from scope.fourier import fit_fourier
from scope.emd import tune_emd, emd_energy_spectrum, emd_noise_conf
from scope.utils import plot_signal, plot_fft_spectrum, plot_emd_spectrum

use_example_data = True # Change to False to use your own data
non_stationary = False # Change to True to generate non-stationary data

if use_example_data:
    if non_stationary:
        t, flare_trend, qpp_signal, x = np.loadtxt('example_data_ns.txt')
    else:
        t, flare_trend, qpp_signal, x = np.loadtxt('example_data.txt')
else:
    if non_stationary:
        L = 30 #length of time series
        N = 300 #number of data points
        dt = L / N
        t = dt * np.arange(N)
        decay_flare = np.max(t) * 0.3
        decay_qpp = np.max(t) * 0.5
        flare_trend = np.exp(-t/decay_flare)
        f0 = 1/4 # initial frequency
        f1 = 1/12 # final frequency (longer period over time)
        f_t = f0 + (f1 - f0) * (t / np.max(t)) # linear frequency drift
        phase = 2 * np.pi * np.cumsum(f_t) * dt # integrate frequency to get phase
        qpp_signal = 0.15 * np.exp(-t/decay_qpp) * np.sin(phase)
        noise = 0.05 * cn.powerlaw_psd_gaussian(0, t.size) + 0.05 * cn.powerlaw_psd_gaussian(2, t.size)
        x = flare_trend + qpp_signal + noise
    else:
        L = 30 #length of time series
        N = 300 #number of data points
        dt = L / N
        t = dt * np.arange(N)
        decay_flare = np.max(t) * 0.3
        decay_qpp = np.max(t) * 0.5
        flare_trend = np.exp(-t/decay_flare)
        qpp_signal = 0.15 * np.exp(-t/decay_qpp) * np.sin(2*np.pi*(1/5)*t)
        noise = 0.05 * cn.powerlaw_psd_gaussian(0, t.size) + 0.05 * cn.powerlaw_psd_gaussian(2, t.size)
        x = flare_trend + qpp_signal + noise

plt.plot(t, x)
plt.plot(t, flare_trend)
plt.plot(t, flare_trend+qpp_signal)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Input Signal')
plt.show()

#%%############################################################################
dt = t[1]-t[0]
N = len(x) #number of data points 

# Determine EMD modes
modes = tune_emd(x, show=True)

# Isolate trend from rest of modes (longest period mode)
trend_emd = modes[:, -2]

fig, ax = plt.subplots(1)
ax.plot(t, x, alpha=0.8)
plot_signal(t, trend_emd, 'Empirical trend of the signal', ax=ax)

# Subtract this trend from the signal 
x = x - trend_emd

# Plot detrended signal
plot_signal(t, x, 'Detrended signal')
#%%############################################################################

# Estimate noise parameters from FFT of the detrended signal
fit_fft = fit_fourier(x, dt, fap=0.05)

alpha = fit_fft['pl_index']

# Convert frequency to period
period = 1 / fit_fft['frequency']

# Plot the results of fit_fourier
plot_fft_spectrum(fit_fft)

#%%############################################################################

#Calculate EMD power spectrum
emd_sp = emd_energy_spectrum(modes, t, plot_fitting=False)

cutoff_period = 0.4 * len(x) * dt #show cutoff period

plot_emd_spectrum(emd_sp, cutoff_period)

#%%############################################################################

# False alarm probability
fap = 0.05

# Confidence limits for coloured noise
conf_c = emd_noise_conf(t, alpha=alpha, period_min=2*dt, 
                        period_max=N*dt, num_samples=300, 
                        signal_energy=fit_fft['color_energy'], fap=fap)
# Confidence limits for white noise
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

# Upper confidence limit for the combined noises
conf_up = conf_c['up'] + conf_w['up']

# Lower confidence limit for the combined noises
conf_down = conf_c['down'] + conf_w['down']

conf_period = conf_c['period']

conf_mean = conf_c['mean_energy'] + conf_w['mean_energy']

# Plot emd spectrum
plot_emd_spectrum(emd_sp, cutoff_period, conf_period, conf_up, conf_down, conf_mean, fap)

#%%############################################################################
# Plot a significant EMD mode together with the initial signal
fig, ax = plt.subplots(1)
ax.plot(t, modes[:,5], alpha=0.8, label='EMD mode')
ax.plot(t, qpp_signal, alpha=0.8, label='Initial signal')
plt.ylim(-0.2,0.2)
plt.xlabel('Time')
plt.ylabel('Signal')
ax.legend()