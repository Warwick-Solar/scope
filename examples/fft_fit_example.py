#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 13:35:20 2024

@author: u2273880
"""

import numpy as np
import colorednoise as cn
import matplotlib.pyplot as plt
import emd
from confEMD.fit_fourier import fit_fourier

def plot_spectrum(fit_fft):
    #Convert frequency to period
    period = 1 / fit_fft['frequency']

    #coloured noise index
    alpha = fit_fft['pl_index']
    alpha_stderr = fit_fft['pl_index_stderr']

    plt.loglog(period, fit_fft['power'], linewidth=2)
    plt.loglog(period, fit_fft['expectation_continuous'],
               label=rf'$\alpha$ = {alpha:.2f} +/- {alpha_stderr:.2f} ',
               color='blue', linewidth=3)
    plt.loglog(period, fit_fft['confidence_limit'], label='95%',
               color='red', linewidth=3)
    plt.title('FFT Spectrum')
    plt.xlabel('Period [a.u.]')
    plt.ylabel('Fourier Magnitude [a.u.]')
    plt.legend()
    plt.show()
    
def plot_signal(x):
    plt.plot(t, x)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Input Signal')
    plt.show()

    

#%% only white noise

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(0, t.size) 
x = noise 

plot_signal(x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_spectrum(fit_fft)

#%% only red noise

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(2, t.size)  
x = noise 

plot_signal(x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_spectrum(fit_fft)

#%% white + red noise

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(0, t.size)  + 0.1 * cn.powerlaw_psd_gaussian(2, t.size) 
x = noise 

plot_signal(x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_spectrum(fit_fft)

#%% white + red noise + oscillatory signal

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
P = 5 #signal period
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(0, t.size)  + 0.1 * cn.powerlaw_psd_gaussian(2, t.size) 
signal = 0.1 * np.sin(2*np.pi*t/P) #freq=5/30
x = noise + signal

plot_signal(x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_spectrum(fit_fft)


#%%

