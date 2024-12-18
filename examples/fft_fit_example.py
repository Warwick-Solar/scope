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
from scope.fourier import fit_fourier
from scope.utils import plot_signal, plot_fft_spectrum


#%% only white noise

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(0, t.size) 
x = noise 

plot_signal(t, x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_fft_spectrum(fit_fft)

#%% only red noise

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(2, t.size)  
x = noise 

plot_signal(t, x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_fft_spectrum(fit_fft)

#%% white + red noise

#Generate time series
L = 30 #length of time series
N = 300 #number of data points 
dt = L / N 
t = dt * np.arange(N)

noise = 0.1 * cn.powerlaw_psd_gaussian(0, t.size)  + 0.1 * cn.powerlaw_psd_gaussian(2, t.size) 
x = noise 

plot_signal(t, x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_fft_spectrum(fit_fft)

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

plot_signal(t, x)

x -= np.mean(x) #set mean to zero

fit_fft = fit_fourier(x, dt, fap=0.05)

plot_fft_spectrum(fit_fft)


#%%

