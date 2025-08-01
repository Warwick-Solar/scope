#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  1 13:27:27 2025

@author: u2273880
"""

import numpy as np
from scipy.interpolate import interp1d
from scope.emd import emd_modes, emd_trend, emd_energy_spectrum, emd_noise_conf
from scope.fourier import fit_fourier


def test_emd_modes():
    np.random.seed(42)
    
    smape = 1
    
    t, trend, signal, x = np.loadtxt('tests/test data/signal.txt')
    
    modes = emd_modes(x, sd_thresh=0.0001)
    signal_emd = modes.sum(axis=1)

    smape_test = 100 * np.mean(np.abs(signal_emd-x)/(np.abs(signal_emd)+np.abs(x)))
    
    assert smape_test < smape
    
    
def test_emd_energy_spectrum():
    np.random.seed(42)
    
    p_max = 6 # real signal period

    t, trend, signal, x = np.loadtxt('tests/test data/signal.txt')
  
    x -= np.mean(x) #set mean to zero

    #Calculate EMD modes and trend
    modes = emd_modes(x, sd_thresh=0.0001)

    #Calculate trend (rough)
    modes = emd_trend(modes, t)

    #Calculate EMD power spectrum
    emd_sp = emd_energy_spectrum(modes, t, plot_fitting=False)
    
    ind = np.argmax(emd_sp['energy'])
    p = emd_sp['period'][ind]
    p_err = emd_sp['period_err'][ind]
    
    left = p - p_err
    right = p + p_err
    
    assert (p_max >= left) and (p_max <= right)


def test_emd_noise_conf():
    np.random.seed(42)
    
    p_max = 6 # real signal period
    
    t, trend, signal, x = np.loadtxt('tests/test data/signal.txt')
    
    dt = t[1]-t[0]
    N = len(x) #number of data points 
    x -= np.mean(x) #set mean to zero

    #Calculate EMD modes and trend
    modes = emd_modes(x, sd_thresh=0.0001)

    #Calculate trend (rough)
    modes = emd_trend(modes, t)
    trend_emd = modes[:, -1]

    #subtract this trend from the signal 
    x = x - trend_emd


    #Estimate noise parameters from FFT of the detrended signal
    fit_fft = fit_fourier(x, dt, fap=0.05)

    alpha = fit_fft['pl_index']


    #Calculate EMD power spectrum
    emd_sp = emd_energy_spectrum(modes, t, plot_fitting=False)


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

    conf_period = conf_c['period']

    
    ind = np.argmax(emd_sp['energy'])
    p = emd_sp['period'][ind]
    p_err = emd_sp['period_err'][ind]
    
    f = interp1d(conf_period, conf_up)
    upper_bound = f(emd_sp['period'][ind])
    
    left = p - p_err
    right = p + p_err
    
    assert (p_max >= left) and (p_max <= right) and (upper_bound < emd_sp['energy'][ind])


