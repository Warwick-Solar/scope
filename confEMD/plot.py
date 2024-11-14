#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:43:19 2024

@author: u2273880
"""

import matplotlib.pyplot as plt

def plot_spectrum(fit_fft):
    """
    Plots the FFT spectrum from fitting results, displaying the power, the model fit,
    and confidence limits.

    Parameters
    ----------
    fit_fft : dict
        A dictionary containing the results of FFT fitting with the following keys:
            - 'frequency' : numpy array
                Frequencies obtained from the FFT fitting.
            - 'power' : numpy array
                Power spectrum values corresponding to each frequency.
            - 'pl_index' : float
                Power-law index of the colored noise model.
            - 'pl_index_stderr' : float
                Standard error of the power-law index.
            - 'expectation_continuous' : numpy array
                Expected power values from the model fit to the data.
            - 'confidence_limit' : numpy array
                Confidence limit values for the power spectrum.
            - 'conf_prob' : float
                Confidence probability associated with the confidence limit.

    Returns
    -------
    None
        This function does not return any values. It displays a log-log plot of the
        power spectrum, including the model fit and confidence limits.

    Notes
    -----
    The function plots the period (1/frequency) on the x-axis and the Fourier magnitude
    on the y-axis. It uses a log-log scale for both axes to better visualize the
    frequency spectrum over a wide range. The power-law index with its error is included
    in the legend, as well as the confidence probability of the confidence limits.

    """

    #Convert frequency to period
    period = 1 / fit_fft['frequency']

    #coloured noise index
    alpha = fit_fft['pl_index']
    alpha_stderr = fit_fft['pl_index_stderr']

    #confidence probability
    prob = fit_fft['conf_prob']

    plt.loglog(period, fit_fft['power'], linewidth=2)
    plt.loglog(period, fit_fft['expectation_continuous'],
               label=rf'$\alpha$ = {alpha:.2f} +/- {alpha_stderr:.2f} ',
               color='blue', linewidth=3)
    plt.loglog(period, fit_fft['confidence_limit'], label=f'{prob*100:.0f}%',
               color='red', linewidth=3)
    plt.title('FFT Spectrum')
    plt.xlabel('Period [a.u.]')
    plt.ylabel('Fourier Magnitude [a.u.]')
    plt.legend()
    plt.show()
    
def plot_signal(t, x):
    """
    Plots the input signal over time.

    Parameters
    ----------
    t : numpy array
        Time values for the signal, used as the x-axis in the plot.
    x : numpy array
        Signal values corresponding to each time point in `t`, used as the y-axis in the plot.

    Returns
    -------
    None
        This function does not return any values. It displays a plot of the input signal
        as a function of time.

    Notes
    -----
    This function produces a simple 2D line plot, showing how the signal varies over time.
    The x-axis represents time, and the y-axis represents the signal magnitude.
    
    """
    plt.plot(t, x)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Input Signal')
    plt.show()