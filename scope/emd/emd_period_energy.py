import numpy as np
import matplotlib.pyplot as plt
from .waveletFunctions import wavelet
from scipy import interpolate
from lmfit.models import QuadraticModel, GaussianModel

def fit_global_ws(period, global_ws):
    ''' A global wavelet spectrum fitting algorithm.
    
    This function fits the global wavelet spectrum of intrinsic mode function 
    (IMF) by a Gaussian + quadratic function, in order to extract the 
    dominant modal period from the position of Gaussian peak.

    Parameters
    ----------
    period : numpy array
        Modal period
    global_ws : numpy array
        Global wavelet spectrum of IMF

    Returns
    -------
    dominant_period : float
        Estimated dominant modal period 
    dominant_period_err : float
        Uncertainty of the dominant modal period
    best_fit : numpy array
        The Gaussian + quadratic function that fits the global wavelet spectrum

    '''
    
    #Interpolate and extrapolate the global wavelet spectrum (linearly)
    global_ws_interp_func = interpolate.interp1d(period, global_ws, kind='linear', fill_value='extrapolate')

    bar = np.max(global_ws) 
    ind = np.argmax(global_ws)
    
    amplitude_guess = bar #initial guess of gaussian amplitude
    centre_guess = period[ind] #initial guess of gaussian centre
    
    period_samples = np.linspace(0.0, period[-1], 10000)
    global_ws_interp = global_ws_interp_func(period_samples) 
    
    ind_interp = np.argmax(global_ws_interp)
    
    #Find the two values of period that correspond to 
    #the full width half maximum (FWHM) of the Gaussian
    root1 = period_samples[(abs(global_ws_interp[:ind_interp] - (bar/2))).argmin()]
    root2 = period_samples[ind_interp + (abs(global_ws_interp[ind_interp:] - (bar/2))).argmin()]
    
    sigma_guess = (root2 - root1)/2.355 #standard deviation is about FWHM/2.355
    
    #Construct spectrum fitting models
    gauss_mod = GaussianModel(prefix='gauss_')
    pars = gauss_mod.make_params(center=dict(value=centre_guess, min=root1, max=root2),
                                   sigma=dict(value=sigma_guess, min=0),
                                   amplitude=dict(value=amplitude_guess, min=0))
    
    quadratic_mod = QuadraticModel(prefix='quadratic_')
    pars.update(quadratic_mod.guess(global_ws, x=period))
    
    mod = quadratic_mod + gauss_mod
    
    #Results
    out = mod.fit(global_ws, pars, x=period)
    best_fit = out.best_fit
    dominant_period = out.params['gauss_center'].value
    dominant_period_err = out.params['gauss_sigma'].value
    
    return dominant_period, dominant_period_err, best_fit

def emd_period_energy(s, t, plot_spectrum=False):
    ''' Calculates the dominant period and energy of input intrinsic
    mode function (IMF).
    
    The function calculates the energy of IMF and performs the wavelet transform 
    to compute its period and global wavelet spectrum. The global wavelet 
    spectrum is fitted to estimate the dominant period.
    

    Parameters
    ----------
    s : numpy array
        Input IMF
    t : numpy array
        Time
    plot_spectrum : bool, optional
        Plots the global wavelet spectrum and wavelet power spectrum.
        The default is False.

    Returns
    -------
    emd_period_energy_result : dict
        Attributes
        ----------
        energy : float
            Energy of the input IMF
        dominant_period : float
            Modal period associated with maximum power in global wavelet spectrum
        dominant_period_err : float
            Uncertainty of dominant modal period 
        period : numpy array
            Modal period
        global_ws : numpy array
            Global wavelet spectrum of input IMF
        best_fit : numpy array
            Values of the function that fits the global wavelet spectrum

    '''
    #Compute signal energy
    energy = np.sum(s**2) 
    
    n = len(s)
    dt = t[1]-t[0]
    
    #Parameters of wavelet transform
    pad = 1 #pad the time series with zeroes (recommended)
    dj = 0.125 #8 suboctaves (1/8)
    s0 =  2 * dt #smallest scale
    j1 =  7 / dj #largest scale
    mother = 'MORLET'
    
    #Wavelet transform
    wave, period, _, _ = wavelet(s, dt, pad, dj, s0, j1, mother)
    power = (np.abs(wave)) ** 2 #compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n) #time-average over all times
    
    #Plot the wavelet spectra
    if plot_spectrum == True:
        plt.subplot(2,1,1)
        plt.plot(period, global_ws)
        plt.xlabel('Period (s)')
        plt.ylabel('Power (a.u.)')
        plt.title('Global wavelet spectrum')
        
        plt.subplot(2,1,2)
        plt.contourf(t, period, power)
        plt.xlabel('Time (s)')
        plt.ylabel('Period (s)')
        plt.title('Wavelet power spectrum')
        plt.tight_layout()
        plt.show()
    
    dominant_period, dominant_period_err, best_fit = fit_global_ws(period, global_ws)


    emd_period_energy_result = {
        'energy': energy,
        'dominant_period': dominant_period,
        'dominant_period_err': dominant_period_err,
        'period': period,
        'global_ws': global_ws,
        'best_fit': best_fit,
        }
    
    return emd_period_energy_result