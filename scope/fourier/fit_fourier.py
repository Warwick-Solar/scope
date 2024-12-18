import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model

OFFSET = 0.25068

def piecewise_linear(x, x0, y0, k1):
    ''' A piecewise linear function to be fitted.
    
    The piecewise linear function consists of the function of a straight line and
    a flat line. 

    Parameters
    ----------
    x : numpy array
        Fourier frequency
    x0 : float
        X-coordinate of the knot 
    y0 : float
        Y-coordinate of the knot 
    k1 : float
        Gradient of the straight line

    Returns
    -------
    numpy array
        The function

    '''
    return np.piecewise(x, [x < x0, x > x0], [lambda x:k1*x + y0 - k1*x0, lambda x:y0])

def broken_power_law (freq, freq0, N_c, pl_index, N_w):
    ''' Model of the broken power law.
    
    A piecewise function joining two models of power law:
    COLOURED NOISE and WHITE NOISE.

    Parameters
    ----------
    freq : numpy array
        Fourier frequencies
    freq0 : float
        X-coordinate of the knot
    N_c : float
        Energy (coefficient) of the coloured-noise model
    pl_index : float
        Index of the coloured-noise model
    N_w : float
        Energy (coefficient) of the white-noise model

    Returns
    -------
    numpy array
        The function

    '''
    return np.piecewise(freq, [freq < freq0, freq > freq0], [lambda freq:N_c*freq**(-pl_index), lambda freq: N_w])

def continuous_power_law(freq, N_c, pl_index, N_w):
    '''Model of the continuous power law.
    
    Superposition of two models of power law:
    COLOURED NOISE + WHITE NOISE.

    Parameters
    ----------
    freq : numpy array
        Fourier frequencies
    N_c : float
        Energy (coefficient) of the coloured-noise model
    pl_index : float
        Index of the coloured-noise model
    N_w : float
        Energy (coefficient) of the white-noise model

    Returns
    -------
    numpy array
        The function

    '''
    return N_c/np.mean(freq**(-pl_index))*freq**(-pl_index) + N_w

def log_power(freq, N_c, pl_index, N_w):
    ''' Function describing the logarithm (base 10) of the power spectrum.
    
    Calculates the logarithm (base 10) of the continuous power law + bias.
    The bias term originates from the fact that the power spectrum is scattered
    around the true spectrum with a chi-squared distribution with 2 degrees of 
    freedom.
    
    Parameters
    ----------
    freq : numpy array
        Fourier frequencies
    N_c : float
        Energy (coefficient) of the coloured-noise model
    pl_index : float
        Index of the coloured-noise model
    N_w : float
        Energy (coefficient) of the white-noise model

    Returns
    -------
    numpy array
        The function

    '''
    return np.log10(N_c/np.mean(freq**(-pl_index))*freq**(-pl_index) + N_w) - OFFSET


def fit_fourier(x, dt, fap, plot_spectrum = False):
    ''' A debiased least squares fitting algorithm.
    
    This function fits the input signal with the continuous power law using 
    debiased least squares fitting, and calculates the frequency-dependent
    confidence limit based on a user-defined false alarm probability.
    
    The method of debiased least squares fitting is based on [1].
    
    [1] 'https://www.aanda.org/articles/aa/abs/2005/07/aa1453/aa1453.html'

    Parameters
    ----------
    x : numpy array
        Noise signal
    dt : float
        Interval of the time series
    fap : float
        False alarm probability
    plot_spectrum: bool, optional
        Plot the power spectrum, with fittings by the broken power law 
        and continuous power law. The confidence limits of the continous power 
        law fit are also plotted. The default is False.

    Returns
    -------
    fit_fourier_result: dict
        Attributes
        ----------
        power : numpy array
            FFT power
        frequency : numpy array
            Fourier frequencies
        frequency0 : float
            Frequency at which the signal varies from coloured noise to white noise
        expectation_broken : numpy array
            Frequency dependent expectation of the FFT power calculated from the broken power law model
        expectation_continuous : numpy array
            Frequency dependent expectation of the FFT power calculated from the continuous power law model      
        white_energy : float
            Energy of the white-noise component
        color_energy : float
            Energy of the coloured-noise component
        pl_index : float
            Index of the coloured-noise component
        confidence_limit : numpy array
            Frequency dependent confidence limit based on the provided false alarm probability

    '''
    
    n = x.size 
    
    #Compute the power spectrum
    sp = np.fft.fft(x) 
    freq = np.fft.fftfreq(n,d=dt)[1:n//2] #discard 0 Hz and ##Nyquist frequency
    power = 2 * (np.abs(sp[1:n//2])**2) / (n**2) # power spectrum
    norm = n*dt / np.std(x)**2 #see eq. (1) in [1], but use normalise by std not by mean
    power *= norm #power spectrum with modified normalisation

    #Take the logarithm
    freq_fit = np.log10(freq)
    power_fit = np.log10(power)
    
    nf = freq_fit.size
    
    #Fit piecewise function to the power
    x0_guess = freq_fit[0] + (freq_fit[-1]-freq_fit[0])/2 #middle
    y0_guess = min(power_fit) + (max(power_fit) - min(power_fit)) / 2 #middle
    k1_guess = (power_fit[0] - y0_guess)/(freq_fit[0] - x0_guess)
    
    mod_piecewise_linear = Model(piecewise_linear)
    pars_piecewise_linear = mod_piecewise_linear.make_params(
        x0={
            'value': x0_guess,
            'min': min(freq_fit),
            'max': max(freq_fit)
            },
        y0={
            'value': y0_guess,
            'min': min(power_fit),
            'max': max(power_fit)
            },
        k1={
            'value': k1_guess
            }
        )
    
    result_piecewise_linear = mod_piecewise_linear.fit(power_fit, pars_piecewise_linear, x=freq_fit)
    
    #Extract parameters
    freq0 = 10**result_piecewise_linear.params['x0'].value
    N_c = 10 ** (
        result_piecewise_linear.params['y0'].value
        - result_piecewise_linear.params['k1'].value * result_piecewise_linear.params['x0'].value
        + OFFSET
    )
    pl_index = -result_piecewise_linear.params['k1'].value
    N_w = 10**(result_piecewise_linear.params['y0'].value + OFFSET) 
    params_broken_power_law = [freq0, N_c, pl_index, N_w]
    
    #Fit log continuous power law model to the power spectrum
    mod_log_power = Model(log_power)
    #use the fitted parameters from the broken power law model as initial guess, ignore freq0
    #N_c*np.mean(freq**(-pl_index)) because the broken power law model is not normalised with the mean of the power
    pars_log_power = mod_log_power.make_params(
        N_c={
            'value':  N_c*np.mean(freq**(-pl_index)), 
            'min': 0.0
            },
        pl_index={
            'value': min(3.0, max(pl_index, 0.0)),
            'min': -0.1,
            'max': 3.0
            }, 
        N_w={
            'value': N_w, 
            'min': 0.0}
        )
    
    result_log_power = mod_log_power.fit(power_fit, pars_log_power, freq=freq)
    
    # check if the fitting of the slope was successfull 
    alpha = np.abs(result_log_power.params['pl_index'].value)
    d_alpha = result_log_power.params['pl_index'].stderr
    N_c = result_log_power.params['N_c'].value
    N_w = result_log_power.params['N_w'].value

    # if there is only red noise d_alpha can be None when fitted by power law + const
    if d_alpha is None:
        d_alpha = 100 * alpha # set d_alpha a high float value
    # if there is high uncertainty in a slope or whitre noise prevails then fit by a simpler model
    if d_alpha / max(alpha, 1e-6) > 0.8 or N_w / N_c > 100:
        # fit with only power-law part
        pars_log_power['N_w'].set(value=0, vary=False)
        result_log_power = mod_log_power.fit(power_fit, pars_log_power, freq=freq)


    #Extract parameters
    N_c = result_log_power.params['N_c'].value
    pl_index = result_log_power.params['pl_index'].value
    pl_index_stderr = result_log_power.params['pl_index'].stderr
    N_w = result_log_power.params['N_w'].value
    params_continuous_power_law = [N_c, pl_index, N_w]
    
    #calculate energy 
    white_energy = N_w * nf * n / norm
    color_energy = N_c * nf * n / norm
    
    # #calculate standard deviation of noise
    # white_std = np.sqrt(N_w*nf)
    # color_std = np.sqrt(N_c*nf)
    
    #calculate expectation
    expectation_broken = broken_power_law(freq, *params_broken_power_law)
    expectation_continuous = continuous_power_law(freq, *params_continuous_power_law)
    
    #calculate confidence limits 
    cl_ratio_gamma = -2*np.log(1 - (1-fap)**(1/nf))
    confidence_limit = 10 ** (
        np.log10(continuous_power_law(freq, *params_continuous_power_law))
        + np.log10(cl_ratio_gamma / 2)
    )
    
    #Plot results
    if plot_spectrum == True:
        plt.loglog(10**x0_guess, 10**y0_guess, 'x') #initial guess of knot
        
        plt.loglog(freq, power, color='grey')
        plt.loglog(freq, broken_power_law(freq, *params_broken_power_law), linestyle='dashed', color='b', label='broken power law')
        plt.loglog(freq, continuous_power_law(freq, *params_continuous_power_law), color='r', label='best fit')
        plt.loglog(freq, confidence_limit, linestyle='dashed', color='black', label = 'P(>)=' + str(int(100-fap*100)) + '%')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (a.u.)')
        plt.tick_params(top=True, right=True, which='both', direction='in')
        plt.legend()
        plt.show()
        
        
    fit_fourier_result = {
        'power': power, 
        'frequency': freq,
        'frequency0': freq0,
        'expectation_broken': expectation_broken,
        'expectation_continuous': expectation_continuous,
        'white_energy': white_energy,
        'color_energy': color_energy,
        'pl_index': pl_index,
        'pl_index_stderr': pl_index_stderr,
        'confidence_limit': confidence_limit,
        'N_w': N_w,
        'N_c': N_c,
        'conf_prob': 1 - fap
    }

    return fit_fourier_result
