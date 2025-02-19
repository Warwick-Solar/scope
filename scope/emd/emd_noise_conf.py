import numpy as np
import colorednoise as cn
import emd
from .emd_period_energy import emd_period_energy
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit
from tqdm import tqdm

'''
This script runs as follows:
1. Input power law index and noise energy from the 'fit_fourier' function.
2. Generate 500 (by default) noise samples with the same power law index 
    and energy as the input.
3. Apply Empirical Mode Decomposition (EMD) to each noise sample.
4. Extract the dominant period and modal energy for each intrinsic mode
    function (IMF).
5. Since the modal energy of the mth IMF should have a chi-square distribution,
    extract the mean period, mean energy and the number of degrees of freedom 
    (DoF) by fitting the distribution of modal energy for each mode number. 
    Achieved by the 'emd_noise_fit' function.
6. Since the energy-period relation and DoF-period relation of IMF are linear 
    in log-log scale, fit a straight line on them to obtain the exact linear 
    relation. Generate 500 data points for the lower and upper confidence
    limits based on the linear relation and 95% (by default) confidence level.
    Achieved by 'make_lin_dep_loglog' and 'emd_noise_conf' separately.
'''
        
def chisqr_pdf(x, mean_energy, dof):
    ''' A modified chi-square distribution function to be fitted.
    
    Degrees of freedom - DoF
    The chi-square distribution function:
    (DoF)/(mean modal energy) * chi^2[DoF*(modal energy)/(mean modal energy)]
   
    Parameters
    ----------
    x : numpy array
        Modal energy
    mean_energy : float
        Mean modal energy
    dof : float
        Number of degrees of freedom of the chi-square distribution

    Returns
    -------
    f : numpy array
        Chi-square distribution function

    '''
    
    y = dof*x/mean_energy 
    f = (dof/mean_energy)*chi2.pdf(y, dof)
    return f

def make_lin_dep_loglog(period, value, period_min, period_max, n_conf):
    ''' Fits a linear function in log-log scale.
    
    This function fits the input data points in log-log scale and generates
    new data points using the fit parameters.
    
    Parameters
    ----------
    period : numpy array
        Modal period
    value : numpy array
        Dependent variable (Energy or DoF)
    period_min : float
        Period interval to make a fitted dependence
    period_max : float
        Period interval to make a fitted dependence
    n_conf : int
        Length of the array to generate

    Returns
    -------
    p : numpy array
        Period 
    v : numpy array
        Dependent variable

    '''
    
    params = np.polyfit(np.log(period), np.log(value), 1)
    
    p = np.exp(np.linspace(np.log(period_min), np.log(period_max), n_conf)) 
    v = np.exp(params[1]) * p**params[0]
    return p, v

def mean_period_energy(period, energy, mode_n, N, dt):
    '''Computes the mean energy of each 'bin' of period.
    
    This function calculates the mean energy within each range of period, 
    which is analogous to bins in a histrogram. 
    
    Parameters
    ----------
    period : numpy array
        Modal period
    energy : numpy array
        Modal energy
    mode_n : numpy array
        Mode number
    N : int
        Length of time series
    dt : float
        Time interval

    Returns
    -------
    p : numpy array
        Mean period of each bin
    e : numpy array
        Mean energy of each bin

    '''
    #exclude the 1st IMF mode 
    ind = np.where(mode_n != 1)[0] 
    period_ind = period[ind]
    energy_ind = energy[ind]
    period_edges = np.exp(np.linspace(np.log(np.min(period_ind)), np.log(np.max(period_ind)), 200)) #in log space
    period_centres = (period_edges[1:] + period_edges[:-1]) / 2 
    mean_modal_energy = np.zeros(len(period_centres)) 
    for j in range (len(period_centres)): 
        mean_modal_energy[j] = np.mean(energy_ind[(period_ind > period_edges[j]) & (period_ind < period_edges[j+1])])
    
    high_cutoff = 0.4*N*dt
    
    #exclude modes with less than 2.5 oscillations
    p = period_centres[(period_centres < high_cutoff)]
    e = mean_modal_energy[(period_centres < high_cutoff)]
    return p, e

def emd_noise_fit(period, energy, mode_n):
    '''Estimates mean modal energy, mean modal period and number of degrees of freedom 
    for every mode number.
    
    This function fits a modified chi-square distribution function on energy
    histogram to extract mean modal energy and number of degrees of freedom.
    
    Parameters
    ----------
    period : numpy array
        Modal period
    energy : numpy array
        Modal energy
    mode_n : numpy array
        Mode number

    Returns
    -------
    mean_energy : numpy array
        Mean modal energy
    dof : numpy array
        Number of degrees of freedom
    mean_period : numpy array
        Mean modal period

    '''
    max_modes = int(np.max(mode_n)) #maximum mode number
    
    mean_period = np.zeros(max_modes-1) #exclude 1st mode
    mean_energy = np.zeros(max_modes-1) #exclude 1st mode
    dof = np.zeros(max_modes-1)

    for i in range(2, max_modes+1): #exclude 1st mode
        #Extract energy and period of specific mode
        ind = np.where(mode_n == i)[0]
        period_ind = period[ind]
        energy_ind = energy[ind]
    
        mean_period[i-2] = np.mean(period_ind) #mean period of each mode
        
        #Histogram of modal energy
        hist, bin_edges = np.histogram(energy_ind, bins='auto')
        bin_centres = (bin_edges[1:] + bin_edges[:-1])/2
        bin_width = bin_edges[1] - bin_edges[0]

        #Normalise the area of histogram to unity
        hist = hist / (np.sum(hist) * bin_width) 
    
        p0 = [bin_centres[np.argmax(hist)], 1.0] #initial guess

        params, _ = curve_fit(chisqr_pdf, bin_centres, hist, p0=p0) 

        mean_energy[i-2] = params[0] #mean modal energy
        dof[i-2] = params[1] #degrees of freedom of chi-square distribution
        
    return mean_energy, dof, mean_period

def emd_noise_conf(t, alpha, period_min, period_max, num_samples=500, signal_energy=1, fap=0.01):
    '''Computes confidence limits for EMD power spectrum.
    
    This function computes confidence limits based on the dyadic property of EMD. 
    
    Parameters
    ----------
    t : numpy array
        Time
    alpha : float
        Power law index 
    period_min : float
        Period interval to make a fitted dependence
    period_max : float
        Period interval to make a fitted dependence
    num_samples : int, optional
        Number of synthetic noise signals to generate. The default is 500.
    energy : float, optional
        Energy of noise estimated by 'fit_fourier'. The default is 1.
    fap : float, optional
        False alarm probability. The default is 0.01.
    
    Returns
    -------
    emd_noise_conf_result : dict
        Attributes
        ----------
        up : numpy array
            Upper confidence limit as a function of period
        down : numpy array
            Lower confidence limit as a function of period
        dof : numpy array
            Number of degrees of freedom as a function of period
        mean_energy : numpy array
            Mean modal energy as a function of period
        period : numpy array
            Period
        mean_period_pt : numpy array
            Mean period of the modal period in each bin
        mean_energy_pt : numpy array
            Mean energy of the modal energy in each bin
        period_all : numpy array
                Periods of all modes found by EMD in generated noise samples
        energy_all : numpy array
            Energies of all modes found by EMD in generated noise samples
        mode_n_all : numpy array
            Mode numbers of all modes found by EMD in generated noise samples
    '''
        
    
    N = len(t) #length of time series
    dt = t[1] - t[0]
    
    period = np.array([])
    energy = np.array([])
    mode_n = np.array([]) #mode number of IMFs
    
    config = emd.sift.get_config('sift')
    config['imf_opts/sd_thresh'] = 0.001
    
    print(fr'Generating and EMD processing {num_samples} noise samples with alpha={round(alpha,2)}')
    for i in tqdm(range(num_samples)):
        x = cn.powerlaw_psd_gaussian(alpha, N) #generate noise signal
        x -= np.mean(x) #normalise mean to zero
        x /= np.std(x) #normalise std to unity
        x *= np.sqrt(signal_energy/N)
        
        modes = emd.sift.sift(x, **config) 
        num_modes = len(modes[0,:]) 
        num_modes -= 1 #exclude the last mode (residual)
        
        for j in range (num_modes): 
            s = modes[:,j]
            emd_period_energy_result = emd_period_energy(s, t)
        
            period = np.append(period, emd_period_energy_result['dominant_period'])
            energy = np.append(energy, emd_period_energy_result['energy'])
            mode_n = np.append(mode_n, j+1)
    
    # check if the number of each specific EMD mode is less than 5 
    # to assure the emd_noise_fit stability
    prohibited_modes = []
    max_modes = int(np.max(mode_n)) #maximum mode number
    for i in range (2, max_modes+1): #exclude 1st mode
        ind = np.where(mode_n == i)[0]
        mode_ind_count = len(energy[ind])
        if (mode_ind_count < 5):
            prohibited_modes.append(i)
  
    period = period[np.logical_not(np.isin(mode_n, prohibited_modes))]
    energy = energy[np.logical_not(np.isin(mode_n, prohibited_modes))]
    mode_n = mode_n[np.logical_not(np.isin(mode_n, prohibited_modes))]
            
    mean_energy, dof, mean_period = emd_noise_fit(period, energy, mode_n)
    mean_period_pt, mean_energy_pt = mean_period_energy(period, energy, mode_n, N, dt)
    
    n_conf = 500 #number of data points of confidence limits
    length_mode = N*dt #length of mode (time duration)
    #Cutoff modes with less than 2.5 oscillations (not suitable for accurate period estimations)
    ind = np.where(mean_period < 0.4*length_mode)[0] 
    ind = ind[1:] #exclude 1st mode
    
    #Fit mean modal energy and dof vs mean modal period 
    period_fit, dof = make_lin_dep_loglog(mean_period[ind], dof[ind], period_min, period_max, n_conf) 
    _, mean_energy = make_lin_dep_loglog(mean_period[ind], mean_energy[ind], period_min, period_max, n_conf)
    
    up = np.zeros(n_conf) #upper confidence limits
    down = np.zeros(n_conf) #lower confidence limits
    for j in range (n_conf):
        down[j] = chi2.ppf(fap*0.5, dof[j])*mean_energy[j]/dof[j]
        up[j] = chi2.ppf(1-fap*0.5, dof[j])*mean_energy[j]/dof[j]
        
        
    emd_noise_conf_result = {
        'up': up,
        'down': down,
        'dof': dof,
        'mean_energy': mean_energy,
        'period': period_fit,
        'mean_period_pt': mean_period_pt,
        'mean_energy_pt': mean_energy_pt,
        'period_all': period,
        'energy_all': energy,
        'mode_n_all': mode_n
        }
        
    return emd_noise_conf_result

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    