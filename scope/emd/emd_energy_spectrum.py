import numpy as np
from .emd_period_energy import emd_period_energy
import matplotlib.pyplot as plt

def emd_energy_spectrum(modes, t, plot_fitting=False):
    ''' Computes EMD power spectrum (modal energy densities). 
    
    This function calculates the modal energy density of input intrinsic mode 
    functions (IMFs). The dominant modal period (and its uncertainty) of all input 
    modes are stored.
    
    Parameters
    ----------
    modes : numpy array
        Input IMFs (EMD modes) 
    t : numpy array
        Time
    plot_fitting : bool, optional
        Plots each global wavelet spectrum and its best fit for all modes in a subplot. 
        The default is False.

    Returns
    -------
    emd_energy_spectrum_result : dict
        Attributes
        ----------
        period : numpy array
            Modal period of all IMFs
        period_err : numpy array
            Uncertainty of modal period
        energy : numpy array
            Modal energy density of all IMFs

    '''
    
    num_modes = len(modes[0,:])-1 #number of input IMFs, exclude the last mode (residual) 
    #from the spectrum fit since it does not have a Gaussian shape
    N = len(modes[:,0])
    
    period = np.zeros(num_modes) #dominant modal period
    period_err = np.zeros(num_modes) #uncertainty of dominant modal period
    energy = np.zeros(num_modes) #modal energy density
    period_modes = np.empty(num_modes, dtype='object') #stores the array of 
    #modal period obtained from the wavelet transform 
    global_ws = np.empty(num_modes, dtype='object') #global wavelet spectrum of each mode
    best_fit = np.empty(num_modes, dtype='object') #values of function that fits 
    #the global wavelet spectrum
    
    for i in range (num_modes): 
        s = modes[:,i]
        
        emd_period_energy_result = emd_period_energy(s, t)
        period[i] = emd_period_energy_result['dominant_period']
        period_err[i] = emd_period_energy_result['dominant_period_err']
        period_modes[i] = emd_period_energy_result['period']
        global_ws[i] = emd_period_energy_result['global_ws']
        best_fit[i] = emd_period_energy_result['best_fit']
        energy[i] = N*np.std(s)**2 
    
    #Calculate spectral density
    # dt = t[1]-t[0]
    # length_mode = len(modes[:,0])*dt #time duration of the IMFs
    # freq = 1/(period*dt)
    # edges = np.concatenate([[0.5], np.sqrt(freq[1:num_modes] * freq[0:num_modes-1]), [1.0 / length_mode]])
    # print (edges)
    # spectral_density = energy / (edges[0:num_modes-1] - edges[1:num_modes]) 
    
    #Plot Gaussian fitting of the global wavelet spectra
    if plot_fitting == True:
        plt.figure(figsize=(16, 9))
        for i in range (num_modes): 
            plt.subplot(3,3,i+1)
            plt.plot(period_modes[i], global_ws[i])
            plt.plot(period_modes[i], best_fit[i])
            plt.title('Mode ' + str(i+1))
            plt.xlabel('Period')
            plt.ylabel('Power')
        plt.tight_layout()
        plt.show()
        
        
    emd_energy_spectrum_result = {
        'period': period,
        'period_err': period_err,
        'energy': energy,
        }
        
    return emd_energy_spectrum_result