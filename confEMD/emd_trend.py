# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:31:40 2024

@author: Weijie Gu
"""

import numpy as np
from emd_energy_spectrum import emd_energy_spectrum

def emd_trend(modes, t, cutoff=0.4):
    '''Calculates trend of signal.
    
    This function identifies the trend of the analysed signal from input EMD modes.

    Parameters
    ----------
    modes : numpy array
        Input IMFs (EMD modes) 
    t : numpy array
        Time
    cutoff : float, optional
        Cutoff value for identifying the trend. The default is 0.4, corresponding
        to 2.5 oscillations.

    Returns
    -------
    result : numpy array
        Trend of the analysed signal

    '''
    dt = t[1]-t[0]
    sp = emd_energy_spectrum(modes, t, plot_fitting=True)
    length_mode = len(modes[:,0])*dt  #length of mode (time duration)
    ind = np.where(sp.period > cutoff*length_mode)[0] 
    result = np.zeros(len(modes[:,0]))
    for i in ind:
        result += modes[:,i]
    
    result += modes[:,-1] #add the last mode (residual), because 'sp.period' 
    #does not include the residual
    return result