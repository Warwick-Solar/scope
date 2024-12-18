import numpy as np
from .emd_energy_spectrum import emd_energy_spectrum

def emd_trend(modes, t, cutoff=0.4):
    """
    Calculates the trend of a signal from its Empirical Mode Decomposition (EMD) modes.

    This function identifies the trend of the analyzed signal by summing specific
    modes based on their periods. Modes with periods exceeding a fraction of the 
    total signal duration, specified by the `cutoff` parameter, are considered part 
    of the trend. The remaining modes are considered oscillatory components.

    Parameters
    ----------
    modes : numpy array, shape (n_samples, n_modes)
        Input Intrinsic Mode Functions (IMFs) or EMD modes, where each column represents a mode.
    t : numpy array, shape (n_samples,)
        Time values corresponding to the signal.
    cutoff : float, optional
        Cutoff value as a fraction of the total signal duration for identifying the trend.
        Default is 0.4, corresponding to approximately 2.5 oscillations over the signal length.

    Returns
    -------
    result : numpy array, shape (n_samples, n_modes)
        Array containing the modified modes:
        - All original modes excluding the trend in the first columns.
        - The trend as the last column of the array.

    Notes
    -----
    - The trend is calculated by summing modes with periods longer than `cutoff * signal_duration`.
    - The residual (last mode) is always included as part of the trend.
    - The function returns the remaining modes alongside the identified trend for further analysis.
    """
    dt = t[1] - t[0]
    sp = emd_energy_spectrum(modes, t, plot_fitting=False)
    
    length_mode = modes.shape[0] * dt  #length of mode (time duration)
    ind = np.where(sp['period'] > cutoff * length_mode)[0] 
    #add the last mode (residual), because 'sp.period' does not include the residu
    ind = np.append(ind, [-1])
    # sum the modes to obtain a trend
    trend = np.sum(modes[:, ind], axis=1)
    # get the residual modes
    modes_no_trend = np.delete(modes, ind, axis=1)
    
    # get the modes where the last one is the trend
    result = np.concatenate((modes_no_trend, trend.reshape((-1, 1))), axis=1)

    return result