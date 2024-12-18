import emd


def emd_modes(x, sd_thresh=1e-4):
    '''Returns set of EMD modes for a given timeseries x.
    
    This function wraps sift function from emd library and returns a set of emd modes.
    
    Parameters
    ----------
    x : numpy array
        Time series data
    sift : floaf
        sifting factor

    Returns
    -------
    modes : numpy array
        Set of emd modes

    '''
    config = emd.sift.get_config('sift')
    config['imf_opts/sd_thresh'] = sd_thresh
    modes = emd.sift.sift(x, **config)
    
    return modes