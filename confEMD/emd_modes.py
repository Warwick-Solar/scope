#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:33:02 2024

@author: u2273880
"""

import emd


def emd_modes(x, sift=1e-4):
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
    config['imf_opts/sd_thresh'] = sift
    modes = emd.sift.sift(x, **config)
    
    return modes