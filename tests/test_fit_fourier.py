import numpy as np
import colorednoise as cn
from scope.fourier import fit_fourier


def test_slope_fit_red():
    # fix seed
    np.random.seed(42)
    
    noise = np.load('test data/red_noise.npy')
    slope = 2
    L = 30 #length of time series
    N = len(noise) #number of data points 
    dt = L / N 

    x = noise
    x -= np.mean(x) #set mean to zero

    fit_fft = fit_fourier(x, dt, fap=0.05)
    
    alpha = fit_fft['pl_index']
    d_alpha = fit_fft['pl_index_stderr']
    
    accetpable_delta = 0.2 * slope
    left = alpha - d_alpha - accetpable_delta
    right = alpha + d_alpha + accetpable_delta
    
    assert (slope >= left) and (slope <= right)
    
    
def test_slope_fit_mixed():
    # fix seed
    np.random.seed(42)
    
    noise = np.load('test data/mixed_noise.npy')
    slope = 2.0
    L = 30 #length of time series
    N = len(noise) #number of data points 
    dt = L / N 

    x = noise
    x -= np.mean(x) #set mean to zero

    fit_fft = fit_fourier(x, dt, fap=0.05)
    
    alpha = fit_fft['pl_index']
    d_alpha = fit_fft['pl_index_stderr']
    
    accetpable_delta = 0.2 * slope
    left = alpha - d_alpha - accetpable_delta
    right = alpha + d_alpha + accetpable_delta
    print(slope)
    
    assert (slope >= left) and (slope <= right)
    

def test_slope_fit_white():
    # fix seed
    np.random.seed(42)
    
    noise = np.load('test data/white_noise.npy')
    slope = 0
    L = 30 #length of time series
    N = 600 #number of data points 
    dt = L / N 

    x = noise
    x -= np.mean(x) #set mean to zero

    fit_fft = fit_fourier(x, dt, fap=0.05)
    
    alpha = fit_fft['pl_index']
    d_alpha = fit_fft['pl_index_stderr']
    
    
    accetpable_delta = 0.1
    left = alpha - d_alpha - accetpable_delta
    right = alpha + d_alpha + accetpable_delta
    
    assert (slope >= left) and (slope <= right)