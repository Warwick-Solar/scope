import numpy as np
import colorednoise as cn
from scope.fourier import fit_fourier


def test_slope_fit_mixed():
    #Generate time series
    slope = 2
    L = 30 #length of time series
    N = 600 #number of data points 
    P = 5 #signal period
    dt = L / N 
    t = dt * np.arange(N)

    noise = 0.1 * cn.powerlaw_psd_gaussian(0, t.size) + 0.1 * cn.powerlaw_psd_gaussian(slope, t.size) 
    x = noise
    x -= np.mean(x) #set mean to zero

    fit_fft = fit_fourier(x, dt, fap=0.05)
    
    alpha = fit_fft['pl_index']
    d_alpha = fit_fft['pl_index_stderr']
    
    
    accetpable_delta = 0.2 * slope
    left = alpha - d_alpha - accetpable_delta
    right = alpha + d_alpha + accetpable_delta
    
    assert (slope >= left) and (slope <= right)
    
    
def test_slope_fit_pink():
    #Generate time series
    slope = 1
    L = 30 #length of time series
    N = 600 #number of data points 
    P = 5 #signal period
    dt = L / N 
    t = dt * np.arange(N)

    noise = 0.1 * cn.powerlaw_psd_gaussian(slope, t.size) 
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
    #Generate time series
    slope = 0
    L = 30 #length of time series
    N = 600 #number of data points 
    P = 5 #signal period
    dt = L / N 
    t = dt * np.arange(N)

    noise = 0.1 * cn.powerlaw_psd_gaussian(slope, t.size) 
    x = noise
    x -= np.mean(x) #set mean to zero

    fit_fft = fit_fourier(x, dt, fap=0.05)
    
    alpha = fit_fft['pl_index']
    d_alpha = fit_fft['pl_index_stderr']
    
    
    accetpable_delta = 0.1
    left = alpha - d_alpha - accetpable_delta
    right = alpha + d_alpha + accetpable_delta
    
    assert (slope >= left) and (slope <= right)