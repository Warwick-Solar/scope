.. scope documentation master file, created by
   sphinx-quickstart on Wed Dec 18 14:01:27 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

scope documentation
===================

SCOPE - Statistical Confidence of Oscillatory Processes with EMD (Empirical Mode Decomposition)

SCOPE is the project to detect quasi-periodicities in the solar atmosphere using the EMD technique. These oscillatory signals are typically accompanied by a combination of white noise and coloured noise with power law spectral dependence. To detect quasi-periodicities, we compute the EMD spectrum containing EMD-revealed modes and the confidence limits of modal energy. This allows us to identify the significant mode beyond the confidence limits, which is expected to be associated with the quasi-periodic oscillatory signal of interest.

The project consists of three main parts:

1. Estimation of power law index and noise energy using a debiased least squares fit of Fourier spectrum based on the method given by `Vaughan (2005) <https://www.aanda.org/articles/aa/abs/2005/07/aa1453/aa1453.html>`__.
2. Extraction of the dominant period of EMD-revealed modes from the global wavelet spectrum produced by the `Wavelet analysis package <https://github.com/ct6502/wavelets>`__.
3. Plot of EMD spectrum and calculation of confidence limits using the method proposed by `Kolotkov et al. (2016) <https://www.aanda.org/articles/aa/full_html/2016/08/aa28306-16/aa28306-16.html>`__. 

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   reference/index 
   tutorial/index
   theory/index


