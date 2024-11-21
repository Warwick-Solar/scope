# confEMD
Empirical Mode Decomposition (EMD) technique for identifying quasi-periodic oscillations in the solar atmosphere from superimposed background noises.

## Project description
confEMD is the project to detect quasi-periodicities in the solar atmosphere using the EMD technique. These oscillatory signals are typically accompanied by a combination of white noise and coloured noise with power law spectral dependence. To detect quasi-periodicities, we compute the EMD spectrum containing EMD-revealed modes and the confidence limits of modal energy. This allows us to identify the significant mode beyond the confidence limits, which is expected to be associated with the quasi-periodic oscillatory signal of interest.

The project consists of three main parts:
- Estimation of power law index and noise energy using a debiased least squares fit of Fourier spectrum based on the method given by [Vaughan (2005)](https://doi.org/10.1051/0004-6361:20041453).
- Extraction of the dominant period of EMD-revealed modes from the global wavelet spectrum produced by the Wavelet analysis package from https://github.com/ct6502/wavelets.
- Plot of EMD spectrum and calculation of confidence limits using the method proposed by [Kolotkov et al. (2016)](https://doi.org/10.1051/0004-6361/201628306).


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites
- Leave blank

## Installation
 - Leave blank

## Example
The sample signal in this example consists of an oscillatory component, a decaying trend and some random noise signals obeying power law:
![](./use-case/)

After setting the mean of the signal to zero, we apply EMD to obtain the first set of intrinsic mode functions (IMFs):
![](./use-case/)

The trend of the signal is estimated using the 'emd_trend' function. This function identifies modes with periods exceeding a fraction of the total signal duration (denoted by the 'cutoff' parameter) as the trend of the signal. The (dominant) period of a mode is estimated by fitting the global wavelet spectrum of the mode with a Gaussian + Quadratic function, conducted in the 'emd_period_energy' function. An example of the global wavelet spectrum fit is shown below:
![](./use-case/)
We can see that for each mode there is a Gaussian-like peak associated with the dominant period.

The position and standard deviation of the Gaussian peak refers to the dominant period and the uncertainty of this estimation. For this simple example, the trend of the signal is simply the residual of EMD:
![](./use-case/)

Hence the detrended signal is:
![](./use-case/)

Now we can estimate the noise parameters of the actual part of the signal we are interested in, the detrended signal, by the 'fit_fourier' function. In the 'fit_fourier' function, we fit the FFT spectrum by a power law model in log-log scale to extract the power law index and noise energy of the signal. Firstly, we must note that each point of the Fourier power $I(f_{j})$ follows a chi-square distribution with 2 degrees of freedom, denoted as:
```math
I(f_{j}) = \mathcal{P}(f_{j}) \chi_{2}^{2}/2
```
where $\mathcal{P}(f_{j})$ is the true power spectrum, and $\chi_{2}^{2}$ is a random variable distributed as $\chi^{2}$ with 2 degrees of freedom. Since the least squares method assumes that each point is Gaussian-distributed, we cannot directly apply this method in the FFT spectrum fitting. Instead, we should consider the mean of the $\chi_{2}^{2}/2$ term. In log scale, $\left\langle \mathrm{log}(\chi^{2}_{2}/2) \right\rangle$ = -0.25068. This term corresponds to the bias that will be introduced to the fitting if we directly implement the least squares method. Hence, we shall include this term in the model function such that the least squares fitting will not be 'biased'. 

The FFT spectrum of the detrended signal below shows a combination of white and coloured noise, with the power law index of this coloured noise being 1.13$\pm$0.32. The peak outside the 95% confidence level is expected to be the period of the oscillatory signal.
![](./use-case/)

The EMD energy spectrum is plotted by the 'emd_energy_spectrum' function. For this example:
![](./use-case/)

With the power law index and noise energy, we can compute the confidence limits of the EMD energy spectrum. [Flandrin et al. (2004)](https://ieeexplore.ieee.org/document/1261951) and [Wu and Huang (2004)](https://royalsocietypublishing.org/doi/10.1098/rspa.2003.1221) investigate the dyadic property of EMD and suggest the following relation between modal energy and modal period:
```math
E_{m}P_{m} = \text{const.}
```
[Kolotkov et al. (2016)](https://doi.org/10.1051/0004-6361/201628306) suggests that the modal energy of the mth IMF should have a chi-square distribution with the $k$ degrees of freedom (DoF). We thus estimate the confidence limits using the percent-point function of the chi-square distribution. Here we use the false alarm probability = 0.05. The 'emd_noise_conf' function generates 500 (by default) noise samples with the same power law index and energy as the input and conducts the EMD. It extracts the dominant period and modal energy for each IMF by calling the 'emd_period_energy' function. The 'emd_noise_fit' function fits the chi-square distribution to the histogram of modal energy for each mode number to extract the mean energy and $k$. We obtain the mean period, mean energy and number of DoF for each mode number. Due to the dyadic property of EMD, we expect both mean energy vs mean period and $k$ vs mean period are linear in log-log scale. The exact linear relationship is found by fitting a straight line. By obtaining this linear relationship, we use it to generate 500 data points of the confidence limits over the whole range of period. 
![](./use-case/)




Place here your tutorial base on your emd example

To write any code, use this snippet
```python
import numpy as np

array = np.array([0, 1, 2)]
```
To insert you plots, use the command below (use should place your plot into the use-case folder):
![](./use-case/sun.jpg)

## Contributing
Leave blank

## Acknowledgements
Leave blank

## License
This project is licensed under the Apache 2.0. License - see the [LICENSE.md](./LICENSE) file for details


