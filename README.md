# `scope`
<img src="./docs/source/_static/scope_logo.png" alt="Scope Logo" width="300">

`scope` - Statistical Confidence of Oscillatory Processes with EMD (Empirical Mode Decomposition).

## Project description
`scope` is the Python-based package for detecting oscillatory signals in observational or experimental time series with the EMD technique and assessing their statistical significance vs. power-law distributed background noise. Oscillatory processes in real data sets of various origins are often contaminated by a combination of white and coloured noise with a power-law spectral dependence, so that the EMD-revealed intrinsic mode functions need to be rigorously tested against the periodic components generated by noise. To do so, we compute the _EMD energy spectrum_ containing the total energy and dominant period of each EMD-revealed intrinsic mode and the noise confidence limits for modal energy. This allows us to identify the significant mode(s) with the energy beyond the confidence limits, which is expected to be of a non-noise origin and associated with a quasi-periodic oscillatory process of interest. The developed package does not assume the physical origin of the input data set, making it readily applicable for analysing oscillatory processes across various fields of science and industry.

The project consists of the following main parts:
- Perform EMD analysis of the original time series and reveal the set of intrinsic modes using [Empirical Mode Decomposition in Python](https://emd.readthedocs.io/en/stable/) ([Quinn et al. 2021](https://doi.org/10.21105/joss.02977)) package. See function `emd_modes` from `emd` module in `scope`.
- Estimation of the power-law index and energy of superimposed background noise from the Fourier power spectrum as described in [Vaughan (2005)](https://doi.org/10.1051/0004-6361:20041453). See function `fit_fourier` from `fourier` module in `scope`.
- Estimation of the dominant period of each EMD-revealed intrinsic mode from the global wavelet spectrum produced with [Torrence & Compo Wavelet Analysis Software](https://github.com/ct6502/wavelets) ([Torrence & Compo 1998](https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf)) package. See function `emd_period_energy` from `emd` module in `scope`.
- Calculate EMD confidence limits using the method proposed by [Kolotkov et al. (2016)](https://doi.org/10.1051/0004-6361/201628306). See function `emd_noise_conf` from `emd` module in `scope`.
- Visualise the results with `plot_signal`, `plot_modes`, `plot_fft_spectrum`, and `plot_emd_spectrum` functions from `utils` module.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Example](#example)
- [Description of Functions](#functions)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Prerequisites
Python &ge; 3.8

## Installation
 - Leave blank

## Example
<details>
 <summary>Click to expand</summary>

The example described below is provided in the `emd_example.py` file.

The sample signal in this example consists of an oscillatory component, an exponentially decaying trend and a combination of white and coloured noise obeying the power law: \
![](./docs/source/_static/input_signal.png)

After setting the mean of the input signal to zero, we apply EMD to obtain the set of intrinsic mode functions (IMFs):
```python
modes = emd_modes(x, sd_thresh=1e-4)
plot_modes(t, modes)
```
where the 'sd_thresh' parameter is the threshold at which the sift of each IMF stops. In our example, we obtained seven EMD modes, six of which are oscillatory IMFs and one is a non-oscillatory residual (usually, the number of EMD modes is about $$\log_2(N)$$ where $$N$$ is the number of data points in the input signal).
![](./docs/source/_static/1st_EMD.png)

The empirical trend of the signal is estimated using the `emd_trend` function. This function identifies modes with periods exceeding a fraction of the total signal duration (denoted by the 'cutoff' parameter) and the residual, combines them into an empirical trend of the input signal, and returns a new set of modes in which all modes have periods shorter than the cutoff and the last mode represents the signal's trend. This cutoff is set to 0.4 of the total signal length by default, which means that a mode with less than 2.5 oscillation cycles is considered as part of the empirical trend. 
```python
modes = emd_trend(modes, t)
trend_emd = modes[:, -1]
plot_signal(t, trend_emd, 'Trend of the signal')
```
For our example, the empirical trend of the signal is found to form by the last EMD mode (the residual) only: \
![](./docs/source/_static/trend_signal.png)

Hence, the detrended signal is: \
![](./docs/source/_static/detrended_signal.png)

Now we can estimate the parameters of superimposed noise by applying the `fit_fourier` function to the detrended signal. The function returns the FFT spectrum of the detrended signal best-fitted by a power-law model, with powers of white (if present) and coloured noise and the power-law index of coloured noise as model parameters. For our example, the FFT spectrum shows a combination of white and coloured noise components in the detrended signal, with the power-law index of coloured noise being 1.1±0.3 and the ratio of the white to coloured noise energies about 0.3. The `fit_fourier` function also estimates the confidence interval of a given value (e.g. 95%, false alarm probability = 0.05). The Fourier peaks outside this confidence interval are attributed to statistically significant oscillatory processes of non-noise origin.
```python
fit_fft = fit_fourier(x, dt, fap=0.05)
plot_fft_spectrum(fit_fft)
```
![](./docs/source/_static/FFT_spectrum.png)

The EMD energy spectrum, i.e. the relationship between the EMD modal energy vs. dominant oscillation period for the set of EMD modes identified in the original signal, is computed by the `emd_energy_spectrum` function:
```python
emd_sp = emd_energy_spectrum(modes, t)
cutoff_period = 0.4 * len(x) * dt #show cutoff period
plot_emd_spectrum(emd_sp, cutoff_period)
```
![](./docs/source/_static/emd_spectrum.png) 

The vertical dashed line corresponds to the cutoff period adopted in the `emd_trend` function; all modes beyond this line are considered as components of trend.

With the power-law index and noise energy returned by the `fit_fourier` function, we can compute the confidence limits of the EMD energy spectrum using the `emd_noise_conf` function (separately for coloured noise and, if present, white noise):
```python
# false alarm probability
fap = 0.05
#Confidence limits for coloured noise
conf_c = emd_noise_conf(t, alpha=alpha, period_min=2*dt, 
                        period_max=N*dt, num_samples=500, 
                        signal_energy=fit_fft['color_energy'], fap=fap)
#Confidence limits for white noise
if fit_fft['white_energy'] > 0: # check if there is only colored noise model
    conf_w = emd_noise_conf(t, alpha=0, period_min=2*dt,
                            period_max=N*dt, num_samples = 500, 
                            signal_energy=fit_fft['white_energy'], fap=fap)
```
Here, the false alarm probability (fap) is set to 0.05 (95% confidence). The `emd_noise_conf` function generates 500 independent noise samples with the same power law index ('alpha') and energy ('signal_energy') as the input. The other two parameters, 'period_min' and 'period_max', set the range of periods over which the confidence limits are computed. Combining the upper and lower confidence limits for white and coloured noise compenents,
```python
#Upper confidence limit for the combined noises
conf_up = conf_c['up'] + conf_w['up']

#Lower confidence limit for the combined noises
conf_down = conf_c['down'] + conf_w['down']
```
and visualising the EMD energy spectrum with confidence,
```python
# plot emd spectrum
plot_emd_spectrum(emd_sp, cutoff_period, conf_period, conf_up, conf_down, conf_mean, fap)
```
it becomes\
![](./docs/source/_static/emd_spectrum_with_conf.png) 

Here, 'conf_mean' stands for the expected mean value of noise energy (`conf_mean = conf_c['mean_energy'] + conf_w['mean_energy']`) and 'conf_period' (`conf_period = conf_c['period']`) is the array of oscillation periods over which the confidence limits are computed.
The EMD modes beyond the confidence limits are considered as significant, that are not likely to be caused by random noise. In our example, only one mode is found to be significant which seems consistent with the input oscillatory component of the original signal.

![](./docs/source/_static/significant_mode.png) 

</details>

## Functions 

<details>
 <summary>Click to expand</summary>
 
### 'emd_period_energy'
As mentioned in the example section, the total energy and (dominant) period of each EMD mode are required for constructing an EMD energy spectrum. The total modal energy is estimated by summing up squares of instantaneous amplitudes of each EMD mode. The dominant period of each EMD mode is estimated by best-fitting the global wavelet spectrum ([Torrence & Compo 1998](https://psl.noaa.gov/people/gilbert.p.compo/Torrence_compo1998.pdf)) of the mode with a Gaussian + Parabolic function, performed in the `emd_period_energy` function. The position and standard deviation of the Gaussian peak are used for the dominant EMD modal period and the uncertainty of this estimation.
<!--
An example of the global wavelet spectrum fit is shown below: \
![](./docs/source/_static/fit_mode.png)
We can see that for each mode there is a Gaussian-like peak associated with the dominant period.
--> 

### 'fit_fourier'
In the `fit_fourier` function, we fit the FFT spectrum by a power-law model in log-log scale to extract the power-law index and energy of the noise component of the signal. Firstly, we must note that, at each Fourier frequency, the Fourier power $$I(f_{j})$$ follows a chi-squared distribution with 2 degrees of freedom, denoted as:

$$I(f_{j}) = \mathcal{P}(f_{j}) \chi_{2}^{2}/2$$

where $\mathcal{P}(f_{j})$ is the true power spectrum, and $\chi_{2}^{2}$ is a random variable distributed as $\chi^{2}$ with 2 degrees of freedom. Since the least squares method assumes that the input data set is Gaussian-distributed, we cannot directly apply this method to best-fit the FFT power spectrum. Instead, we should consider the mean of the $$\chi_{2}^{2}/2$$ term. In log scale, $$\left\langle \mathrm{log}(\chi^{2}_{2}/2) \right\rangle$$ = -0.25068 ([Vaughan (2005)](https://doi.org/10.1051/0004-6361:20041453)). This term corresponds to the bias that will be introduced to the fitting if one directly implements the least squares method. Hence, we shall include this term in the model function such that the least squares fitting will not be 'biased'. We also note that the value of this bias term is independent of the choice of normalisation of the FFT power spectrum.

<!--
Additionally, we can visualise this bias factor. Since the Fourier power follows a chi-square distribution with 2 DoF is essentially an exponential function, we consider the integration of an exponential function over the entire range of power, which gives a constant value:
```math
\int_{0}^{P_{\mathrm{max}}} e^{-x} dx = const.
```
One can transform it to log scale by considering a new variable $τ = lnx$. Hence, the new integral becomes:
```math
\int_{-\infty}^{\mathrm{ln} P_{\mathrm{max}}} e^{-e^{\tau}} e^{\tau} d\tau = const.
```
where $F(\tau) = e^{-e^{\tau}} e^{\tau}$ is the distribution of the Fourier power in log scale. By plotting this function, we see an asymmetric distribution with its mean positioned at -0.25068.
![](./docs/source/_static/bias_visualisation.png)
--> 

The power law model we used in the `fit_fourier` function is a superposition of white and coloured noise components, given by:

$$ \mathcal{P}(f) = \mathcal{P}_{c}(f) + \mathcal{P}_{w}(f) = Z_{c} f^{-\alpha_{c}} + Z_{w},$$

where $Z_{c}$ and $Z_{w}$ are the proportionality constants of coloured and white noises, respectively, and $\alpha$ is the power law index of coloured noise. After obtaining the proportionality constants from the debiased least squares fit, we can estimate the energy of each noise type using:

$$\text{Energy} = N \cdot nf \cdot Z_{c/w},$$

where $N$ is the number of data points in the time series and $nf$ is the number of Fourier frequencies, which does not include 0 Hz and the Nyquist frequency. And estimate the confidence limit for a given false alarm probability (fap) as:

$$ .. $$

### 'emd_noise_conf'
[Flandrin et al. (2004)](https://ieeexplore.ieee.org/document/1261951) and [Wu and Huang (2004)](https://royalsocietypublishing.org/doi/10.1098/rspa.2003.1221) investigate the dyadic property of EMD and suggest the following relation between modal energy and modal period:
```math
E_{m}P_{m} = \text{const.}
```
![](./docs/source/_static/mc_emd_spectra.png)

[Kolotkov et al. (2016)](https://doi.org/10.1051/0004-6361/201628306) suggests that the modal energy of the mth IMF should have a chi-square distribution with the $k$ degrees of freedom (DoF). We thus estimate the confidence limits using the percent-point function of the chi-square distribution. Here we use the false alarm probability = 0.05. The 'emd_noise_conf' function generates 500 (by default) noise samples with the same power law index and energy as the input and conducts the EMD. It extracts the dominant period and modal energy for each IMF by calling the 'emd_period_energy' function. The 'emd_noise_fit' function fits the chi-square distribution to the histogram of modal energy for each mode number to extract the mean energy and $k$. We obtain the mean period, mean energy and number of DoF for each mode number. Due to the dyadic property of EMD, we expect both mean energy vs mean period and $k$ vs mean period are linear in log-log scale. The exact linear relationship is found by fitting a straight line. By obtaining this linear relationship, we use it to generate 500 data points of the confidence limits over the whole range of period.

</details>

## Acknowledgements
Leave blank

## License
This project is licensed under the Apache 2.0. License - see the [LICENSE.md](./LICENSE) file for details


