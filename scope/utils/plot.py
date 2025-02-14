import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def plot_fft_spectrum(fit_fft):
    """
    Plots the FFT spectrum from fitting results, displaying the power, the model fit,
    and confidence limits.

    Parameters
    ----------
    fit_fft : dict
        A dictionary containing the results of FFT fitting with the following keys:
            - 'frequency' : numpy array
                Frequencies obtained from the FFT fitting.
            - 'power' : numpy array
                Power spectrum values corresponding to each frequency.
            - 'pl_index' : float
                Power-law index of the colored noise model.
            - 'pl_index_stderr' : float
                Standard error of the power-law index.
            - 'expectation_continuous' : numpy array
                Expected power values from the model fit to the data.
            - 'confidence_limit' : numpy array
                Confidence limit values for the power spectrum.
            - 'conf_prob' : float
                Confidence probability associated with the confidence limit.

    Returns
    -------
    None
        This function does not return any values. It displays a log-log plot of the
        power spectrum, including the model fit and confidence limits.

    Notes
    -----
    The function plots the period (1/frequency) on the x-axis and the Fourier magnitude
    on the y-axis. It uses a log-log scale for both axes to better visualize the
    frequency spectrum over a wide range. The power-law index with its error is included
    in the legend, as well as the confidence probability of the confidence limits.

    """

    # Convert frequency to period
    period = 1 / fit_fft['frequency']

    # coloured noise index
    alpha = fit_fft['pl_index']
    alpha_stderr = fit_fft['pl_index_stderr']

    # confidence probability
    prob = fit_fft['conf_prob']

    # energy relation
    e_rel = fit_fft['white_energy'] / fit_fft['color_energy']

    plt.loglog(period, fit_fft['power'], linewidth=2)
    plt.loglog(period, fit_fft['expectation_continuous'],
               label=rf'$\alpha$ = {alpha:.2f} +/- {alpha_stderr:.2f} ',
               color='blue', linewidth=3)
    plt.loglog(period, fit_fft['confidence_limit'], label=f'{prob*100:.0f}%',
               color='red', linewidth=3)
    plt.title(fr'FFT Spectrum ($E_w / E_c =${e_rel:.2f})')
    plt.xlabel('Period [a.u.]')
    plt.ylabel('Fourier Magnitude [a.u.]')
    plt.legend()
    plt.show()


def plot_signal(t, x, title='Input signal', ax=None):
    """
    Plots the input signal over time.

    Parameters
    ----------
    t : numpy array
        Time values for the signal, used as the x-axis in the plot.
    x : numpy array
        Signal values corresponding to each time point in `t`, used as the y-axis in the plot.
    title : string
        Tittle of the plot
    ax : matplotlib.axes.Axes
        Axis to alight the plot with other plots

    Returns
    -------
    None
        This function does not return any values. It displays a plot of the input signal
        as a function of time.

    Notes
    -----
    This function produces a simple 2D line plot, showing how the signal varies over time.
    The x-axis represents time, and the y-axis represents the signal magnitude.

    """
    if ax is None:
        fig, ax = plt.subplots(1)
        
    ax.plot(t, x)
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title(title)
    plt.show()


def plot_modes(t, modes):
    """
    Plots the modes obtained using Empirical Mode Decomposition (EMD) as a series
    of subplots, each representing one mode over time.

    Parameters
    ----------
    t : numpy array
        Time values for each mode, used as the x-axis in each subplot.
    modes : numpy array, shape (n_samples, n_modes)
        2D array where each column represents a mode obtained from EMD, with
        `n_samples` time points and `n_modes` modes.

    Returns
    -------
    None
    This function does not return any values. It displays a series of subplots,
    with each subplot representing one mode as a function of time.

    Notes
    -----
    Each mode is plotted in its own subplot, aligned vertically in a single column.
    The x-axis represents time, while the y-axis shows the amplitude of each mode.
    The y-limits of each subplot are set based on the global minimum and maximum
    across all modes, ensuring consistent scaling.
    """

    num_modes = modes.shape[1]

    global_min, global_max = np.min(modes), np.max(modes)

    # Set the figure size to increase the width of the plot
    plt.figure(figsize=(10, 2 * num_modes))

    for i in range(num_modes):
        plt.subplot(num_modes, 1, i + 1)
        plt.plot(t, modes[:, i])
        plt.ylabel(f'Mode {i + 1}')
        plt.ylim(global_min, global_max)

    plt.xlabel('Time')
    plt.tight_layout()
    plt.show()


def plot_emd_spectrum(emd_sp, cutoff_period, conf_period=None, conf_up=None,
                      conf_down=None, conf_mean=None, fap=None):
    """
    Plots the Empirical Mode Decomposition (EMD) energy spectrum along with optional 
    confidence intervals and significant mode markers.
    
    Parameters
    ----------
    emd_sp : dict
        Dictionary containing the EMD spectrum data with the following keys:
            - 'period': numpy array, periods of the EMD modes.
            - 'energy': numpy array, energy values of the EMD modes.
            - 'period_err': numpy array, uncertainties in the period values.
    cutoff_period : float
        The cutoff period value, shown as a vertical dashed line on the plot.
    conf_period : numpy array, optional
        Period values corresponding to the confidence intervals. If provided, confidence
        intervals are plotted.
    conf_up : numpy array, optional
        Upper confidence boundary corresponding to `conf_period`.
    conf_down : numpy array, optional
        Lower confidence boundary corresponding to `conf_period`.
    conf_mean : numpy array, optional
        Mean confidence boundary corresponding to `conf_period`.
    fap : float, optional
        False alarm probability. If provided, used to calculate the significance 
        confidence level (e.g., 95% confidence corresponds to `fap=0.05`) and label the
        confidence interval.

    Returns
    -------
    None
        This function does not return any values. It displays a log-log plot of the 
        EMD spectrum.

    Notes
    -----
    - The EMD spectrum is plotted as points with error bars for period uncertainties.
    - A vertical dashed line indicates the `cutoff_period`.
    - Confidence intervals (if provided) are plotted as red lines (upper and lower bounds)
      and a blue line (mean, if available).
    - Significant modes, determined by comparing `emd_sp['energy']` against `conf_up`, 
      are highlighted in green.
    - The x-axis (period) and y-axis (energy) are plotted on a logarithmic scale.
    - Gridlines and a legend enhance plot readability.

    Example
    -------
    ```python
    emd_sp = {
        'period': [0.5, 1.0, 2.0, 4.0],
        'energy': [0.01, 0.1, 0.5, 2.0],
        'period_err': [0.05, 0.1, 0.2, 0.4]
        }
    cutoff_period = 1.5
    conf_period = [0.5, 1.0, 2.0, 4.0]
    conf_up = [0.02, 0.15, 0.6, 2.5]
    conf_down = [0.005, 0.05, 0.4, 1.5]
    plot_emd_spectrum(emd_sp, cutoff_period, conf_period, conf_up, conf_down)
    ```
    """

    plt.errorbar(emd_sp['period'], emd_sp['energy'], xerr=emd_sp['period_err'],
                 label='EMD Spectrum', fmt='.', color='orange',
                 ms=15, capsize=5, mew=2)
    plt.axvline(x=cutoff_period, color='black', linestyle='dashed')

    if fap is not None:
        prob = 1 - fap
        label = f'{prob*100:.0f}%'
    else:
        label = ''

    if conf_period is not None:
        if conf_up is not None:
            plt.plot(conf_period, conf_up, color='red', label=label)

            # get upper boundaries for the periods of the EMD modes found
            
            if (emd_sp['period'][0] < conf_period[0]):
                conf_period[0] = emd_sp['period'][0] 
            
            f = interp1d(conf_period, conf_up)
            upper_bounds = f(emd_sp['period'])
            mask = emd_sp['energy'] > upper_bounds

            # mark signigicant EMD modes
            plt.errorbar(emd_sp['period'][mask], emd_sp['energy'][mask],
                         xerr=emd_sp['period_err'][mask],
                         label='Signigicant modes', fmt='.', color='green',
                         ms=15, capsize=5, mew=2)

        if conf_down is not None:
            plt.plot(conf_period, conf_down, color='red')
        if conf_mean is not None:
            plt.plot(conf_period, conf_mean, color='blue')

    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-3, 1e1)
    plt.xlim(0.1, 60)
    plt.title('EMD Spectrum')
    plt.xlabel('Period [a.u.]')
    plt.ylabel('EMD Modal Energy [a.u.]')
    plt.legend()
    plt.grid()
    plt.show()
