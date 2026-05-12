import numpy as np
import matplotlib.pyplot as plt
from scope.emd import emd_modes, emd_energy_spectrum
from scipy.signal import savgol_filter


def tune_emd(x, n_sh=50, min_sh=5e-8, max_sh=1e-4, trend_scale=0.4, criterion='wape', cont_thresh=0.001, show=False):
    """
    Returns a set of EMD modes for a given time series data.

    This function iteratively applies empirical mode decomposition (EMD) to the input time series data, tuning the sifting factor across a specified range to optimise the 
    extraction of modes based on a chosen accuracy criterion. The process continues until the residual energy of the remaining signal falls below a defined threshold.

    Parameters
    ----------
    x : numpy array
        Time series data
    n_sh : int, optional
        Number of shift factors to test (default is 50)
    min_sh : float, optional
        Minimum shift factor to test (default is 5e-8) 
    max_sh : float, optional
        Maximum shift factor to test (default is 1e-4)
    trend_scale : float, optional
        Relative scale, as a fraction of the total signal length, above which modes are considered to be trend-like (default is 0.4)
    criterion : str, optional
        Accuracy criterion to determine the best mode among candidates for each shift factor. Options are 'wape' (default), 'nrmse', 'smape', and 'mase'. If an invalid
        criterion is provided, defaults to 'wape' with a warning.
    cont_thresh : float, optional
        Threshold for residual energy, as a fraction of original signal variance, below which the decomposition process will stop (default is 0.001)
    show : bool, optional
        Whether to show plots of the residual and determined mode at each iteration, as well as the final EMD spectrum and modes (default is False)
    
    Returns
    -------
    modes : numpy array
        2D array where each column corresponds to a determined mode, sorted by period, with the final column representing the residual
    
    """
    x = x - np.mean(x)
    x_original = x.copy()

    # Verifying valid input for accuracy criterion
    criterion = criterion.lower()
    if criterion not in ('nrmse', 'smape', 'mase', 'wape'):
        print("Invalid input entered for criterion. Defaulting to WAPE for this run. Please check our documentation for valid criterion compatible with this function to date.")
        criterion = 'wape'

    if show:
        # Calculate initial y-axis limits for consistent plotting across iterations
        initial_x_min = np.min(x)
        initial_x_max = np.max(x)
        if initial_x_max == initial_x_min: # Handle constant signal case
            plot_ymin = initial_x_min - 0.1 # A small default range
            plot_ymax = initial_x_max + 0.1
        else:
            padding = (initial_x_max - initial_x_min) * 0.05 # 5% padding
            plot_ymin = initial_x_min - padding
            plot_ymax = initial_x_max + padding

    shiftfactors = np.logspace(np.log10(min_sh), np.log10(max_sh), n_sh)

    result = []

    threshold = cont_thresh
    var_orig = np.std(x) ** 2

    for k in range(101):
        successful_modes = []  # List of tuples (accuracy criterion, mode, sh_factor) for current iteration

        print(f"##### NOW RUNNING ITERATION: {k+1} #####")
        print("")

        for i in range(n_sh):
            sh_factor = shiftfactors[i]
            try:
                print(f"Shift factor {i+1}: {sh_factor:.2e}")

                modes_attempt = emd_modes(x, sd_thresh=sh_factor)

                if modes_attempt is None or (modes_attempt.ndim > 0 and modes_attempt.shape[1] == 0):
                    continue

                if modes_attempt.ndim == 1:
                    modes_attempt = modes_attempt[:, None]

                candidates = []

                # Use chosen accuracy criterion to determine which attempt most closely resembles the current state of the data
                if criterion == 'wape':
                    for j in range(modes_attempt.shape[1]):
                        if np.sum(np.abs(x)) == 0:  # Avoid division by zero
                            wape_val = np.inf
                        else:
                            wape_val = np.sum(np.abs(x - modes_attempt[:, j])) / np.sum(np.abs(x))
                        candidates.append(wape_val)
                elif criterion == 'nrmse':
                    for j in range(modes_attempt.shape[1]):
                        if np.sum(np.abs(x)) == 0:
                            nrmse_val = np.inf
                        else:
                            nrmse_val = np.linalg.norm(x - modes_attempt[:, j]) / np.linalg.norm(x)
                        candidates.append(nrmse_val)
                elif criterion == 'smape':
                    for j in range(modes_attempt.shape[1]):
                        if np.sum(np.abs(x)) == 0:
                            smape_val = np.inf
                        else:
                            epsilon = 1e-10
                            smape_val = np.mean(2.0 * np.abs(x - modes_attempt[:, j]) / (np.abs(x) + np.abs(modes_attempt[:, j]) + epsilon))
                        candidates.append(smape_val)
                elif criterion == 'mase':
                    for j in range(modes_attempt.shape[1]):
                        if np.sum(np.abs(x)) == 0:
                            mase_val = np.inf
                        else:
                            naive_errors = np.abs(x[1:] - x[:-1])
                            scale = np.mean(naive_errors)
                            mase_val = np.mean(np.abs(x - modes_attempt[:, j])) / scale
                        candidates.append(mase_val)

                if not candidates or np.all(np.isinf(candidates)):
                    print(f"Sift failed for shift factor {sh_factor:.2e}. All candidates were inf.")
                    print("")
                    continue

                mode_ind = np.argmin(candidates)
                best_mode_sh = modes_attempt[:, mode_ind]
                best_error_sh = candidates[mode_ind]

                successful_modes.append((best_error_sh, best_mode_sh, sh_factor))
                print("Sift successful")
                print("")

            except Exception as e:
                print("")
                continue

        # After trying all shift factors for the current outer iteration (k)
        if not successful_modes:  # If no valid modes were found for any shift factor
            print(f"No valid modes found for iteration {k+1}. Breaking outer loop.")
            break

        # Find the overall best mode among the successfully extracted ones in this iteration
        successful_modes.sort(key=lambda item: item[0]) # Sort by accuracy criterion
        best_error, selected_mode, selected_sh_used = successful_modes[0]

        # Apply running average smoothing only for the first iteration (k=0)
        if k == 0:
            window_size = int(trend_scale * len(x_original))

            if window_size < 5:
                window_size = 5
            if window_size % 2 == 0:
                window_size += 1

            selected_mode = savgol_filter(
                selected_mode,
                window_length=window_size,
                polyorder=2,
                mode='interp'
            )

            print("Applied Savitzky-Golay smoothing to first mode.")

        result.append(selected_mode)

        if show:
            plt.figure()
            if k != 0:
                plt.plot(x_original, alpha=0.2, label="Original data")
            plt.plot(x, label="Residual data")
            plt.plot(selected_mode, label="Determined mode")
            plt.xlabel("Time")
            plt.ylabel("Amplitude")
            plt.title(f"Iteration {k+1}: Residual and Determined Mode (Shift factor: {selected_sh_used:.2e})")
            plt.legend()
            plt.ylim(plot_ymin, plot_ymax) # Set consistent y-axis limits
            plt.show()

        x = x - selected_mode

        residual_energy = np.std(x) ** 2 / var_orig if var_orig != 0 else 0

        print("")
        print("Residual energy remaining signal:",residual_energy)
        print("Shift factor correlating to this iteration's determined mode:",selected_sh_used)
        print("")
        print("########################################")
        print("")
        if residual_energy < threshold:
            print(f"Residual energy ({residual_energy:.2e}) is below threshold ({threshold}). Ending signal decomposition.")
            break

    # Take the final remaining data as the residual
    residual = x

    # Final processing of results
    nm = len(result)
    nt = len(x_original) # Use original length for modes array
    t = np.arange(nt)

    # Initialize modes array. Handle case where result might be empty.
    modes = np.zeros((nt, nm)) if nm > 0 else np.array([]).reshape(nt, 0)

    for i in range(nm):
        modes[:, i] = result[i]

    print("")
    print(f"Total number of modes determined: {nm}")

    # Sorting modes by period (periods estimated using FFT)
    periods = []
    for mode in modes.T: # Iterate over each column (mode)
        fft = np.fft.rfft(mode)
        freq = np.fft.rfftfreq(len(mode), d=t[1]-t[0])
        power = np.abs(fft) ** 2
        peak_freq = freq[np.argmax(power)]
        # Handle case where peak_freq might be zero to avoid division by zero
        if peak_freq == 0:
            periods.append(np.inf) # Assign infinity for zero frequency (infinite period)
        else:
            periods.append(1 / peak_freq)
    sorted_indices = np.argsort(periods)
    modes = modes[:, sorted_indices]

    # Adding the residual as the final mode
    modes = np.hstack((modes, residual[:, None]))

    if show:
        print("")
        print("EMD power spectrum produced for modes determined in this analysis:")
        sp = emd_energy_spectrum(modes, t, plot_fitting=False)
        plt.errorbar(sp['period'], sp['energy'], xerr=sp['period_err'],
                 label='Detected modes', fmt='.', color='orange',
                 ms=15, capsize=5, mew=2)
        plt.axvline(x=trend_scale*len(x_original), color='black', linestyle='dashed', label='Trend scale cutoff period')
        plt.xscale('log')
        plt.yscale('log')
        plt.title('EMD Spectrum')
        plt.xlabel('Period [a.u.]')
        plt.ylabel('EMD Modal Energy [a.u.]')
        plt.legend()
        plt.grid()
        plt.show()

        print("")
        print("List of modes determined in this analysis sorted by period + residual")
        num_modes = modes.shape[1]
        global_min, global_max = np.min(modes), np.max(modes)
        plt.figure(figsize=(10, 2 * num_modes))
        for i in range(num_modes):
            plt.subplot(num_modes, 1, i + 1)
            plt.plot(t, modes[:, i])
            if i==num_modes-1:
                plt.ylabel('Residual')
            else:
                plt.ylabel(f'Mode {i + 1}')
            plt.ylim(global_min, global_max)
        plt.xlabel('Time')
        plt.tight_layout()
        plt.show()

    return modes