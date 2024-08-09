import numpy as np

from pywt import wavedec, waverec

from scipy.interpolate import interp1d
from scipy.signal import filtfilt, butter, hilbert, sosfiltfilt, windows, medfilt

def butter_filter(arr : np.ndarray, n : int, wn : np.ndarray, filter_type : str, fs : int):

    """
    Performs zero-phase Butterworth filtering.

    Parameters
    ----------
    arr : np.ndarray
        Signal to be filtered.
    n : int
        Order of filter.
    wn : np.ndarray
        Cutoff frequencies.
    filter_type : str
        Type of filter.
        Alternatives are 'lowpass', 'highpass', 'bandpass', and 'bandstop'.
    fs : int
        Sampling rate.
    
    Returns
    -------
    arr_filtered : any
        Filtered signal.

    Examples
    --------
    To filter without normalization.

    .. code-block:: python
        from vitalwave import basic_algos
        basic_algos.butter_filter(arr=nd_ecg, n=4, wn=[0.5, 8], filter_type='bandpass', fs=200)

    \nTo filter with normalization.

    .. code-block:: python
        from vitalwave import basic_algos
        basic_algos.min_max_normalize(basic_algos.butter_filter(arr=nd_ecg, n=4, wn=[0.5, 8],
                                                                filter_type='bandpass', fs=200))
    """

    # Second-order sections.
    sos = butter(n, wn, filter_type, fs=fs, output='sos')
    # Filtering.
    arr_filtered = sosfiltfilt(sos, arr)

    return arr_filtered

def min_max_normalize(arr : np.ndarray, min_val : float = 0.0, max_val : float = 1.0):

    """
    Min-max normalizes array.

    Parameters
    ----------
    arr : np.ndarray
        Signal to be normalized.
    min_val : float
        Minimum value of resulting signal.
    max_val : float
        Maximum value of resulting signal.

    Returns
    -------
    s_norm : np.ndarray
        Normalized version of signal to be normalized.

    Examples
    --------
    To normalize signal-values without using standard-scaler based method.

    .. code-block:: python 
        from vitalwave import basic_algos
        basic_algos.min_max_normalize(arr=nd_ecg)
    """

    s_norm = min_val + (arr - np.nanmin(arr)) * (max_val - min_val) / \
        (np.nanmax(arr) - np.nanmin(arr))

    return s_norm

def resample(timestamps : np.ndarray, arr : np.ndarray, timestamps_new : np.ndarray = None, dt : float = None):
    
    """
    Resamples time series to new time axis.

    Parameters
    ----------
    timestamps : np.ndarray
        Original timestamps.
    arr : np.ndarray
        Original values.
    timestamps_new : np.ndarray
        Timestamps used as basis for resampling.
        Must be in same unit as original timestamps.
        Default is None.
    dt : float
        Timestep of new time series.
        Must be in same unit as original timestamps.
        Default is None.
    
    Returns
    -------
    tuple
        Array of resampled timestamps.
        Array of resampled values.

    Examples
    --------
    To setup new time-based frequency to existing signal.

    .. code-block:: python
       basic_algos.resample(timestamps=ecg_ts, arr=ecg, ts_new=timestamps_new)

    \nOr by proving :math:`dt` time variable.

    .. code-block:: python
       basic_algos.resample(timestamps=ecg_ts, arr=ecg, dt=0.005)
    """

    if timestamps_new is None and dt is None:
        raise ValueError('Either timestamps_new or dt must be given.')

    # Array of new timestamps.
    if timestamps_new is None:
        timestamps_new = np.arange(timestamps[0], timestamps[-1] + dt, dt)
    # Interpolation function.
    f_interp = interp1d(timestamps, arr, kind='cubic', fill_value='extrapolate')
    # Resampled timeseries.
    arr_new = f_interp(timestamps_new)

    return timestamps_new, arr_new

def derivative_filter(arr : np.ndarray, fs : int):
    
    """
    Does derivative filtering according to Pan-Tompkins algorithm.

    Parameters
    ----------
    arr : np.ndarray
        Data to be filtered.        
    fs : int
        Sampling rate.

    Returns
    -------
    arr_filt : np.ndarray
        Filtered data.

    Examples
    --------
    To highlight peaks and valleys of original signal.

    .. code-block:: python
        from vitalwave import basic_algos
        basic_algos.derivative_filter(arr=ecg, fs=200)

    \nExample is linked with moving_average_filter function found in same module.
    """

    # Filter coefficients.
    coeffs = np.array([1, 2, 0, -2, -1]) * (1 / 8) * fs
    # Forward-backward filtering.
    arr_filt = filtfilt(coeffs, 1, arr)

    return arr_filt

def moving_average_filter(arr : np.ndarray, window : int, type : str = 'triang'):
    
    """
    Does moving window integration and moving average.

    Parameters
    ----------
    arr : np.ndarray
        Data to be integrated.
    window : int
        Number of samples in window.
    type : str
        Alternatives are 'triang' and 'moving_avg'.
        Default is 'triang'.

    Returns
    -------
    data : np.ndarray
        Integrated data or moving average data.

    Examples
    --------
    To produce distorted signal highlighting R peak in QRS complex of ECG.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.basic_algos import moving_average_filter
       from vitalwave.example_data import load_biosignal

       import matplotlib.pyplot as plt

       limits = [0, 2000]

       time, ecg = load_biosignal(type="ECG")
       fs = (1 / np.mean(np.diff(time)))

       nd_arr_triang = moving_average_filter(arr=ecg, window=int(fs * 0.15))

       fig, ax = plt.subplots(1, 1, sharex=True)

       start, stop = limits
       ax.plot(time[start:stop], nd_arr_triang[start:stop])
       ax.set_title('Smoothing')
       ax.set_xlabel('Time [s]')

       fig.tight_layout()

       plt.show()

    Example is linked with derivative_filter function found in same module.
    """

    match type:
        case "triang":
            data = np.convolve(arr, windows.triang(window), mode='same')
        case "moving_avg":
            data = np.convolve(arr, np.ones(window), mode='same') / window
        case _:
            data = None

    return data

def wavelet_transform_signal(arr : np.ndarray, dlevels : int, cutoff_low : int, cutoff_high : int, dwt_transform : str = 'bior4.4'):
    
    """
    Designed to work with noisy signals as first-pass mechanism.
    Performs wavelet decomposition on input channel using pywt.wavedec.
    Produces list of coefficients.
    Coefficients in specified ranges are multiplied by zero to remove their contribution.
    Reconstructed signal is returned based on wavelet coefficients.

    Parameters
    ----------
    arr : np.ndarray
        Signal to process.
    dlevels : int
        Wavedeck level parameter.
    cutoff_low : int
        Scale up to which coefficients will be zeroed.
    cutoff_high : int
        Scale from which coefficients will be zeroed.
    dwt_transform : str
        Wavelet transformation function.
        Default is 'bior4.4'.

    Returns
    -------
    wavelet_trans : any
        Corrected signal with inverse wavelet transform.

    Examples
    --------
    To clean up noisy signals prior to processing them with Butterworth bandpass filter.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.basic_algos import butter_filter, min_max_normalize, wavelet_transform_signal
       from vitalwave.example_data import load_biosignal

       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")
       fs = (1 / np.mean(np.diff(time)))

       nd_ecg_denoiced = wavelet_transform_signal(arr=ecg, dwt_transform='bior4.4', dlevels=9,
                                                  cutoff_low=1, cutoff_high=9)

       ecg_filt_cleaned = min_max_normalize(butter_filter(arr=nd_ecg_denoiced, n=4, wn=[0.5, 8],
                                                          filter_type='bandpass', fs=fs))

       fig, axes = plt.subplots(2, 1, sharex=True)
       start, stop = limits

       axes[0].plot(time[start:stop], ecg[start:stop])
       axes[1].plot(time[start:stop], ecg_filt_cleaned[start:stop])

       axes[0].set_title('Filtered ECG')
       axes[1].set_title('wavedeck ')

       axes[1].set_xlabel('Time [s]')
       fig.tight_layout()

       plt.show()
       
       Example is linked with Butterworth bandpass filter found in same module.
    """

    coeffs = wavedec(arr, dwt_transform, level=dlevels)

    # Scale 0 to cutoff_low.
    for ca in range(0, cutoff_low):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])

    # Scale cutoff_high to end.
    for ca in range(cutoff_high, len(coeffs)):
        coeffs[ca] = np.multiply(coeffs[ca], [0.0])

    wavelet_trans = waverec(coeffs, dwt_transform)

    return wavelet_trans

def extract_waveforms(arr : np.ndarray, fid_points : np.ndarray, mode : str, window : int = None):
    
    """
    Extracts waveforms from signal using array of fiducial points.

    Parameters
    ----------
    arr : np.ndarray
        Signal with extracted waveforms.
    fid_points : np.ndarray
        Fiducial points used as basis for extracting waveforms.
    mode : str
        Usage style of fiducial points to extract waveforms.
        Alternatives:
        'fid_to_fid' from one fiducial point to next one.
        'nn_interval' waveform is extracted around each fiducial point by taking half of NN interval before and after.
        'window' waveform is extracted around each fiducial point using window that must then be defined.
    window : int
        Number of samples to take around fiducial points.
        Must be odd because number of samples taken from both sides of window is window / 2.
        Default is None.

    Returns
    -------
    tuple
        Array of extracted waveforms where each row corresponds to one waveform.
        Calculated mean waveform.

    Examples
    --------
    To plot extracted waveforms.

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       time, ecg = load_biosignal(type="ECG")
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import extract_waveforms
       from vitalwave.peak_detectors import ecg_modified_pan_tompkins, msptd

       make_odd = lambda x: x + (x % 2 == 0)

       # Calculate ECG r-peaks.
       ecg_r_peaks = ecg_modified_pan_tompkins(ecg, fs=fs)

       # Calculate ppg peaks and valleys wiht modified smoothed peak detection (MSPTD)
       ppg_msptd_peaks, ppg_msptd_feet = msptd(ppg, fs=fs)

       ppg_wfs, ppg_wfs_mean = extract_waveforms(ppg, ppg_msptd_feet, 'fid_to_fid')
       ecg_wfs1, ecg_wfs1_mean = extract_waveforms(ecg, ecg_r_peaks, 'window', int(make_odd(fs)))
       ecg_wfs2, ecg_wfs2_mean = extract_waveforms(ecg, ecg_r_peaks, 'nn_interval')

       # Plot the waveforms.
       def plot_wfs(wfs, wfs_mean, title):
          fig, ax = plt.subplots()

          for wf in wfs:
             ax.plot(wf, c='tab:blue', alpha=0.2)

          ax.plot(wfs_mean, c='tab:orange', linewidth=2)
          ax.set_title(title)
          fig.tight_layout()
          plt.show()

       plot_wfs(ppg_wfs, ppg_wfs_mean, 'PPG waveforms, feet to feet')
       plot_wfs(ecg_wfs1, ecg_wfs1_mean, 'ECG waveforms, window')
       plot_wfs(ecg_wfs2, ecg_wfs2_mean, 'ECG waveforms, NN interval')
    """

    # Parameter validation.
    if mode == 'window':
        if window is None:
            raise ValueError('Window parameter must be given in window mode.')
        elif window % 2 != 1:
            raise ValueError('Window must be an odd integer.')

    # Max NN interval.
    nn_max = np.max(np.diff(fid_points))
    if mode == 'fid_to_fid':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points) - 1, int(nn_max)), np.nan)
        # Loop through the fiducial points in pairs.
        for i, fds in enumerate(zip(fid_points[:-1], fid_points[1:])):
            waveforms[i, :int(fds[1] - fds[0])] = arr[fds[0]:fds[1]]
    
    elif mode == 'nn_interval':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points) - 2, int(nn_max)), np.nan)
        # Center point of the longest NN interval.
        nn_max_center = nn_max // 2
        # Loop through the fiducial points starting from the second until the second last.
        for i in range(1, len(fid_points) - 1):
            # Number of samples to take from left and right.
            samples_left = (fid_points[i] - fid_points[i - 1]) // 2
            samples_right = (fid_points[i + 1] - fid_points[i]) // 2
            # Place the waveform into the matrix.
            waveforms[i - 1, nn_max_center - samples_left:nn_max_center + samples_right] = \
                arr[fid_points[i] - samples_left:fid_points[i] + samples_right]

        # Remove columns with just NaN values.
        # These columns could happen due to integer divisions used above.
        # This line is just a way to get rid of nanmean's "Mean of empty slice" warning.
        waveforms = waveforms[:, ~np.isnan(waveforms).all(axis=0)]    

    elif mode == 'window':
        # Create an empty array for holding the waveforms.
        waveforms = np.full((len(fid_points), window), np.nan)
        # Center point.
        wf_center = window // 2
        # Loop through the fiducial points.
        for i, fp in enumerate(fid_points):
            # Number of samples to take from left and right.
            samples_left = min(fp, wf_center)
            samples_right = min(len(arr) - fp, wf_center + 1)
            # Place the waveform into the matrix.
            waveforms[i, wf_center - samples_left:wf_center + samples_right] = \
                arr[fid_points[i] - samples_left:fid_points[i] + samples_right]
        
    # Compute the mean waveform.
    mean_waveform = np.nanmean(waveforms, 0)
    
    return waveforms, mean_waveform

def filter_hr(heart_rates : np.ndarray, kernel_size : int = 7, hr_max_diff : int = 16, hr_min : int = 40, hr_max : int = 180):
    
    """
    Filters instantaneous HRs with median filter.

    Parameters
    ----------
    heart_rates : np.ndarray
        Array of instantaneous HRs in bpm.
    kernel_size : int
        Kernel size used in median filter.
        Default is 7.
    hr_max_diff : int
        Maximum allowed HR difference in bpm.
        Default is 16.
    hr_min : int
        Lowest allowed HR in bpm.
        Default is 40.
    hr_max : int
        Highest allowed HR in bpm.
        Default is 180.
    
    Returns
    -------
    heart_rates : np.ndarray
        Filtered instantaneous HRs.

    Examples
    --------
    To calculate initial heart beat validity based on existing set of values.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import butter_filter, filter_hr
       from vitalwave import peak_detectors

       ecg_filt = butter_filter(ecg, 4, [0.5, 25], 'bandpass', fs=fs)
       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg_filt, fs=fs)
       heart_rates = 60 / np.diff(time[ecg_r_peaks])
       heart_rates_filt = filter_hr(heart_rates, 11)

       fig, ax = plt.subplots()
       ax.plot(heart_rates)
       ax.plot(heart_rates_filt)
       ax.set_xlabel('Time [s]')
       ax.set_ylabel('Heart rate [bpm]')
       ax.set_ylim(40, 200)
       fig.tight_layout()
       plt.show()

    Results show normal heart-rate variability along with abnormal.
    """

    # Make a copy to avoid modifying the original array.
    heart_rates = np.copy(heart_rates)
    # Median filtering.
    heart_rates_med = medfilt(heart_rates, kernel_size)
    # Indices to select.
    idxs = (np.abs(heart_rates - heart_rates_med) <= hr_max_diff) & \
        (heart_rates_med >= hr_min) & (heart_rates_med <= hr_max)
    # Mark the rest of the indices to NaNs.
    heart_rates[~idxs] = np.nan

    return heart_rates

def homomorphic_hilbert_envelope(arr : np.ndarray, fs : int, order : int = 1, cutoff_fz : int = 8):
    
    """
    Enhances waveform's envelope.

    Parameters
    ----------
    arr : np.ndarray
        Signal designed for transformation.
    fs : int
        Sampling rate.
    order : int
        Sharpness of transition between passband and stopband.
        Default is 1.
    cutoff_fz : int
        Critical frequency or frequencies of butter filter.
        Default is 8.

    Returns
    -------
    filtered_envelope : np.ndarray
        Filtered envelope of input signal.

    Examples
    --------
    To calculate homomorphic Hilbert envelope in order to produce a low-resolution mock-up signal of original signal.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import homomorphic_hilbert_envelope

       ecg_hilbert = homomorphic_hilbert_envelope(arr=ecg, fs=fs)

       fig, axes = plt.subplots(2, 1, sharex=True)
       start, stop = limits

       axes[0].plot(time[start:stop], ecg[start:stop])
       axes[1].plot(time[start:stop], ecg_hilbert[start:stop])

       axes[0].set_title('ECG-signal')
       axes[1].set_title('with Hilbert Envelope')

       plt.show()

    Result is signal with key features retained from original signal.
    """

    # Apply a zero-phase low-pass 1st order Butterworth filter with a cutoff frequency of 8 Hz.
    b_low, a_low = butter(N = order, Wn = cutoff_fz, fs=fs, btype='lowpass')

    # Calculate the Hilbert envelope of the input signal.
    analytic_signal = hilbert(arr)
    envelope = np.abs(analytic_signal)

    # Apply the low-pass filter to the log of the envelope.
    log_envelope = np.log(envelope)
    filtered_envelope = np.exp(filtfilt(b_low, a_low, log_envelope))

    # Remove spurious spikes in the first sample.
    filtered_envelope[0] = filtered_envelope[1]

    return filtered_envelope

def calculate_time_delay(arr_ecg : np.ndarray, arr_ppg : np.ndarray, peaks_ecg : np.ndarray, fs : int):
    
    """
    Calculates time delay between ECG and PPG signals based on corresponding peaks.

    Parameters
    ----------
    arr_ecg : np.ndarray
        ECG signal.
    arr_ppg : np.ndarray
        PPG signal.
    peaks_ecg : np.ndarray
        Peaks in ECG signal.
    fs : int
        Sampling rate.

    Returns
    -------
    locs_ppg : np.ndarray
        Timestamps of corresponding PPG peaks.

    Examples
    --------
    To syncronize ECG and PPG peaks discovery.

    .. plot::
       :include-source:

       import numpy as np

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       limits = [0, 1000]

       time, ecg = load_biosignal(type="ECG")
       time, ppg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.basic_algos import calculate_time_delay
       from vitalwave.peak_detectors import ecg_modified_pan_tompkins, msptd

       # Calculate ECG R peaks.
       ecg_r_peaks = ecg_modified_pan_tompkins(ecg, fs=fs)

       # Calculate PPG peaks and valleys with modified smoothed peak detection (MSPD).
       ppg_msptd_peaks, ppg_msptd_feet = msptd(ppg, fs=fs)

       locs_ppg = calculate_time_delay(arr_ecg=ecg, arr_ppg=ppg,
                                       peaks_ecg=ecg_r_peaks, fs=int(fs))

       fig, ax = plt.subplots()
       fig.set_size_inches(10, 6)

       ax.plot(time, ppg)

       ax.plot(time[ppg_msptd_feet], ppg[ppg_msptd_feet], 'go')
       ax.plot(time[locs_ppg], ppg[locs_ppg], 'ro')

       ax.set_xlabel('Time [s]')
       ax.set_title("sync ECG with PPG ")

       fig.tight_layout()
       plt.show()

    Results show systolic PPG-peak discovery by using Pan-Tompkins algorithm.
    """

    locs_ecg_corrected = _find_corresponding(arr_ecg, peaks_ecg, 0.5 * fs)
    locs_ppg = _find_corresponding(arr_ppg, locs_ecg_corrected, 0.5 * fs, sym=False)

    return locs_ppg

def _find_corresponding(arr : int, peaks : list, w : int, sym : bool = True):
    
    """
    Finds corresponding peaks in given channel by using certain-sized window.

    Parameters
    ----------
    arr : int
        Signal channel.
    peaks : list
        List of peaks.
    w : int
        Window size.
    sym : bool, optional
        If True, uses a symmetric window.
        If False, uses an asymmetric window.
        Default is True.

    Returns
    -------
    corr_locs : np.ndarray
        Corresponding peaks in array.

    Examples
    --------
    To find peaks corresponding to a given set of peaks.

    .. code-block:: python
        from vitalwave import basic_algos
        basic_algos._find_corresponding(arr=1, peaks=peaks, w=5, sym=True)
    """

    lower_w, upper_w = (int(w / 2), int(w / 2)) if sym else (int(0), int(w))
    corr_locs = []
    for loc in peaks:
        l1, l2 = max(loc - lower_w, 0), min(loc + upper_w, len(arr))
        corr_locs.append(l1 + np.argmax(arr[l1:l2]))
    
    corr_locs = np.array(corr_locs)

    return corr_locs

def segmenting (arr : np.ndarray, window_size : int, overlap : int):
    
    """
    Segments array into overlapping frames.

    Parameters
    ----------
    arr : np.ndarray
        Input array to segment.
    window_size : int
        Size of window.
    overlap : int
        Overlap between segments.

    Returns
    -------
    frames : list
        Segmented frames.
        test.

    Examples
    --------
    To segment signal into overlapping frames.

    .. code-block:: python
        from vitalwave import basic_algos
        basic_algos.segmenting(arr=ecg, window_size=5, w=5, overlap=1)
    """

    frames = []
    step = window_size - overlap

    for i in range(0, len(arr), step):
        frame = arr[i:i + window_size]
        frames.append(frame)

    return frames