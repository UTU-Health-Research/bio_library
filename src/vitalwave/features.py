import numpy as np

def get_ppg_peak_amplitude_ratio(arr : np.ndarray, feets : np.ndarray, systolic_peaks : np.ndarray, diastolic_peaks : np.ndarray):

    """
    Calculates amplitude ratio between systolic and diastolic peaks in an array.

    amplitude_ratio = get_amplitude_ratio(arr = nd_ppg_peaks["filt_ppg_signal"], feets = ppg_feets,
                                      systolic_peaks=ppg_systolic, diastolic_peaks=ppg_diastolic)

    Parameters
    ----------
    arr : np.ndarray
        Input data.
    feets : np.ndarray
        Indices for feets or footprints of specific events.
    systolic_peaks : np.ndarray
        Indices for systolic peaks.
    diastolic_peaks : np.ndarray
        Indices for diastolic peaks.

    Returns
    -------
    ratio : list
        Amplitude ratios calculated as arr[systolic_peaks[index]] / arr[diastolic_peaks[index]].

    Examples
    --------
    To find amplitude ratios of systolic and diastolic peaks in given signal.

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.features import get_ppg_peak_amplitude_ratio
       from vitalwave.peak_detectors import msptd

       from src.vitalwave.experimental import derive_ppg_signal_peaks

       ppg_peaks, ppg_feets = msptd(ppg, fs=fs)

       nd_ppg_with_peaks, nd_ppg_with_peak_types = derive_ppg_signal_peaks(arr=ppg,
                                                                           ppg_peaks=ppg_peaks, ppg_feets=ppg_feets,
                                                                           window_length=9, polyorder=5)

       ppg_feets = np.asarray(np.where(nd_ppg_with_peak_types == 1.0))[0]
       ppg_systolic = np.asarray(np.where(nd_ppg_with_peak_types == 2.0))[0]
       ppg_diastolic = np.asarray(np.where(nd_ppg_with_peak_types == 4.0))[0]

       l_ratio = get_ppg_peak_amplitude_ratio(arr=ppg, feets=ppg_feets, systolic_peaks=ppg_systolic,
                                              diastolic_peaks=ppg_diastolic)

       fig, ax = plt.subplots()
       ax.boxplot(l_ratio)
       plt.show()
    """

    ratio = []

    for index in range(feets.shape[0]):
        amplitude_ratio = arr[systolic_peaks[index]] / arr[diastolic_peaks[index]]
        ratio.append(amplitude_ratio)

    return ratio

def get_ppg_peak_integral_ratio(arr : np.ndarray, feets : np.ndarray, dicrotic_valley : np.ndarray, problems : np.ndarray):
    
    """
    Calculates integral ratio between systolic and diastolic intervals in an array.

    integrals = get_integral_ratio(arr = nd_ppg_peaks["filt_ppg_signal"], feets = ppg_feets,
                                   dicrotic_valley = ppg_dicrotic, problems = [121])

    Parameters
    ----------
    arr : np.ndarray
        Input data.
    feets : np.ndarray
        Indices for feets or footprints of specific events.
    dicrotic_valley : np.ndarray
        Indices representing dicrotic valleys.
    problems : np.ndarray
       Indices representing problematic data points to be excluded from calculations.

    Returns
    -------
    volumes : list
        Integral ratios calculated as volume of systolic to diastolic peak.

    Examples
    --------
    To find volume-ratios of systolic and diastolic peaks in given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from src.vitalwave.features import get_ppg_peak_integral_ratio

       from vitalwave.peak_detectors import msptd
       from src.vitalwave.experimental import derive_ppg_signal_peaks

       ppg_peaks, ppg_feets = msptd(ppg, fs=fs)

       nd_ppg_with_peaks, nd_ppg_with_peak_types = derive_ppg_signal_peaks(arr=ppg,
                                                                           ppg_peaks=ppg_peaks, ppg_feets=ppg_feets,
                                                                           window_length=9, polyorder=5)

       ppg_feets = np.asarray(np.where(nd_ppg_with_peak_types == 1.0))[0]
       ppg_dicrotic = np.asarray(np.where(nd_ppg_with_peak_types == 3.0))[0]

       l_ratio = get_ppg_peak_integral_ratio(arr=ppg, feets=ppg_feets,
                                             dicrotic_valley=ppg_dicrotic, problems=[121])

       fig, ax = plt.subplots()
       ax.boxplot(l_ratio)
       plt.show()
    """

    feets = np.delete(feets, problems)

    volumes = []

    for index in range(feets.shape[0]):
        try:
            systolic_integral  = np.trapz(arr[feets[index]:dicrotic_valley[index]])
            diastolic_integral = np.trapz(arr[dicrotic_valley[index]:feets[index + 1]])

            volumes.append(systolic_integral/diastolic_integral)
        except:
            continue

    return volumes

def get_egc_interval_q_s(arr : np.ndarray, q_points : np.ndarray, s_points : np.ndarray, fs : int = 200, max_len : int = 50, threshold : float = 0.0001):
    
    """
    Calculates time intervals between Q and S points in ECG signal.

    Parameters
    ----------
    arr : np.ndarray
        Input ECG signal.
    q_points : np.ndarray
        Indices for Q points in ECG signal.
    s_points : np.ndarray
        Indices for S points in ECG signal.
    fs : int
        Sampling rate of ECG signal in Hz.
        Default is 200.
    max_len : int, optional
        Scroll-distance to discover valleys.
        Default is 50.
    threshold : float, optional
        Threshold for identifying valleys.
        Default is 0.0001.

    Returns
    -------
    ratio : list
        Time intervals in seconds between Q and S points in ECG signal.

    Examples
    --------
    To find interval between Q and S peak ratios in given signal.

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from src.vitalwave.features import get_egc_interval_q_s
       from src.vitalwave import peak_detectors
       from src.vitalwave.experimental import get_ecg_signal_peaks

       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)

       nd_ecg_with_peaks, nd_ecg_with_peak_types = get_ecg_signal_peaks(arr=ecg, r_peaks=ecg_r_peaks,
                                                                        fs=fs)

       egc_q_notch = np.asarray(np.where(nd_ecg_with_peak_types == 2.0))[0]
       egc_s_notch = np.asarray(np.where(nd_ecg_with_peak_types == 4.0))[0]

       interval = get_egc_interval_q_s(arr=ecg, q_points=egc_q_notch, s_points=egc_s_notch)

       fig, ax = plt.subplots()
       ax.boxplot(interval)
       plt.show()
    """

    ratio = []

    for q, s in zip (q_points, s_points):

        q_drift = _get_local_valley(arr=-arr[:q], direction="left", max_len=max_len, threshold=threshold)
        s_drift = _get_local_valley(arr=-arr[s:], direction="right", max_len=max_len, threshold=threshold)

        if q_drift is None:
            print("q valley not found")
            q_drift = 0

        if s_drift is None:
            print("t valley not found")
            s_drift = 0

        in_seconds = ((s + s_drift) - (q - q_drift)) / fs
        ratio.append(in_seconds)

    return ratio

def get_egc_interval_p_t(arr : np.ndarray, p_points : np.ndarray, t_points : np.ndarray, fs : int = 200, max_len: int = 50, threshold : float = 0.0001):
    
    """
    Calculates time intervals between Q and T points in ECG signal.

    Parameters
    ----------
    arr : np.ndarray
        Input ECG signal.
    p_points : np.ndarray
        List of indices for P points in ECG signal.
    t_points : np.ndarray
        List of indices for T points in ECG signal.
    fs : int
        Sampling frequency of ECG signal in Hz.
        Default is 200.
    max_len : int, optional
        Scroll-distance to discover valleys.
        Default is 50.
    threshold : float, optional
        Threshold for identifying valleys.
        Default is 0.0001.

    Returns
    -------
    ratio : list
        Time intervals in seconds between Q and T points in ECG signal.

    Examples
    --------
    To find interval between P and T peaks ratios in given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from src.vitalwave.features import get_egc_interval_p_t
       from src.vitalwave import peak_detectors
       from src.vitalwave.experimental import get_ecg_signal_peaks

       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)

       nd_ecg_with_peaks, nd_ecg_with_peak_types = get_ecg_signal_peaks(arr=ecg, r_peaks=ecg_r_peaks,
                                                                        fs=fs)

       egc_p_peaks = np.asarray(np.where(nd_ecg_with_peak_types == 1.0))[0]
       egc_t_peaks = np.asarray(np.where(nd_ecg_with_peak_types == 5.0))[0]

       interval = get_egc_interval_p_t(arr=ecg, p_points=egc_p_peaks, t_points=egc_t_peaks)

       fig, ax = plt.subplots()
       ax.boxplot(interval)
       plt.show()
    """

    ratio = []

    for p, t in zip(p_points, t_points):
        p_drift = _get_local_valley(arr=arr[:p], direction="left", max_len=max_len, threshold=threshold)
        t_drift = _get_local_valley(arr=arr[t:], direction="right", max_len=max_len, threshold=threshold)

        if p_drift is None:
            print("q valley not found")
            p_drift = 0

        if t_drift is None:
            print("t valley not found")
            t_drift = 0

        in_seconds = ((t + t_drift) - (p - p_drift)) / fs
        ratio.append(in_seconds)

    return ratio

def get_egc_interval_q_t(arr : np.ndarray, q_points : np.ndarray, t_points : np.ndarray, fs : int = 200, max_len : int = 50, threshold : float = 0.0001):
    
    """
    Calculates time intervals between Q and T points in ECG signal.

    Parameters
    ----------
    arr : np.ndarray
        Input ECG signal
    q_points : np.ndarray
       Indices representing Q points in ECG signal.
    t_points : np.ndarray
        Indices representing T points in ECG signal.
    fs : int
        Sampling frequency of ECG signal in Hz.
        Default is 200.
    max_len : int, optional
        Scroll-distance to discover valleys.
        Default is 50.
    threshold : float, optional
        Threshold for identifying valleys.
        Default is 0.0001.

    Returns
    -------
    ratio : list
        Ttime intervals in seconds between Q and T points in ECG signal.

    Examples
    --------
    To find interval between Q and T peaks ratios in given signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from src.vitalwave.features import get_egc_interval_q_t
       from vitalwave import peak_detectors
       from src.vitalwave.experimental import get_ecg_signal_peaks

       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)

       nd_ecg_with_peaks, nd_ecg_with_peak_types = get_ecg_signal_peaks(arr=ecg, r_peaks=ecg_r_peaks, fs=fs)

       egc_q_peaks = np.asarray(np.where(nd_ecg_with_peak_types == 2.0))[0]
       egc_t_peaks = np.asarray(np.where(nd_ecg_with_peak_types == 5.0))[0]

       interval = get_egc_interval_q_t(arr=ecg, q_points=egc_q_peaks, t_points=egc_t_peaks)

       fig, ax = plt.subplots()
       ax.boxplot(interval)
       plt.show()
    """

    ratio = []

    for q, t in zip (q_points, t_points):
        q_drift = _get_local_valley(arr = -arr[:q], direction="left", max_len=max_len, threshold = threshold)
        t_drift = _get_local_valley(arr = arr[t:], direction="right", max_len=max_len, threshold = threshold)

        if q_drift is None:
            print ("q valley not found")
            q_drift = 0

        if t_drift is None:
            print ("t valley not found")
            t_drift = 0

        in_seconds = ((t + t_drift) - (q - q_drift)) / fs
        ratio.append(in_seconds)

    return ratio

def compute_meandist(diff_rri : any):
    
    """
    Computes mean distance between successive elements in given time series.

    Parameters
    ----------
    diff_rri : any
        Input time series data representing successive differences of RR intervals.

    Returns
    -------
    mean_distance : float
        Calculated mean distance value.
    
    Examples
    --------
    To get mean distances in input data.
    
    .. code-block:: python
        from vitalwave import features
        features.compute_meandist(diff_rri=data)
    """

    dist = []
    for i in range(len(diff_rri) - 1):
        dist.append(np.linalg.norm([diff_rri[i + 1], diff_rri[i]]))
    mean_distance = np.mean(np.array(dist))

    return mean_distance

def get_global_egc_features(r_peaks : np.ndarray, fs : int = 100, min_rr_interval : int = 50):

    """
    Calculates global ECG features from R peak indices.

    Parameters
    ----------
    r_peaks : np.ndarray
        R peak indices in ECG signal.
    fs : int
        Sampling rate of ECG signal in Hz.
        Deafult is 100.
    min_rr_interval : int
        Minimum RR interval in ms used for computing pNNxx.
        Default is 50.

    Returns
    -------
    global_features : object
        Scalar variables.
    
    Examples
    --------
    To get ECG features.
    
    .. code-block:: python
        from vitalwave import features
        features.get_global_egc_features(r_peaks=r, fs=100, min_rr_interval=50)
    """
    global_features = _Global_EGC_Features(r_peaks = r_peaks, fs = fs, min_rr_interval = min_rr_interval)
    return global_features

class _Global_EGC_Features:

    def __init__(self, r_peaks, fs = 100, min_rr_interval = 50):
        """
        rmssd            = Root Mean Square of Successive Differences
        mean_rr_interval = Mean RR Interval
        sdnn             = Standard Deviation of NN Intervals
        mean_heart_rate  = Mean Heart Rate
        std_heart_rate   = Standard Deviation of Heart Rate
        nn_xx            = Number of RR Intervals > xx ms
        pnn_xx           = pNN50: Percentage of NN50 of all RR intervals
        corr_heart_rate  = Standard Deviation of Heart Rate
        min_heart_rate   = Minimum Heart Rate
        max_heart_rate   = Maximum Heart Rate
        """

        self.ecg_r_peak_indices = r_peaks
        self.heart_rate = (fs * 60) / self.ecg_r_peak_indices

        self.root_mean_square_of_successive_differences = np.sqrt(np.mean(np.square(self.ecg_r_peak_indices)))
        self.mean_rr_interval = np.mean(self.ecg_r_peak_indices)
        self.sdnn = np.std(self.ecg_r_peak_indices)
        self.mean_heart_rate = 60 * fs / np.mean(self.ecg_r_peak_indices)
        self.std_heart_rate = np.std(self.heart_rate)

        self.nn_xx = np.sum(np.abs(self.ecg_r_peak_indices) > min_rr_interval) * 1
        self.pnn_xx = fs * self.nn_xx / len(self.ecg_r_peak_indices)

        self.min_heart_rate = np.min(self.heart_rate)
        self.max_heart_rate = np.max(self.heart_rate)

    def __str__(self):
        return (
            f"HRV and Heart Rate Statistics:\n"
            f"RMSSD: {self.root_mean_square_of_successive_differences:.2f}\n"
            f"Mean RR Interval: {self.mean_rr_interval:.2f}\n"
            f"SDNN: {self.sdnn:.2f}\n"
            f"Mean Heart Rate: {self.mean_heart_rate:.2f}\n"
            f"Standard Deviation of Heart Rate: {self.std_heart_rate:.2f}\n"
            f"NNxx: {self.nn_xx}\n"
            f"pNNxx: {self.pnn_xx:.2f}\n"
            f"Minimum Heart Rate: {self.min_heart_rate:.2f}\n"
            f"Maximum Heart Rate: {self.max_heart_rate:.2f}"
        )

def _get_local_valley(arr : np.ndarray, threshold : float = 0.0001, direction : str = 'right', max_len : int = 100):
    
    """
    Finds index of first local valley that defines point where gradient changes from decreasing to increasing.

    Parameters:
    -----------
    arr : np.ndarray
        Input values.
    threshold : float
        Threshold for gradient magnitude.
        Default is 0.0001.
    direction : str
        Search direction.
        Left to search backwards and right to onwards.
        Alternatives are 'left' and 'right'.
    max_len : int
        Maximum length of array to consider.
        Default is 100.

    Returns:
    --------
    gradients : int
        Index of first local valley or None if no suitable valley is found.
    
    Examples
    --------
    To find first local valley.
    
    .. code-block:: python
        from vitalwave import features
        features._get_local_valley(arr=data, threshold=0.0001, direction='right', max_len=100)
    """

    if direction == "left":
        arr = np.flip(arr)

    gradients = np.diff(arr[:max_len])

    for count, i in enumerate(range(len(gradients) - 1)):
        if (gradients[i] - threshold) < 0 and (gradients[i + 1] + threshold) > 0:
            if gradients[i] != gradients[i + 1]:
                return count + 1

    return None