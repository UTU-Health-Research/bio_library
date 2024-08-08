import numpy as np

from vitalwave import basic_algos
from vitalwave.peak_detectors import ampd

from scipy.signal import find_peaks, savgol_filter
from scipy.signal import argrelextrema, windows


def get_ecg_signal_peaks(arr : np.ndarray, r_peaks : np.ndarray, fs : int):

    """
    Extracts and categorizes P, Q, R, S, and T peaks from ECG signal.

    Parameters
    ----------
    arr : np.ndarray
        ECG signal.
    r_peaks : np.ndarray
       R-peak locations in samples.
    fs : int
        Sampling rate.

    Returns
    -------
    tuple
        Binary array indicating presence of identified peaks in ECG signal.
        Array categorizing identified P, Q, R, S, and T peaks in ECG signal.

    Examples
    --------
    To find R peaks in given ECG signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave import peak_detectors
       from vitalwave.experimental import get_ecg_signal_peaks

       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)
       heights_peaks = ecg[ecg_r_peaks]

       plt.figure(figsize=(10, 5))
       plt.plot(ecg)
       plt.scatter(ecg_r_peaks, heights_peaks, marker='o', color='green')

       plt.show()

       nd_ecg_with_peaks, nd_ecg_with_peak_types = get_ecg_signal_peaks(arr=ecg, r_peaks=ecg_r_peaks,
                                                                        fs=fs)

       plt.figure(figsize=(10, 5))
       plt.plot(ecg)

       ecg_peaks = np.where(nd_ecg_with_peaks == 1.0)[0]
       heights_peaks = ecg[ecg_peaks]

       plt.scatter(ecg_peaks, heights_peaks, marker='o', color='green')
       plt.show()
    """

    # Extract waveforms and initial peaks.
    ecg_hilbert = basic_algos.homomorphic_hilbert_envelope(arr, fs)

    waveforms_ecg, mean_waveform_ecg = basic_algos.extract_waveforms(ecg_hilbert, (np.rint(r_peaks)).astype(int),
                                                                     mode="nn_interval")

    # Identify peaks in waveforms.
    cumulative_ecg_peaks, initial_peaks_ecg_types, waveforms_with_error  = _identify_peaks_in_waveform(waveforms_ecg,
                                                                                                       get_bad_indexes=True)

    # Process identified peaks and types.
    ecg_signal = np.ravel(waveforms_ecg)
    ecg_signal = ecg_signal[~np.isnan(ecg_signal)]

    ecg_peaks_all = np.concatenate(cumulative_ecg_peaks)
    ecg_peaks_all_types = np.concatenate(initial_peaks_ecg_types)

    nd_ecg_with_peaks = np.zeros(ecg_signal.shape)
    nd_ecg_with_peak_types = np.zeros(ecg_signal.shape)

    nd_ecg_with_peaks[ecg_peaks_all] = 1

    for index, peak in enumerate(ecg_peaks_all):
        nd_ecg_with_peak_types[peak] = ecg_peaks_all_types[index]

    # Count R-peaks and adjust peak arrays.
    count = np.where(nd_ecg_with_peak_types == 3)[0].shape[0]

    r_peaks = np.delete(r_peaks, waveforms_with_error)
    r_peaks = r_peaks[:count]

    t_peak = r_peaks + (np.where(nd_ecg_with_peak_types == 5)[0] - np.where(nd_ecg_with_peak_types == 3)[0])
    p_peak = r_peaks + (np.where(nd_ecg_with_peak_types == 1)[0] - np.where(nd_ecg_with_peak_types == 3)[0])

    # Find P, Q, S, and T peaks.
    nd_ecg_with_peaks, nd_ecg_with_peak_types = _find_p_q_s_t(arr = arr, r_peaks=r_peaks,
                                                              t_peaks=t_peak, p_peaks=p_peak, window=3)

    return nd_ecg_with_peaks, nd_ecg_with_peak_types

def _find_p_q_s_t(arr : np.ndarray, r_peaks : np.ndarray, t_peaks : np.ndarray, p_peaks : np.ndarray, window : int = 3):
    
    """
    Identifies P, Q, S, and T peaks in ECG signal and categorizes their types.

    Parameters
    ----------
    arr : np.ndarray
        ECG signal.
    r_peaks : np.ndarray
        R peak locations in ECG signal.
    t_peaks : np.ndarray
        T peak locations in ECG signal.
    p_peaks : np.ndarray
       P peak locations in ECG signal.
    window : int
        Half of window size used for peak identification around P and T peaks.
        Default is 3.

    Returns
    -------
    tuple
        Binary array indicating presence of identified peaks in ECG signal.
        Array categorizing identified P, Q, R, S, and T peaks in ECG signal.

    Examples
    --------
    To find peaks in ECG signal.
    
    .. code-block:: python
        from vitalwave import experimental
        experimental._find_p_q_s_t(arr=ecg, r_peaks=r, t_peaks=t, p_peaks=p, window=3)
    """

    nd_ecg_with_peaks = np.zeros(arr.shape[0])
    nd_ecg_with_peak_types = np.zeros(arr.shape[0])

    for index, (p, r, t) in enumerate (zip(p_peaks, r_peaks, t_peaks)):
        try:
            nd_ecg_with_peaks[(p-window) + (np.argmax(arr[(p-window):(p+window)]))] = 1
            nd_ecg_with_peak_types[(p-window) + (np.argmax(arr[(p-window):(p+window)]))] = 1

            nd_ecg_with_peaks[p + (np.argmin(arr[p:r]))] = 1
            nd_ecg_with_peak_types[p + (np.argmin(arr[p:r]))] = 2

            nd_ecg_with_peaks[r] = 1
            nd_ecg_with_peak_types[r] = 3

            nd_ecg_with_peaks[r + (np.argmin(arr[r:t]))] = 1
            nd_ecg_with_peak_types[r + (np.argmin(arr[r:t]))] = 4

            nd_ecg_with_peaks[(t-window) + np.argmax(arr[(t-window):(t+window)])] = 1
            nd_ecg_with_peak_types[(t-window) + np.argmax(arr[(t-window):(t+window)])] = 5
        except:
            continue

    return nd_ecg_with_peaks, nd_ecg_with_peak_types

def _identify_peaks_in_waveform(waveforms_ecg : list, get_bad_indexes : bool = False):

    """
    Identifies peaks in ECG waveforms and categorizes their types.

    Parameters
    ----------
    waveforms_ecg : list
        ECG waveforms for peak identification.
    get_bad_indexes : bool
        Flag to indicate whether to return indexes of waveforms with errors.
        Default is False.

    Returns
    -------
    tuple
        List of cumulative peak locations for all waveforms categorized by types.
        List of peak type categorizations corresponding to initial_peaks_ecg.
        List of indexes of waveforms with missing points or errors if get_bad_indexes is True.

    Examples
    --------
    To identify peaks in ECG signal.
    
    .. code-block:: python
        from vitalwave import experimental
        experimental._identify_peaks_in_waveform(waveforms_ecg=waves, get_bad_indexes=False)
    """

    initial_peaks_ecg_types = []
    cumulative_ecg_peaks = []
    waveforms_with_error = []

    idx_peak_ids = 0

    for index, waveform in enumerate(waveforms_ecg):

        try:
            waveform = waveform[~np.isnan(waveform)]
            r_peak = np.argmax(waveform)

            peaks_ecg_high = argrelextrema(waveform, np.greater)[0]
            peaks_ecg_low = argrelextrema(waveform, np.less)[0]

            initial_peaks_ecg = np.sort(np.concatenate((peaks_ecg_high, peaks_ecg_low)))

            idx = (np.abs(np.asarray(initial_peaks_ecg) - r_peak)).argmin()

            if idx >= 2 and (initial_peaks_ecg.shape[0] - idx) >= 3:
                nd_peak_type = np.zeros(initial_peaks_ecg.shape)

                nd_peak_type[idx - 2] = 1
                nd_peak_type[idx - 1] = 2
                nd_peak_type[idx] = 3
                nd_peak_type[idx + 1] = 4
                nd_peak_type[idx + 2] = 5

                peaks = initial_peaks_ecg
                peaks += idx_peak_ids
                idx_peak_ids += waveform.shape[0]

                initial_peaks_ecg_types.append(nd_peak_type)
                cumulative_ecg_peaks.append(peaks)

            else:
                #print("missing points: ", index)
                waveforms_with_error.append(index)

                idx_peak_ids += waveform.shape[0]
        except:
            continue

    if get_bad_indexes:
        return cumulative_ecg_peaks, initial_peaks_ecg_types, waveforms_with_error
    else:
        return cumulative_ecg_peaks, initial_peaks_ecg_types

def derive_ppg_signal_peaks(arr : np.ndarray, ppg_peaks : np.ndarray, ppg_feets : np.ndarray, window_length : int = 9, polyorder : int = 5):

    """
    Derives and categorizes PPG signal peaks based on waveforms and peak types.

    Parameters
    ----------
    arr : np.ndarray
        PPG signal.
    ppg_peaks : np.ndarray
        Indices of detected peaks in PPG signal.
    ppg_feets : np.ndarray
        Indices of calculated feet.
    window_length : int
        Window length for Savitzky-Golay filter.
        Default is 9.
    polyorder : int
        Polynomial order for Savitzky-Golay filter.
        Default is 5.

    Returns
    -------
    tuple
        Array indicating presence of identified dicrotic and diastolic peaks in PPG signal.
        Array categorizing feet, systolic peak, dicrotic notch, and diastolic peaks in PPG signal.

    Examples
    --------
    To find R peaks of given ECG signal.

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from src.vitalwave.experimental import derive_ppg_signal_peaks
       from vitalwave.peak_detectors import msptd

       ppg_signal = ppg

       ppg_peaks, ppg_feets = msptd(ppg, fs=fs)

       heights_peaks = ppg[ppg_peaks]
       heights_feets = ppg[ppg_feets]

       plt.figure(figsize=(10, 5))
       plt.plot(ppg)

       plt.scatter(ppg_peaks, heights_peaks, marker='o', color='green')
       plt.scatter(ppg_feets, heights_feets, marker='o', color='red')

       plt.show()

       nd_ppg_with_peaks, nd_ppg_with_peak_types = derive_ppg_signal_peaks(arr=ppg,
                                                                           ppg_peaks=ppg_peaks,
                                                                           ppg_feets=ppg_feets,
                                                                           window_length=9,
                                                                           polyorder=5)

       plt.figure(figsize=(10, 5))
       plt.plot(ppg)

       ppg_peaks = np.where(nd_ppg_with_peaks == 1.0)[0]
       heights_peaks = ppg[ppg_peaks]

       plt.scatter(ppg_peaks, heights_peaks, marker='o', color='green')
       plt.scatter(ppg_feets, heights_feets, marker='o', color='red')

       plt.show()
    """

    # Extract waveforms and initial peaks.
    waveforms_ppg, mean_waveform_ppg = basic_algos.extract_waveforms(arr, (np.rint(ppg_feets)).astype(int), mode="fid_to_fid")

    # Identify PPG peaks in waveforms.
    cumulative_ppg_peaks, initial_peaks_ppg_types, problems = _identify_ppg_peaks_in_waveform(waveforms_ppg,
                                                                                              window_length=window_length,
                                                                                              polyorder=polyorder)

    problems = np.unique(problems)

    # Process identified peaks and types.
    ppg_signal = np.ravel(waveforms_ppg)
    ppg_signal = ppg_signal[~np.isnan(ppg_signal)]

    ppg_peaks_all = np.concatenate(cumulative_ppg_peaks)
    ppg_peaks_all_types = np.concatenate(initial_peaks_ppg_types)

    nd_ppg_with_peak_types = np.zeros(ppg_signal.shape)

    for index, peak in enumerate(ppg_peaks_all):
        nd_ppg_with_peak_types[peak] = ppg_peaks_all_types[index]

    # Find dicrotic and diastolic points and adjust the ppg-signal to fit using window based approach.
    d2x = savgol_filter(arr, window_length=9, polyorder=5, deriv=2)
    systolic_peaks = np.delete(ppg_peaks, problems)

    systoles   = np.where(nd_ppg_with_peak_types == 1)[0]
    dictrotics = np.where(nd_ppg_with_peak_types == 2)[0]
    diastolics = np.where(nd_ppg_with_peak_types == 3)[0]

    size = min ((systolic_peaks.shape[0], systoles.shape[0]))

    dicrotic = systolic_peaks[:size] + (dictrotics - systoles)
    diastolic = systolic_peaks[:size]  + (diastolics - systoles)

    nd_ppg_with_peaks, nd_ppg_with_peak_types = _find_dicrotic_and_diastolic(d2x_arr=d2x,
                                                                             feets=ppg_feets,
                                                                             systolic=systolic_peaks,
                                                                             dicrotic=dicrotic,
                                                                             diastolic=diastolic,
                                                                             window=4)

    return nd_ppg_with_peaks, nd_ppg_with_peak_types

def _find_dicrotic_and_diastolic(d2x_arr : np.ndarray, feets : np.ndarray, systolic : np.ndarray,
                                 dicrotic : np.ndarray, diastolic : np.ndarray, window : int = 4):
    
    """
    Identifies and categorizes dicrotic and diastolic points based on second derivative of PPG signal.

    Parameters
    ----------
    d2x_arr : np.ndarray
        Second derivative of signal.
    feets : np.ndarray
        Indices for foot points of signal.
    systolic : np.ndarray
        Indices for systolic points of signal.
    dicrotic : np.ndarray
        Indices for dicrotic points of signal.
    diastolic : np.ndarray
        Indices for diastolic points of signal.
    window : int
        Half-width of window for peak identification.
        Default is 4.

    Returns
    -------
    tuple
        Array with peaks marked as 1.
        Array with peak types marked as 1, 2, 3 or 4.

    Examples
    --------
    To identify and categorize dicrotic and diastolic points in PPG.
    
    .. code-block:: python
        from vitalwave import experimental
        experimental._find_dicrotic_and_diastolic(d2x_arr=der, feets=f, systolic=sys, dicrotic=dic, diastolic=dia, window=4)
    """

    nd_ecg_with_peaks = np.zeros(d2x_arr.shape[0])
    nd_ecg_with_peak_types = np.zeros(d2x_arr.shape[0])

    for index, (f, syst, dic, dia) in enumerate (zip(feets, systolic, dicrotic, diastolic)):

        nd_ecg_with_peaks[(f - window) + (np.argmax(d2x_arr[(f - window):(f + window)]))] = 1
        nd_ecg_with_peak_types[(f - window) + (np.argmax(d2x_arr[(f - window):(f + window)]))] = 1

        nd_ecg_with_peaks[(syst-window) + (np.argmin(d2x_arr[(syst - window):(syst + window)]))] = 1
        nd_ecg_with_peak_types[(syst-window) + (np.argmin(d2x_arr[(syst - window):(syst + window)]))] = 2

        nd_ecg_with_peaks[(dic-window) + (np.argmax(d2x_arr[(dic - window):(dic + window)]))] = 1
        nd_ecg_with_peak_types[(dic-window) + (np.argmax(d2x_arr[(dic - window):(dic + window)]))] = 3

        nd_ecg_with_peaks[(dia-window) + (np.argmin(d2x_arr[(dia - window):(dia + window)]))] = 1
        nd_ecg_with_peak_types[(dia-window) + (np.argmin(d2x_arr[(dia - window):(dia + window)]))] = 4

    return nd_ecg_with_peaks, nd_ecg_with_peak_types


def _identify_ppg_peaks_in_waveform(waveforms_ppg : list, window_length : int = 9, polyorder : int = 5):

    """
    Identifies and categorizes PPG peaks in waveforms.

    Parameters
    ----------
    waveforms_ppg : list
        PPG waveforms for peak identification.
    window_length : int
        Window length for Savitzky-Golay filter.
        Default is 9.
    polyorder : int
        Polynomial order for Savitzky-Golay filter.
        Default is 5 due to twin-peak structure of waveform.

    Returns
    -------
    tuple
        List of cumulative peak locations for all waveforms categorized by types.
        List of peak type categorizations corresponding to initial_peaks_ppg.
        List of indexes of waveforms where peak identification encountered issues.
    
    Examples
    --------
    To identify and categorize PPG peaks.
    
    .. code-block:: python
        from vitalwave import experimental
        experimental._identify_ppg_peaks_in_waveform(waveforms_ppg=waves, window_length=9, polyorder=5)
    """

    idx_peak_ids = 0
    real_index = 0

    initial_peaks_ppg_3 = []
    initial_peaks_ppg_types = []
    cumulative_ppg_peaks = []

    problems = []

    for index, waveform in enumerate(waveforms_ppg):
        try:
            waveform = waveform[~np.isnan(waveform)]

            p_1 = np.argmax(waveform)
            p_2 = p_1 + np.argmin(waveform[p_1:])

            peaks_ecg_high = argrelextrema(waveform, np.greater)[0]
            peaks_ecg_low = argrelextrema(waveform, np.less)[0]

            initial_peaks_ppg = np.sort(np.concatenate((peaks_ecg_high, peaks_ecg_low)))

            idx = (np.abs(np.asarray(initial_peaks_ppg) - p_1)).argmin()

            if initial_peaks_ppg.shape[0] == 3:
                # Three peaks found.
                initial_peaks_ppg_3.append(initial_peaks_ppg)

            elif initial_peaks_ppg.shape[0] == 1:
                # Main peak found and two peaks derived.
                d2x = savgol_filter(waveform, window_length=window_length, polyorder=polyorder, deriv=2)
                diastolic, dicrotic = _get_diastolic_dicrotic_points(arr=d2x, start=p_1, stop=p_2)

                initial_peaks_ppg_3.append(np.array([p_1, dicrotic.item(), diastolic.item()]))

            else:
                idx_peak_ids += waveform.shape[0]

                problems.append(index)
                continue

            index = real_index

            nd_peak_type = np.zeros(initial_peaks_ppg_3[index].shape)

            nd_peak_type[idx] = 1
            nd_peak_type[idx + 1] = 2
            nd_peak_type[idx + 2] = 3

            peaks = initial_peaks_ppg_3[index]
            peaks += idx_peak_ids
            idx_peak_ids += waveform.shape[0]

            initial_peaks_ppg_types.append(nd_peak_type)
            cumulative_ppg_peaks.append(peaks)

            real_index += 1
        except:
            problems.append(index)
            continue

    return cumulative_ppg_peaks, initial_peaks_ppg_types, problems

def _get_diastolic_dicrotic_points(arr : np.ndarray, start : int, stop : int):

    """
    Calculates diastolic and dicrotic indices based on PPG signal, peak indices, and onset indices.

    If only systolic peak is found then key steps selected to find dicrotic and diastolic peaks are:
    Start from second-order-derivative of PPG signal; from waveform arg_max to waveform argmin.
    Isolate peaks between positive and negative highs; dicrotic signal is found from positive-highs.
    Find index of maximum peak in dicrotic region and find nearest diastolic peak.
    Return set of indexes for diastolic and dicrotic points.

    Article to reference in https://www.frontiersin.org/articles/10.3389/fbioe.2023.1199604/full.
    Reference to Savitzky-Golay used to calculate derivatives in https://eigenvector.com/wp-content/uploads/2020/01/SavitzkyGolay.pdf.

    Parameters
    ----------
    arr : np.ndarray
        Second order derivative of PPG signal.
    start : int
        Main peak of PPG signal.
    stop : int
        Local minimum at end of waveform.

    Returns
    -------
    tuple
        Indices representing diastolic points.
        Indices representing dicrotic points.

    Examples
    --------
    To calculate PPG parameter values.
    
    .. code-block:: python
        from vitalwave import experimental
        experimental._get_diastolic_dicrotic_points(arr=ppg, start=start, stop=stop)
    """

    # Extract the relevant portion of the signal.
    signal_portion = arr[start:stop]

    # Isolate between positive and negative highs.
    aux_dic, _ = find_peaks(signal_portion)
    aux_dia, _ = find_peaks(-signal_portion)

    if len(aux_dic) != 0:

        # Find the index of the maximum peak in the dicrotic region.
        ind_max, = np.where(signal_portion[aux_dic] == np.max(signal_portion[aux_dic]))
        aux_dic_max = aux_dic[ind_max]

        if len(aux_dia) != 0:

            # Calculate distances between dicrotic and diastolic peaks.
            nearest = aux_dia - aux_dic_max
            aux_dic = aux_dic_max
            dicrotic = (aux_dic + start).astype(int)

            ind_dia, = np.where(nearest > 0)
            aux_dia = aux_dia[ind_dia]
            nearest = nearest[ind_dia]

            if len(nearest) != 0:

                # Find the nearest diastolic peak.
                ind_nearest, = np.where(nearest == np.min(nearest))
                aux_dia = aux_dia[ind_nearest]
                diastolic = (aux_dia + start).astype(int)
            else:
                dicrotic = (aux_dic_max + start).astype(int)

    return diastolic, dicrotic

def do_ppg_from_raw_signal(arr : np.ndarray, fs : float, signal_integration_window : float = 0,
                           segment : int = 2, threshold : float = 2, return_as_segment : bool = False):
    
    """
    Processes PPG data from raw signals while segmenting and detecting anomalies.

    Parameters
    ----------
    arr : np.ndarray
        Raw PPG signal.
    fs : float
        Sampling rate.
    signal_integration_window : float, optional
        Integration window size.
        Default is 0.
        If greater than 0, function will perform area under curve integration.
    segment : int, optional
        Number of segments to divide signal into.
        Default is 2.
        If set to 0, function will process entire array as a single segment.
    threshold : float, optional
        Threshold for anomaly detection.
        Default is 2.
        Anomalies are detected based on differences between consecutive peaks.
        Peaks with differences greater than this threshold are considered anomalies.
    return_as_segment, optional
        If False, returns a single array with segments or complete signal without anomalies.
        If True, returns individual segments along with any detected anomalies.
        Default is False.

    Returns
    -------
    tuple
        Segmented or complete PPG signal.
        Detected peaks.

    Examples
    --------
    To preprocess, validate, and find peaks or valleys in a PPG signal:

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ppg = load_biosignal(type="PPG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.experimental import do_ppg_from_raw_signal

       ppg_data_all, peak_indices_all = do_ppg_from_raw_signal(arr=ppg, fs=fs, segment=6, threshold=3,
                                                               return_as_segment=False)
       heights_peaks_all = ppg_data_all[peak_indices_all]

       ppg_data, peak_indices = do_ppg_from_raw_signal(arr=ppg, fs=fs, segment=3, threshold=3,
                                                       return_as_segment=True)
       heights_peaks = ppg_data[0][peak_indices[0]]

       fig, axes = plt.subplots(2, 1, figsize=(10, 8))

       axes[0].plot(ppg_data_all)
       axes[0].scatter(peak_indices_all, heights_peaks_all, marker='o', color='green')
       axes[0].set_title("PPG Data as complete (Segment = 3)")

       axes[1].plot(ppg_data[0])
       axes[1].scatter(peak_indices[0], heights_peaks, marker='o', color='green')
       axes[1].set_title("PPG Data as segments (Segment = 6)")

       plt.tight_layout()

       plt.show()

    If segmentation is enabled meaning segment > 0, function divides signal into multiple segments and detects peaks within each segment.
    Anomalies are detected based on differences between consecutive peak indices.
    Segments without anomalies are returned.
    If segmentation is disabled  meaning segment = 0, entire input signal is processed as single segment.

    ppg_data, peak_indices = do_ppg_from_raw_signal(arr = nd_ppg["ppg_1_ir"], fs = fs, segment = 6,
                                                    threshold = 3, return_as_segment=True)
    """

    # To allow for segmented and non segmented signal.
    idx_segment_ids = 0

    segments = []
    peaks_all = []

    # Pre-filter.
    ppg_filtered = [basic_algos.min_max_normalize(basic_algos.butter_filter(arr=arr, n=4, wn=[0.5, 8],
                                                                            filter_type='bandpass', fs=int(fs)))]

    # Make an area under the curve.
    if signal_integration_window > 0:
        ppg_filtered = basic_algos.moving_average_filter(arr=ppg_filtered[0],
                                                         window=int(signal_integration_window),
                                                         type = "triang")

    # Segment the signal using np.split.
    # Causes an error unless the split is not even.
    if segment > 0:
        segment_length = len(ppg_filtered[0]) // segment * segment
        ppg_filtered = np.split(ppg_filtered[0][:segment_length], segment)

    # Single segment and multiple segments are processed within the same process.
    for ppg_segment in ppg_filtered:
        feet_2_ppg = ampd(-ppg_segment, int(fs))
        peaks_2_ppg = ampd(ppg_segment, int(fs))

        initial_peaks_ppg = np.sort(np.concatenate((feet_2_ppg, peaks_2_ppg)))
        initial_peaks_ppg += idx_segment_ids

        idx_segment_ids += ppg_segment.shape[0]

        min_length = min(feet_2_ppg.shape[0], peaks_2_ppg.shape[0])
        difference = feet_2_ppg[:min_length] - peaks_2_ppg[:min_length]

        outliers = _has_gap_between_segments(difference, threshold=threshold)

        if outliers.__len__() == 0:
            segments.append(ppg_segment)
            peaks_all.append(initial_peaks_ppg)

        elif segment == 0:
            return ppg_segment, outliers

    # Check the completed signal.
    # The peak detection is re-iterated with the passing signal.
    if not return_as_segment:
        ppg_segments_all = np.concatenate(segments)

        feet_2_ppg_all = ampd(-ppg_segments_all, int(fs))
        peaks_2_ppg_all = ampd(ppg_segments_all, int(fs))

        min_length = min(feet_2_ppg_all.shape[0], peaks_2_ppg_all.shape[0])
        differences = feet_2_ppg_all[:min_length] - peaks_2_ppg_all[:min_length]

        outliers = _has_gap_between_segments(differences, threshold=threshold)

        if outliers.__len__() == 0:
            peaks_ppg = np.sort(np.concatenate((feet_2_ppg_all, peaks_2_ppg_all)))
            return ppg_segments_all, peaks_ppg

        else:
            return ppg_segments_all, outliers

    return segments, peaks_all

def _has_gap_between_segments(difference_values : np.ndarray, threshold : float = 2):
    
    """
    Detects anomalies based on actionable difference value based metric.

    Parameters
    ----------
    difference_values : np.ndarray
        Array of differences between PPG cycles feet and peaks.
    threshold : float
        Threshold for anomaly detection.
        Default is 2.

    Returns
    -------
    outliers : list
        Indexes of potential outliers.

    Examples
    --------
    To get anomalies of signal.
    
    .. code-block:: python
        from vitalwave import experimental
        experimental._has_gap_between_segments(difference_values=dif, threshold=2)
    """

    q1 = np.percentile(difference_values, 25)
    q3 = np.percentile(difference_values, 75)

    iqr = q3 - q1

    lower_bound = q1 - (iqr * threshold)
    upper_bound = q3 + (iqr * threshold)

    # Derive coefficient and do min threshold + coef * 1.
    outliers = [i for i, value in enumerate(difference_values) if value < lower_bound or value > upper_bound]

    return outliers