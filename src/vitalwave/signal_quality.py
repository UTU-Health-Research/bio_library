import numpy as np

from scipy.stats import pearsonr

from vitalwave.basic_algos import extract_waveforms

def quality_index_for_waveform(arr : np.ndarray, r_peaks : np.ndarray):

    """
    Computes correlation coefficient between each individual waveform with average template.

    Parameters
    ----------
    arr : np.ndarray
        ECG signal.
    r_peaks : np.ndarray
        Indices of detected R peaks in ECG signal.

    Returns
    -------
    l_correlations : list
        Correlation coefficients and p-values for each individual waveform compared to average template.

    Examples
    --------
    To find R peaks of ECG in given signal.

    .. plot::
       :include-source:

       from vitalwave.example_data import load_biosignal
       import matplotlib.pyplot as plt

       import numpy as np

       limits = [0,2000]
       time, ecg = load_biosignal(type="ECG")

       fs = (1 / np.mean(np.diff(time)))

       from vitalwave.signal_quality import quality_index_for_waveform
       from vitalwave import peak_detectors

       ecg_r_peaks = peak_detectors.ecg_modified_pan_tompkins(ecg, fs)

       l_correlation = quality_index_for_waveform(arr=ecg, r_peaks=ecg_r_peaks, type="waveform")

       fig, ax = plt.subplots()
       ax.boxplot(l_correlation)
       plt.show()
    """

    l_correlations = []

    waveforms, mean_waveform_ecg = extract_waveforms(arr, r_peaks, mode="nn_interval")
    mean_waveform_ecg = mean_waveform_ecg[~np.isnan(mean_waveform_ecg)]

    for waveform in waveforms:
        waveform[np.isnan(waveform)] = 0
        correlation = np.ma.corrcoef(x=waveform, y=mean_waveform_ecg)[0][1]
        l_correlations.append(correlation)

    return l_correlations