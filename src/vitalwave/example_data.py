import os

import numpy as np

def load_biosignal(l_columns : list = None, type : str = 'ECG', clean : bool = True):

    """
    Loads ECG or PPG data from files.

    Parameters
    ----------
    l_columns : list, optional
        List of column names to include in returned data.
        If not provided, default column names will be used for respective ECG or PPG data.
        Default is None.
    type : str, optional
        Alternatives are 'ECG' and 'PPG'.
        Default is 'ECG'.
    clean : bool, optional
        If True, load clean data.
        If False, load data with motion artifacts.
        Default is True.

    Returns
    -------
    tuple
        Time.
        Data-columns.

    Examples
    --------
    To x with already defined column names.
    
    .. code-block:: python
        from vitalwave import example_data
        example_data.load_biosignal(l_columns=columns, type='ECG', clean=True)

    \nTo x with column names from data.
    
    .. code-block:: python
        from vitalwave import example_data
        example_data.load_biosignal(l_columns=None, type='ECG', clean=True)
    """

    if type in "ECG":
        filename = 'clean_ecg.npy' if clean else 'motion_ecg.npy'
        if l_columns is None:
            l_columns = ["time", "minMax_ecg", "ecg" ]

    elif type in "PPG":
        filename = 'clean_ppg.npy' if clean else 'motion_ppg.npy'
        if l_columns is None:
            l_columns = ["time", "minMax_ppg_1_ir", "minMax_ppg_1_green"]

    data_path = os.path.join(os.path.dirname(__file__), 'example_data', filename)
    nd_array = np.load(data_path)

    decouple= np.stack([nd_array[field_name].astype(float) for field_name in l_columns], axis=-1)

    time = decouple[:, 0]
    data = decouple[:, 1]

    return time, data