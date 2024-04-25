import numpy as np
import torchaudio
from numba import jit
from torchaudio import AudioMetaData


def read_audio_metadata(file: str) -> AudioMetaData:
    """
    Reads the metadata of an audio file.

    Args:
        file (str): The path to the audio file.

    Returns:
        AudioMetaData: The metadata of the audio file.
    """
    # Use torchaudio to read the audio file metadata
    return torchaudio.info(file)


"""
mostly taken from:
https://github.com/schmiph2/pysepm
"""


def extract_overlapped_windows(x, nperseg, noverlap, window=None):
    # source: https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/spectral.py
    """
    Extract overlapping windows from a signal.

    Parameters:
        x (ndarray): The input signal.
        nperseg (int): The number of samples per segment.
        noverlap (int): The number of samples to overlap.
        window (ndarray, optional): The window function to apply. Defaults to None.

    Returns:
        ndarray: Overlapped windows of the signal.

    Notes:
        - This function is inspired by scipy.signal.spectral.
    """
    # Calculate the step size
    step = nperseg - noverlap

    # Calculate the shape of the resulting windows
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // step, nperseg)

    # Calculate the strides for creating windows
    strides = x.strides[:-1] + (step * x.strides[-1], x.strides[-1])

    # Create overlapped windows
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

    # Apply window function if provided
    if window is not None:
        result = window * result

    return result


@jit
def find_loc_peaks(slope, energy):
    """
    Find the locations of peaks based on the provided slope and energy arrays.

    Parameters:
    slope (ndarray): The slope array.
    energy (ndarray): The energy array.

    Returns:
    ndarray: Array containing the locations of peaks.
    """
    num_crit = len(energy)

    # Initialize the array to store the locations of peaks
    loc_peaks = np.zeros_like(slope)

    for ii in range(len(slope)):
        n = ii
        if slope[ii] > 0:
            while ((n < num_crit - 1) and (slope[n] > 0)):
                n = n + 1
            loc_peaks[ii] = energy[n - 1]
        else:
            while ((n >= 0) and (slope[n] <= 0)):
                n = n - 1
            loc_peaks[ii] = energy[n + 1]

    return loc_peaks


@jit
def lpcoeff(speech_frame, model_order):
    """
    Compute the LP coefficients using Autocorrelation and Levinson-Durbin algorithm.

    Parameters:
        speech_frame (ndarray): Input speech frame.
        model_order (int): Order of the model.

    Returns:
        tuple: LP parameters and Autocorrelation lags.
    """
    eps = np.finfo(np.float64).eps

    # Compute Autocorrelation Lags
    winlength = max(speech_frame.shape)
    R = np.zeros((model_order + 1,))
    for k in range(model_order + 1):
        if k == 0:
            R[k] = np.sum(speech_frame[0:] * speech_frame[0:])
        else:
            R[k] = np.sum(speech_frame[0:-k] * speech_frame[k:])

    # Levinson-Durbin
    a = np.ones((model_order,))
    a_past = np.ones((model_order,))
    rcoeff = np.zeros((model_order,))
    E = np.zeros((model_order + 1,))

    E[0] = R[0]

    for i in range(0, model_order):
        a_past[0:i] = a[0:i]

        sum_term = np.sum(a_past[0:i] * R[i:0:-1])

        if E[i] == 0.0:
            rcoeff[i] = np.inf
        else:
            rcoeff[i] = (R[i + 1] - sum_term) / (E[i])

        a[i] = rcoeff[i]

        if i > 0:
            a[0:i] = a_past[0:i] - rcoeff[i] * a_past[i - 1::-1]

        E[i + 1] = (1 - rcoeff[i] * rcoeff[i]) * E[i]

    acorr = R
    refcoeff = rcoeff
    lpparams = np.ones((model_order + 1,))
    lpparams[1:] = -a
    return (lpparams, R)
