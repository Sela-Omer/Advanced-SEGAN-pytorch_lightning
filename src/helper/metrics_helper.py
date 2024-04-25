import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import stft
import pesq

from src.helper.audio_helper import extract_overlapped_windows, find_loc_peaks, lpcoeff

"""
Helper functions for calculating metrics.

mostly taken from:
https://github.com/schmiph2/pysepm
"""


def SNRseg(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    """
    Calculate the segmental SNR (Signal-to-Noise Ratio) for the processed speech.

    Parameters:
        clean_speech (ndarray): The clean speech signal.
        processed_speech (ndarray): The processed speech signal.
        fs (int): The sampling frequency.
        frameLen (float, optional): The length of each frame in seconds. Defaults to 0.03.
        overlap (float, optional): The overlap between frames as a fraction of the frame length. Defaults to 0.75.

    Returns:
        float: The mean segmental SNR.

    Notes:
        - The SNR is used to measure the quality of the processed speech compared to the clean speech.
        - The SNR is calculated using the energy measures of the clean and processed speech.
        - The frames are extracted from the clean and processed speech using the specified frame length and overlap.
        - The signal energy and noise energy are calculated for each frame.
        - The segmental SNR is calculated using the formula: 10 * log10(signal_energy / (noise_energy + eps) + eps)
        - The segmental SNR is clipped to the range [MIN_SNR, MAX_SNR] to prevent outliers.
        - The last frame is removed as it is not valid.
        - The mean segmental SNR is returned.
    """
    eps = np.finfo(np.float64).eps  # small value to prevent division by zero

    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    MIN_SNR = -10  # minimum SNR in dB
    MAX_SNR = 35  # maximum SNR in dB

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))  # Hann window

    # Extract overlapped frames from clean speech
    clean_speech_framed = extract_overlapped_windows(clean_speech, winlength, winlength - skiprate, hannWin)

    # Extract overlapped frames from processed speech
    processed_speech_framed = extract_overlapped_windows(processed_speech, winlength, winlength - skiprate, hannWin)

    # Calculate signal energy for each frame
    signal_energy = np.power(clean_speech_framed, 2).sum(-1)

    # Calculate noise energy for each frame
    noise_energy = np.power(clean_speech_framed - processed_speech_framed, 2).sum(-1)

    # Calculate segmental SNR
    segmental_snr = 10 * np.log10(signal_energy / (noise_energy + eps) + eps)

    # Clip segmental SNR to the range [MIN_SNR, MAX_SNR]
    segmental_snr = np.clip(segmental_snr, MIN_SNR, MAX_SNR)

    # Remove last frame as it is not valid
    segmental_snr = segmental_snr[:-1]

    # Return the mean segmental SNR
    return np.mean(segmental_snr)


def wss(clean_speech, processed_speech, fs, frameLen=0.03, overlap=0.75):
    """
    Calculate the Weighted Spectral Distortion (WSS) between the clean speech and processed speech signals.

    Parameters:
    - clean_speech (ndarray): The clean speech signal as a NumPy array.
    - processed_speech (ndarray): The processed speech signal as a NumPy array.
    - fs (int): The sample rate of the speech signals.
    - frameLen (float, optional): The length of each frame in seconds. Default is 0.03.
    - overlap (float, optional): The overlap between consecutive frames as a fraction of the frame length. Default is 0.75.

    Returns:
    - distortion (float): The mean weighted spectral distortion between the clean speech and processed speech signals.
    """
    Kmax = 20  # value suggested by Klatt, pg 1280
    Klocmax = 1  # value suggested by Klatt, pg 1280
    alpha = 0.95
    if clean_speech.shape != processed_speech.shape:
        raise ValueError('The two signals do not match!')
    eps = np.finfo(np.float64).eps
    clean_speech = clean_speech.astype(np.float64) + eps
    processed_speech = processed_speech.astype(np.float64) + eps
    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    max_freq = fs / 2  # maximum bandwidth
    num_crit = 25  # number of critical bands
    n_fft = 2 ** np.ceil(np.log2(2 * winlength))
    n_fftby2 = int(n_fft / 2)

    cent_freq = np.zeros((num_crit,))
    bandwidth = np.zeros((num_crit,))

    cent_freq[0] = 50.0000
    bandwidth[0] = 70.0000
    cent_freq[1] = 120.000
    bandwidth[1] = 70.0000
    cent_freq[2] = 190.000
    bandwidth[2] = 70.0000
    cent_freq[3] = 260.000
    bandwidth[3] = 70.0000
    cent_freq[4] = 330.000
    bandwidth[4] = 70.0000
    cent_freq[5] = 400.000
    bandwidth[5] = 70.0000
    cent_freq[6] = 470.000
    bandwidth[6] = 70.0000
    cent_freq[7] = 540.000
    bandwidth[7] = 77.3724
    cent_freq[8] = 617.372
    bandwidth[8] = 86.0056
    cent_freq[9] = 703.378
    bandwidth[9] = 95.3398
    cent_freq[10] = 798.717
    bandwidth[10] = 105.411
    cent_freq[11] = 904.128
    bandwidth[11] = 116.256
    cent_freq[12] = 1020.38
    bandwidth[12] = 127.914
    cent_freq[13] = 1148.30
    bandwidth[13] = 140.423
    cent_freq[14] = 1288.72
    bandwidth[14] = 153.823
    cent_freq[15] = 1442.54
    bandwidth[15] = 168.154
    cent_freq[16] = 1610.70
    bandwidth[16] = 183.457
    cent_freq[17] = 1794.16
    bandwidth[17] = 199.776
    cent_freq[18] = 1993.93
    bandwidth[18] = 217.153
    cent_freq[19] = 2211.08
    bandwidth[19] = 235.631
    cent_freq[20] = 2446.71
    bandwidth[20] = 255.255
    cent_freq[21] = 2701.97
    bandwidth[21] = 276.072
    cent_freq[22] = 2978.04
    bandwidth[22] = 298.126
    cent_freq[23] = 3276.17
    bandwidth[23] = 321.465
    cent_freq[24] = 3597.63
    bandwidth[24] = 346.136

    W = np.array(
        [0.003, 0.003, 0.003, 0.007, 0.010, 0.016, 0.016, 0.017, 0.017, 0.022, 0.027, 0.028, 0.030, 0.032, 0.034, 0.035,
         0.037, 0.036, 0.036, 0.033, 0.030, 0.029, 0.027, 0.026,
         0.026])

    bw_min = bandwidth[0]
    min_factor = np.exp(-30.0 / (2.0 * 2.303))  # % -30 dB point of filter

    all_f0 = np.zeros((num_crit,))
    crit_filter = np.zeros((num_crit, int(n_fftby2)))
    j = np.arange(0, n_fftby2)

    for i in range(num_crit):
        f0 = (cent_freq[i] / max_freq) * (n_fftby2)
        all_f0[i] = np.floor(f0)
        bw = (bandwidth[i] / max_freq) * (n_fftby2)
        norm_factor = np.log(bw_min) - np.log(bandwidth[i])
        crit_filter[i, :] = np.exp(-11 * (((j - np.floor(f0)) / bw) ** 2) + norm_factor)
        crit_filter[i, :] = crit_filter[i, :] * (crit_filter[i, :] > min_factor)

    num_frames = len(clean_speech) / skiprate - (winlength / skiprate)  # number of frames
    start = 1  # starting sample

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    scale = np.sqrt(1.0 / hannWin.sum() ** 2)

    f, t, Zxx = stft(clean_speech[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs, window=hannWin,
                     nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False, return_onesided=True,
                     boundary=None, padded=False)
    clean_spec = np.power(np.abs(Zxx) / scale, 2)
    clean_spec = clean_spec[:-1, :]

    f, t, Zxx = stft(processed_speech[0:int(num_frames) * skiprate + int(winlength - skiprate)], fs=fs, window=hannWin,
                     nperseg=winlength, noverlap=winlength - skiprate, nfft=n_fft, detrend=False, return_onesided=True,
                     boundary=None, padded=False)
    proc_spec = np.power(np.abs(Zxx) / scale, 2)
    proc_spec = proc_spec[:-1, :]

    clean_energy = (crit_filter.dot(clean_spec))
    log_clean_energy = 10 * np.log10(clean_energy)
    log_clean_energy[log_clean_energy < -100] = -100
    proc_energy = (crit_filter.dot(proc_spec))
    log_proc_energy = 10 * np.log10(proc_energy)
    log_proc_energy[log_proc_energy < -100] = -100

    log_clean_energy_slope = np.diff(log_clean_energy, axis=0)
    log_proc_energy_slope = np.diff(log_proc_energy, axis=0)

    dBMax_clean = np.max(log_clean_energy, axis=0)
    dBMax_processed = np.max(log_proc_energy, axis=0)

    numFrames = log_clean_energy_slope.shape[-1]

    clean_loc_peaks = np.zeros_like(log_clean_energy_slope)
    proc_loc_peaks = np.zeros_like(log_proc_energy_slope)
    for ii in range(numFrames):
        clean_loc_peaks[:, ii] = find_loc_peaks(log_clean_energy_slope[:, ii], log_clean_energy[:, ii])
        proc_loc_peaks[:, ii] = find_loc_peaks(log_proc_energy_slope[:, ii], log_proc_energy[:, ii])

    Wmax_clean = Kmax / (Kmax + dBMax_clean - log_clean_energy[:-1, :])
    Wlocmax_clean = Klocmax / (Klocmax + clean_loc_peaks - log_clean_energy[:-1, :])
    W_clean = Wmax_clean * Wlocmax_clean

    Wmax_proc = Kmax / (Kmax + dBMax_processed - log_proc_energy[:-1])
    Wlocmax_proc = Klocmax / (Klocmax + proc_loc_peaks - log_proc_energy[:-1, :])
    W_proc = Wmax_proc * Wlocmax_proc

    W = (W_clean + W_proc) / 2.0

    distortion = np.sum(W * (log_clean_energy_slope - log_proc_energy_slope) ** 2, axis=0)
    distortion = distortion / np.sum(W, axis=0)
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion) * alpha))]
    return np.mean(distortion)


def llr(clean_speech, processed_speech, fs, used_for_composite=False, frameLen=0.03, overlap=0.75):
    """
    Calculate the log-likelihood ratio (LLR) for the processed speech.

    Parameters:
        clean_speech (ndarray): The clean speech signal.
        processed_speech (ndarray): The processed speech signal.
        fs (int): The sampling frequency.
        used_for_composite (bool, optional): Whether the LLR is used for composite measure or not. Defaults to False.
        frameLen (float, optional): The length of each frame in seconds. Defaults to 0.03.
        overlap (float, optional): The overlap between frames as a fraction of the frame length. Defaults to 0.75.

    Returns:
        float: The mean log-likelihood ratio.

    Notes:
        - The log-likelihood ratio is used to measure the distortion of the processed speech compared to the clean speech.
        - The LLR is calculated using the autocorrelation of the LPC coefficients (Linear Prediction Coefficients) of the clean and processed speech.
        - The LLR is then sorted and the mean is taken.
        - If used_for_composite is False, the LLR values greater than 2 are set to 2. This is not in the composite measure but in the LLR matlab implementation of Loizou.
    """
    eps = np.finfo(np.float64).eps
    alpha = 0.95
    winlength = round(frameLen * fs)  # window length in samples
    skiprate = int(np.floor((1 - overlap) * frameLen * fs))  # window skip in samples
    if fs < 10000:
        P = 10  # LPC Analysis Order
    else:
        P = 16  # this could vary depending on sampling frequency.

    hannWin = 0.5 * (1 - np.cos(2 * np.pi * np.arange(1, winlength + 1) / (winlength + 1)))
    clean_speech_framed = extract_overlapped_windows(clean_speech + eps, winlength, winlength - skiprate, hannWin)
    processed_speech_framed = extract_overlapped_windows(processed_speech + eps, winlength, winlength - skiprate,
                                                         hannWin)
    numFrames = clean_speech_framed.shape[0]
    numerators = np.zeros((numFrames - 1,))
    denominators = np.zeros((numFrames - 1,))

    for ii in range(numFrames - 1):
        A_clean, R_clean = lpcoeff(clean_speech_framed[ii, :], P)
        A_proc, R_proc = lpcoeff(processed_speech_framed[ii, :], P)

        numerators[ii] = A_proc.dot(toeplitz(R_clean).dot(A_proc.T))
        denominators[ii] = A_clean.dot(toeplitz(R_clean).dot(A_clean.T))

    frac = numerators / (denominators)
    frac[np.isnan(frac)] = np.inf
    frac[frac <= 0] = 1000
    distortion = np.log(frac)
    if not used_for_composite:
        distortion[
            distortion > 2] = 2  # this line is not in composite measure but in llr matlab implementation of loizou
    distortion = np.sort(distortion)
    distortion = distortion[:int(round(len(distortion) * alpha))]
    return np.mean(distortion)


def composite_metrics(clean_speech, processed_speech, fs):
    """
    Calculate the composite metrics for the processed speech.

    Args:
        clean_speech (ndarray): The clean speech signal.
        processed_speech (ndarray): The processed speech signal.
        fs (int): The sampling frequency.

    Returns:
        dict: A dictionary containing the composite metrics:
            - CSIG (float): Composite signal-to-interference+gradient ratio.
            - CBAK (float): Composite background quality.
            - COVL (float): Composite over-all quality.
            - wss_dist (float): Weighted signal to noise+distortion ratio.
            - llr_mean (float): Mean log-likelihood ratio.
            - PESQ (float): Perceptual Evaluation of Speech Quality.
            - SSNR (float): Signal-to-noise ratio segment-wise.
    """
    # Calculate the weighted signal to noise+distortion ratio
    wss_dist = wss(clean_speech, processed_speech, fs)

    # Calculate the mean log-likelihood ratio
    llr_mean = llr(clean_speech, processed_speech, fs, used_for_composite=True)

    # Calculate the signal-to-noise ratio segment-wise
    segSNR = SNRseg(clean_speech, processed_speech, fs)

    # Calculate the Perceptual Evaluation of Speech Quality
    used_pesq_val = pesq.pesq(fs, clean_speech, processed_speech, 'wb')

    # Calculate the composite signal-to-interference+gradient ratio
    Csig = 3.093 - 1.029 * llr_mean + 0.603 * used_pesq_val - 0.009 * wss_dist
    Csig = np.max((1, Csig))
    Csig = np.min((5, Csig))  # limit values to [1, 5]

    # Calculate the composite background quality
    Cbak = 1.634 + 0.478 * used_pesq_val - 0.007 * wss_dist + 0.063 * segSNR
    Cbak = np.max((1, Cbak))
    Cbak = np.min((5, Cbak))  # limit values to [1, 5]

    # Calculate the composite over-all quality
    Covl = 1.594 + 0.805 * used_pesq_val - 0.512 * llr_mean - 0.007 * wss_dist
    Covl = np.max((1, Covl))
    Covl = np.min((5, Covl))  # limit values to [1, 5]

    return {
        'CSIG': Csig,
        'CBAK': Cbak,
        'COVL': Covl,
        'wss_dist': wss_dist,
        'llr_mean': llr_mean,
        'PESQ': used_pesq_val,
        'SSNR': segSNR
    }