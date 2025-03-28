import numpy as np
from scipy.signal import firwin, lfilter

def octave_band_filter(x, fs, fc, N=1024):
    """1/1オクターブバンドフィルター"""
    fl = fc / np.sqrt(2) / (fs / 2)
    fh = fc * np.sqrt(2) / (fs / 2)
    b = firwin(N, [fl, fh], pass_zero=False)
    y = lfilter(b, 1, np.append(x, np.zeros(N)))
    return y[N // 2 : len(y) - N // 2]

def schroeder_curve(y):
    """Schroeder積分曲線"""
    sc = np.cumsum(y[::-1]**2)[::-1]
    sc_db = 10 * np.log10(sc / np.max(sc))
    return sc_db

def reverb_time_T30(y, fs):
    """T30（RT60）をSchroeder法で推定"""
    sc = schroeder_curve(y)
    t = np.arange(len(y)) / fs
    fit_start = np.sum(sc >= -5)
    fit_end = np.sum(sc >= -35)
    z = np.polyfit(t[fit_start:fit_end], sc[fit_start:fit_end], 1)
    T30 = -60 / z[0]
    return T30

def cut_signal_by_threshold(signal: np.ndarray, threshold_db: float) -> np.ndarray:
    """
    振幅しきい値で信号末尾をカットする関数（dB単位指定）。

    Parameters:
    - signal: 正規化済みの信号
    - threshold_db: カットする振幅のdB（例：-60）

    Returns:
    - カット後の信号（末尾）
    """
    threshold = 10 ** (threshold_db / 20)
    valid_indices = np.where(np.abs(signal) > threshold)[0]
    if len(valid_indices) > 0:
        return signal[:valid_indices[-1] + 1]
    else:
        return signal
