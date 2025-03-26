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
