import math
import warnings
import numpy as np

from scipy.ndimage.filters import maximum_filter1d
from scipy.signal import (butter, filtfilt, iirnotch, iirfilter, sosfilt, zpk2sos)


def good_beat_annotations(
        annotation
):
    """ Get rid of non-beat markers """
    #                                 '"NOR": ["N"], '
    #                                 '"LBBB": ["L"], '
    #                                 '"RBBB": ["R"], '
    #                                 '"APC": ["A"], '
    #                                 '"PVC": ["V"], '
    #                                 '"PACE": ["/"], '
    #                                 '"AP": ["a"], '
    #                                 '"VF": ["!"], '
    #                                 '"VFN": ["F"], '
    #                                 '"NE": ["j"], '
    #                                 '"FPN": ["f"], '
    #                                 '"VE": ["E"], '
    #                                 '"NP": ["J"], '
    #                                 '"AE": ["e"], '
    #                                 '"UN" : ["Q"]'
    # good = ['N', 'L', 'R', 'A', 'V', '/', 'a', '!', 'F', 'j', 'f', 'E', 'J', 'e', 'Q', 'S']
    good = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    # normal = ['N', 'L', 'R', 'j', 'e']
    # supra_ventricular = ['a', 'S', 'A', 'J']
    # ventricular = ['!', 'E', 'V']
    # fusion = ['F']
    # unknow = ['P', 'Q', 'f']

    ids = np.in1d(annotation.symbol, good)
    samples = annotation.sample[ids]
    symbols = np.asarray(annotation.symbol)[ids]

    return samples, symbols


def afib_annotations(
        annotation,
        convert2int=True
):
    """ Get rid of non-beat markers """

    normal = ['(AB', '(AFL', '(B', '(BII', '(IVR', '(N', '(NOD', '(P', '(PREX', '(SBR', '(SVTA', '(T', '(VFL', '(VT']

    afib = ['(AFIB']
    ids = np.in1d(annotation.aux_note, normal + afib)
    samples = annotation.sample[ids]
    symbols = np.asarray(annotation.aux_note)[ids]
    if convert2int:
        symbols = ['0' if s in normal else s for s in symbols]
        symbols = ['1' if s in afib else s for s in symbols]
        symbols = np.asarray(symbols)

    return samples, np.asarray(list(map(int, symbols)))


def bwr(raw, fs, l1=0.2, l2=0.6):
    flen1 = int(l1 * fs / 2)
    flen2 = int(l2 * fs / 2)

    if flen1 % 2 == 0:
        flen1 += 1

    if flen2 % 2 == 0:
        flen2 += 1

    out1 = smooth(raw, flen1)
    out2 = smooth(out1, flen2)

    return raw - out2


def bwr_smooth(
        raw,
        fs,
        l1=0.2,
        l2=0.6
):
    flen1 = int(l1 * fs / 2)
    flen2 = int(l2 * fs / 2)

    if flen1 % 2 == 0:
        flen1 += 1

    if flen2 % 2 == 0:
        flen2 += 1

    out1 = smooth(raw, flen1)
    out2 = smooth(out1, flen2)

    return raw - out2


def smooth(
        x,
        window_len=11,
        window='hanning'
):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this:
    return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise (ValueError, "smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise (ValueError, "Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window_len % 2 == 0:
        window_len += 1

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise (ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    # output = np.argwhere(np.isnan(y))
    # if len(output) > 0:
    #     print(output)

    return y[int(window_len / 2):-int(window_len / 2)]


def norm(
        raw,
        window_len,
        samp_from=-1,
        samp_to=-1
):
    # The window size is the number of samples that corresponds to the time analogue of 2e = 0.5s
    if window_len % 2 == 0:
        window_len += 1

    abs_raw = np.abs(raw)
    # Remove outlier
    while True:
        g = maximum_filter1d(abs_raw, size=window_len)
        if np.max(abs_raw) < 5.0:
            break

        abs_raw[g > 5.0] = 0

    g_smooth = smooth(g, window_len, window='hamming')
    g_mean = max(np.mean(g_smooth) / 2.0, 0.1)
    g_smooth = np.clip(g_smooth, g_mean, None)
    g_smooth[g_smooth < 0.01] = 1
    normalized = np.divide(raw, g_smooth)

    return normalized


def butter_lowpass(
        cutoff,
        fs,
        order=5
):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    return b, a


def norm2(raw, baseline, window_len, fs):
    # The window size is the number of samples that corresponds to the time analogue of 2e = 0.5s
    if window_len % 2 == 0:
        window_len += 1

    abs_raw = abs(raw)

    baseline = smooth(baseline, window_len=int(2.5 * fs), window='hanning')
    baseline += np.median(abs_raw) / 2
    crossings = raw - baseline
    start_crossings = len(np.flatnonzero(np.diff(np.sign(crossings))))

    num_up_crossings = start_crossings
    while num_up_crossings > (0.1 * start_crossings):
        baseline = baseline + 0.05
        crossings = raw - baseline
        num_up_crossings = len(np.flatnonzero(np.diff(np.sign(crossings))))

    g = maximum_filter1d(abs_raw, size=window_len)
    g_smooth = smooth(g, window_len, window='hamming')
    g_mean = np.mean(baseline) / 2.0
    g_max = np.mean(baseline)
    g_smooth = np.clip(g_smooth, g_mean, g_max)
    g_smooth[g_smooth < 0.01] = 1
    normalized = np.divide(raw, g_smooth)

    return normalized


def butter_lowpass_filter(
        data,
        cutoff,
        fs,
        order=5
):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_bandpass(
        lowcut,
        highcut,
        fs,
        order=5
):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    return b, a


def butter_bandpass_filter(
        data,
        lowcut,
        highcut,
        fs,
        order=5
):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)

    return y


def butter_notch_filter(
        x,
        fscut,
        fs,
        Q=30.0
):
    w0 = fscut / (fs / 2)  # Normalized Frequency
    # Design notch filter
    b, a = iirnotch(w0, Q)
    y = filtfilt(b, a, x)

    return y


def butter_highpass(
        cutoff,
        fs,
        order=5
):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)

    return b, a


def butter_highpass_filter(
        data,
        cutoff,
        fs,
        order=5
):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def multi_butter_bandpass_filter(
        data,
        lowcut,
        highcut,
        fs,
        order=5
):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y0 = filtfilt(b, a, data[:, 0])
    y1 = filtfilt(b, a, data[:, 1])
    y2 = filtfilt(b, a, data[:, 2])

    return np.vstack((y0, y1, y2)).T


def iir_bandpass(
        data,
        freqmin,
        freqmax,
        df,
        corners=4,
        zerophase=True
):
    """
    :copyright:
    The ObsPy Development Team (devs@obspy.org)

    Butterworth-Bandpass Filter.

    Filter data from ``freqmin`` to ``freqmax`` using ``corners``
    corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freqmin: Pass band low corner frequency.
    :param freqmax: Pass band high corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the filter order but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    low = freqmin / fe
    high = freqmax / fe
    # raise for some bad scenarios
    if high - 1.0 > -1e-6:
        msg = (f"Selected high corner frequency ({freqmax}) of bandpass is at or "
               "above Nyquist ({fe}). Applying a high-pass instead.")
        warnings.warn(msg)
        return highpass(data, freq=freqmin, df=df, corners=corners,
                        zerophase=zerophase)
    if low > 1:
        msg = "Selected low corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, [low, high], btype='band', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, float(k))
    if zerophase:
        firstpass = sosfilt(sos, data)

        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def highpass(
        data,
        freq,
        df,
        corners=4,
        zerophase=False
):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency ``freq`` using
    ``corners`` corners.
    The filter uses :func:`scipy.signal.iirfilter` (for design)
    and :func:`scipy.signal.sosfilt` (for applying the filter).

    :type data: numpy.ndarray
    :param data: Data to filter.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz.
    :param corners: Filter corners / order.
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    f = freq / fe
    # raise for some bad scenarios
    if f > 1:
        msg = "Selected corner frequency is above Nyquist."
        raise ValueError(msg)
    z, p, k = iirfilter(corners, f, btype='highpass', ftype='butter', output='zpk')
    sos = zpk2sos(z, p, float(k))
    if zerophase:
        firstpass = sosfilt(sos, data)

        return sosfilt(sos, firstpass[::-1])[::-1]
    else:
        return sosfilt(sos, data)


def eclipse_distance(
        a,
        b
):
    return math.sqrt(math.pow((a - b), 2))


def calculate_slope(
        ecg,
        index_beat,
        fs
):
    # mean slope of the waveform at that position
    slope = np.mean(np.diff(ecg[index_beat - round(0.075 * fs): index_beat]))

    return slope
