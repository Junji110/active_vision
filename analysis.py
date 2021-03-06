import numpy as np
import scipy as sp
import scipy.signal as spsig
import scipy.fftpack as fftpack
import scipy.signal.signaltools as signaltools
from scipy.signal.windows import get_window
from six import string_types
import warnings
import quantities as pq

import stats

def gen_tapered_window(len_window, len_taper, taper_func=np.hamming):
    window = np.ones(len_window)
    window[:len_taper] = taper_func(len_taper * 2)[:len_taper]
    window[-len_taper:] = taper_func(len_taper * 2)[-len_taper:]
    return window


def gen_segment_indices(idx_trig, idx_range):
    idxs_segments = []
    idx_ini, idx_fin = idx_range
    for i in idx_trig:
        idxs_segments.extend(range(i+idx_ini, i+idx_fin))
    return np.array(idxs_segments)


def segment_sequence(data, idx_trig, idx_range):
    data_asarray = np.asarray(data)
    idx_pick =  gen_segment_indices(idx_trig, idx_range)
    return data_asarray[idx_pick].reshape((len(idx_trig), -1))


def butterworth_filter(signal, Fs, highpassfreq=None, lowpassfreq=None, order=4, filtfunc='filtfilt'):
    """
    Apply Butterworth filter to a given signal. Filter type is determined
    according to how values of highpassfreq and lowpassfreq are given:
    
        highpassfreq < lowpassfreq:    bandpass filter
        highpassfreq > lowpassfreq:    bandstop filter
        highpassfreq only (lowpassfreq = None):    highpass filter
        lowpassfreq only (highpassfreq = None):    lowpass filter
    
    **Args**:
    signal: 1D array_like
        signal to be filtered
    Fs: float
        sampling rate of the signal
    highpassfreq: float
        lower boundary of the pass-band. Negative values are replaced by None (i.e., no highpass filtering).
    lowpassreq: float
        higher boundary of the pass-band. Values larger than Fs/2 (the Nyquist frequency) are replaced by None (i.e.,
        no lowpass filtering.
    order: int (default: 4)
        Order of Butterworth filter.
    filtfunc: string (default: 'filtfilt')
        Filtering function to be used. Either 'filtfilt' (scipy.signal.filtfilt)
        or 'lfilter' (scipy.signal.lfilter)
    
    **Return**:
    signal_out : float 1D-array
        Filtered signal
    """
    Fn = Fs / 2.
    
    # set the function for filtering
    if filtfunc is 'lfilter':
        ffunc = spsig.lfilter
    elif filtfunc is 'filtfilt':
        ffunc = spsig.filtfilt
    else:
        raise ValueError("filtfunc must to be either 'filtfilt' or 'lfilter'")
    
    # set parameters
    if highpassfreq <= 0:
        highpassfreq = None
    if lowpassfreq >= Fs / 2:
        lowpassfreq = None
    if lowpassfreq and highpassfreq:
        if highpassfreq < lowpassfreq:
            Wn = (highpassfreq / Fn, lowpassfreq / Fn)
            btype = 'bandpass'
        else:
            Wn = (lowpassfreq / Fn, highpassfreq / Fn)
            btype = 'bandstop'
    elif lowpassfreq:
        Wn = lowpassfreq / Fn
        btype = 'lowpass'
    elif highpassfreq:
        Wn = highpassfreq / Fn
        btype = 'highpass'
    else:
        raise ValueError("Specify highpassfreq and/or lowpathfreq")
    
    # filter design
    b, a = spsig.butter(order, Wn, btype=btype)
        
    return ffunc(b, a, signal)

def wavelet_transform(signal, freq, nco, Fs):
    '''
    Compute the wavelet transform of a given signal with Morlet mother wavelet.
    The definition of the wavelet is based on Le van Quyen et al. J
    Neurosci Meth 111:83-98 (2001).
    
    **Args**:
    signal : 1D or 2D array_like
        Signal to be transformed
    freq : float
        Center frequency of the Morlet wavelet.
    nco : float
        Size of the mother wavelet (approximate number of cycles within a
        wavelet). A larger nco value leads to a higher frequency resolution but
        a lower temporal resolution, and vice versa. Typically used values are
        in a range of 3 - 8.
    Fs : float
        Sampling rate of the signal.
    
    **Return**:
    signal_trans: 1D complex array
        Transformed signal
    '''
    # Morlet wavelet generator (c.f. Le van Quyen et al. J Neurosci Meth
    # 111:83-98 (2001))
    def morlet_wavelet(freq, nco, Fs, N):
        sigma = nco / (6. * freq)  
        t = (np.arange(N, dtype='float') - N / 2) / Fs
        if N % 2 == 0:
            t = np.roll(t, N / 2) 
        else:
            t = np.roll(t, N / 2 + 1) 
        return np.sqrt(freq) * np.exp(-(t * t) / (2 * sigma ** 2) + 1j * 2 * np.pi * freq * t)
            
    # check whether the given central frequency is less than the Nyquist
    # frequency of the signal
    if freq >= Fs / 2:
        raise ValueError("freq must be less than the half of Fs (sampling rate of the original signal)")
   
    fft = np.fft.fft
    ifft = np.fft.ifft
    ndim_signal = signal.ndim
    if ndim_signal == 1:
        num_ch = 1
        N = signal.shape[0]
    else:
        num_ch, N = signal.shape

    # zero-padding to a power of 2 for efficient convolution
    N_pow2 = 2 ** (int(np.log2(N)) + 1)  # the least power of 2 greater than N
    tmpdata = np.zeros((num_ch, N_pow2))
    tmpdata[:, 0:N] = np.asarray(signal)
    
    # generate Morlet wavelet
    wavelet = morlet_wavelet(freq, nco, Fs, N_pow2)
    
    # convolution of the signal with the wavelet
    signal_wt = ifft(fft(tmpdata) * fft(wavelet))
    if ndim_signal == 1:
        return signal_wt[0, 0:N]
    else:
        return signal_wt[:, 0:N]

def segment(data, idx_trig, idx_ini, idx_fin):
    idx_pick = []
    for i in idx_trig:
        idx_pick.extend(range(i + idx_ini, i + idx_fin))
    return data[idx_pick].reshape((len(idx_trig), idx_fin - idx_ini))

def segmentPSD(data, idx_trig, idx_ini, idx_fin, pad_to):
    len_seg = idx_fin - idx_ini
    segs = segment(data, idx_trig, idx_ini, idx_fin)
    segs_FT = np.fft.fft(segs * np.hanning(len_seg), n=pad_to, axis=1)
    return np.abs(segs_FT) ** 2
    
def bin_spike_train(spk, bin_width, timerange=(None, None), bin_step=None, clip=False):
    '''
    bin a given spike train with given binning parameters

    Arguments
    ---------
    spk : array-like
    Spike times.
    bin_width : float
    Temporal width of bins
    t_start, t_stop: float, optional
    Starting and ending time points of binning. If not given, determined from spk as the first and the last spike times -/+ bin_width, respectively.
    bin_step : float, optional
    interval between neighboring bins. If not given, identical to bin_width.
    clip : boolean, optional
    When True, spike counts of the bins with multiple spikes are clipped to 1. Default is False.

    Returns
    -------
    spike_counts : spike counts for each bin
    bin_pos : temporal positions of (the centers of) bins
    '''
    spk_arr = np.asarray(spk)
    t_start, t_stop = timerange
    if t_start is None:
        t_start = spk_arr[0] - bin_width
    if t_stop is None:
        t_stop = spk_arr[-1] + bin_width
    if bin_step is None:
        bin_step = bin_width

    # define bins. The second argument of np.arrange should be some number
    # between (t_stop - bin_width) and (t_stop - bin_width + bin_step) so that
    # the last bin is correctly defined.
    ts_left = np.arange(t_start, t_stop - bin_width + bin_step / 2, bin_step)
    num_bin = ts_left.size

    # count spikes that fall into each bin
    spike_counts = np.zeros(num_bin, int)
    if bin_width == bin_step:
        # case of exclusive binning
        idx_spk = ((spk_arr[(t_start < spk_arr) & (spk_arr < t_stop)] - t_start) / bin_width).astype(int)
        if clip:
            spike_counts[idx_spk] = 1
        else:
            for i in idx_spk:
                spike_counts[i] += 1
    else:
        # case of non-exclusive binning
        ts_right = ts_left + bin_width
        for i_bin in xrange(num_bin):
            spk_bin = spk_arr[(ts_left[i_bin] < spk_arr) & (spk_arr < ts_right[i_bin])]
            spike_counts[i_bin] = spk_bin.size
        if clip:
            spike_counts[spike_counts > 1] = 1

    return spike_counts, ts_left + bin_width / 2

def period_histogram(binned_spike, phase, num_phasebin, masks=None):
    pick = binned_spike.astype(bool)
    if masks is not None:
        for mask in masks:
            pick = pick & mask
    return np.histogram(phase[pick], num_phasebin, (-np.pi, np.pi), True)

def phase_locking_value(binned_spike, phase, masks=None):
    pick = binned_spike.astype(bool)
    if masks is not None:
        for mask in masks:
            pick = pick & mask
    return stats.circ_r(phase[pick])

def idx2mask(idx, n):
    if np.max(idx) >= n:
        raise ValueError("The maximum index must be less than n.")
    mask = np.zeros(n, dtype=bool)
    mask[idx] = True
    return mask

def percentile_mask(x, q, mode="above"):
    '''
    Generate an boolean array according to the comparison between the values of the elements of an
    given array and a given percentile of the element values.
    
    Arguments
    ---------
    x : array
    input array
    q : float (0-100)
    percentile
    mode : ["above", "below"], optional (default is "above")
    When "above", True for x > q-percentile, and vice versa. "below" is the opposite of "above".
  
    Returns
    -------
    mask : array
    a boolean array of the same size as x, containing True and False according to the comparison of x and q-percentile.
    '''
    threshold = np.percentile(x, q)
    if mode is "above":
        return x > threshold
    elif mode is "below":
        return x < threshold
    else:
        raise ValueError("Undefined mode '{0}'".format(mode))

def event_mask(idx_ev, idx_range, masklen):
    idx = []
    for i in idx_ev:
        idx.extend(range(i + idx_range[0], i + idx_range[1]))
    return idx2mask(idx, masklen)

def estimateCSD(LFP, h, R, sigma):
    '''
    Current source density estimation based on the inverse CSD method by
    Pettersen et al., 2006
    
    Arguments
    ---------
        LFP : 2 dimensional array (unit: mV)
            LFP recordings from multiple channels (1st dimension) at multiple
            time points (2nd dimension)
        h : float (unit: mm)
            Spatial separation of neighboring channels
        R : float (unit: mm)
            Radius of current source (c.f. Pettersen et al., 2006)
        sigma : float (unit: S/m)
            Electric conductivity of the extracellular medium
    
    Returns
    -------
        CSD : 2 dimensional array (unit: uA/mm3)
            Estimated CSD (same shape as LFP)
    '''
    
    def forward_matrix(N_ch, h, R, sigma):
        '''
        Return the matrix that converts CSD to LFP
        '''
        F = np.zeros((N_ch, N_ch))
        for i in range(N_ch):
            for j in range(0, i+1):
                F[i,j] = np.sqrt((j - i)**2 + (R / h)**2) - np.abs(j - i)
        for i in range(N_ch):
            for j in range(i, N_ch):
                F[i,j] = F[j, i]
        return F * h**2 / (2 * sigma)
    
    N_ch = LFP.shape[0]
    F = forward_matrix(N_ch, h, R, sigma)
    
    # solve the inverse problem of LFP = F * CSD
    return np.linalg.solve(F, LFP)

def smoothing_matrix_3p(N_ch):
    smthmat = np.zeros((N_ch, N_ch))
    for ch in range(N_ch):
        if ch == 0:
            smthmat[0][0:2] = [3, 1] 
        elif ch == N_ch - 1:
            smthmat[N_ch-1][N_ch-2:N_ch] = [1, 3]
        else:
            smthmat[ch][ch-1:ch+2] = [1, 2, 1]
    return smthmat / 4.

def interpolation_matrix(N_ch, badch):
    '''
    Interpolate values in bad channels keeping the 2nd derivative smooth.
    NOTE: bad channels must be separated at least by 2 channels.
    '''
    if not (0 <= min(badch) and max(badch) < N_ch):
        raise ValueError("Bad channel ID must be within the range of [0, %d]" % N_ch)

    itplmat = np.eye(N_ch)
    for ch in badch:
        if ch == 0:
            itplmat[ch][0:3] = [0., 2., -1.]
        elif ch == 1:
            itplmat[ch][0:4] = [1./3, 0., 1., -1./3]
        elif ch == N_ch - 2:
            itplmat[ch][N_ch-4:N_ch] = [-1./3, 1., 0., 1./3]
        elif ch == N_ch - 1:
            itplmat[ch][N_ch-3:N_ch] = [-1., 2., 0.]
        else:
            itplmat[ch][ch-2:ch+3] = [-1./6, 2./3, 0, 2./3, -1./6]
    return itplmat


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # a sample for cohere()
    fs = 1000.0
    t = np.arange(0, 1, 1/fs)
    x = np.cos(2*np.pi*50.0*t) + 0.5 * np.random.normal(0, 1, len(t))
    y = np.sin(2*np.pi*50.0*t) + 0.5 * np.random.normal(0, 1, len(t))
    freqs, coherency, phaselag = cohere(x, y, fs=fs)

    plt.subplot(311)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.grid()
    plt.xlabel("Time (s)")
    plt.subplot(312)
    plt.plot(freqs[:len(freqs)/2], coherency[:len(freqs)/2])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherency")
    plt.subplot(313)
    plt.plot(freqs[:len(freqs)/2], phaselag[:len(freqs)/2])
    plt.grid()
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase lag (rad)")
    plt.show()
