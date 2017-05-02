import numpy as np
import scipy as sp
import scipy.signal as spsig
import scipy.fftpack as fftpack
import scipy.signal.signaltools as signaltools
from scipy.signal.windows import get_window
from six import string_types
import warnings
import quantities as pq
import neo

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

def welch(x, y, fs=1.0, window='hanning', nperseg=256, noverlap=None,
          nfft=None, detrend='constant', scaling='density', axis=-1):
        """

        Estimate cross spectral density using Welch's method.

        Welch's method [1]_ computes an estimate of the cross spectral density
        by dividing the data into overlapping segments, computing a modified
        periodogram for each segment and averaging the cross-periodograms. This
        function is a slightly modified version of `scipy.signal.welch()` with
        modifications based on `matplotlib.mlab._spectral_helper()`.

        Parameters
        ----------
        x, y : array_like
            Time series of measurement values
        fs : float, optional
            Sampling frequency of the `x` time series in units of Hz. Defaults
            to 1.0.
        window : str or tuple or array_like, optional
            Desired window to use. See `get_window` for a list of windows and
            required parameters. If `window` is array_like it will be used
            directly as the window and its length will be used for nperseg.
            Defaults to 'hanning'.
        nperseg : int, optional
            Length of each segment.  Defaults to 256.
        noverlap: int, optional
            Number of points to overlap between segments. If None,
            ``noverlap = nperseg / 2``.  Defaults to None.
        nfft : int, optional
            Length of the FFT used, if a zero padded FFT is desired.  If None,
            the FFT length is `nperseg`. Defaults to None.
        detrend : str or function, optional
            Specifies how to detrend each segment. If `detrend` is a string,
            it is passed as the ``type`` argument to `detrend`. If it is a
            function, it takes a segment and returns a detrended segment.
            Defaults to 'constant'.
        scaling : { 'density', 'spectrum' }, optional
            Selects between computing the power spectral density ('density')
            where Pxx has units of V**2/Hz if x is measured in V and computing
            the power spectrum ('spectrum') where Pxx has units of V**2 if x is
            measured in V. Defaults to 'density'.
        axis : int, optional
            Axis along which the periodogram is computed; the default is over
            the last axis (i.e. ``axis=-1``).

        Returns
        -------
        f : ndarray
            Array of sample frequencies.
        Pxy : ndarray
            Cross spectral density or cross spectrum of x and y.

        Notes
        -----
        An appropriate amount of overlap will depend on the choice of window
        and on your requirements.  For the default 'hanning' window an
        overlap of 50% is a reasonable trade off between accurately estimating
        the signal power, while not over counting any of the data.  Narrower
        windows may require a larger overlap.

        If `noverlap` is 0, this method is equivalent to Bartlett's method [2]_.

        References
        ----------
        .. [1] P. Welch, "The use of the fast Fourier transform for the
               estimation of power spectra: A method based on time averaging
               over short, modified periodograms", IEEE Trans. Audio
               Electroacoust. vol. 15, pp. 70-73, 1967.
        .. [2] M.S. Bartlett, "Periodogram Analysis and Continuous Spectra",
               Biometrika, vol. 37, pp. 1-16, 1950.
        """
        # The checks for if y is x are so that we can use the same function to
        # obtain both power spectrum and cross spectrum without doing extra
        # calculations.
        same_data = y is x
        # Make sure we're dealing with a numpy array. If y and x were the same
        # object to start with, keep them that way
        x = np.asarray(x)
        if same_data:
            y = x
        else:
            if x.shape != y.shape:
                raise ValueError("x and y must be of the same shape.")
            y = np.asarray(y)

        if x.size == 0:
            return np.empty(x.shape), np.empty(x.shape)

        if axis != -1:
            x = np.rollaxis(x, axis, len(x.shape))
            if not same_data:
                y = np.rollaxis(y, axis, len(y.shape))

        if x.shape[-1] < nperseg:
            warnings.warn('nperseg = %d, is greater than x.shape[%d] = %d, using '
                          'nperseg = x.shape[%d]'
                          % (nperseg, axis, x.shape[axis], axis))
            nperseg = x.shape[-1]

        if isinstance(window, string_types) or type(window) is tuple:
            win = get_window(window, nperseg)
        else:
            win = np.asarray(window)
            if len(win.shape) != 1:
                raise ValueError('window must be 1-D')
            if win.shape[0] > x.shape[-1]:
                raise ValueError('window is longer than x.')
            nperseg = win.shape[0]

        if scaling == 'density':
            scale = 1.0 / (fs * (win*win).sum())
        elif scaling == 'spectrum':
            scale = 1.0 / win.sum()**2
        else:
            raise ValueError('Unknown scaling: %r' % scaling)

        if noverlap is None:
            noverlap = nperseg // 2
        elif noverlap >= nperseg:
            raise ValueError('noverlap must be less than nperseg.')

        if nfft is None:
            nfft = nperseg
        elif nfft < nperseg:
            raise ValueError('nfft must be greater than or equal to nperseg.')

        if not hasattr(detrend, '__call__'):
            detrend_func = lambda seg: signaltools.detrend(seg, type=detrend)
        elif axis != -1:
            # Wrap this function so that it receives a shape that it could
            # reasonably expect to receive.
            def detrend_func(seg):
                seg = np.rollaxis(seg, -1, axis)
                seg = detrend(seg)
                return np.rollaxis(seg, axis, len(seg.shape))
        else:
            detrend_func = detrend

        step = nperseg - noverlap
        indices = np.arange(0, x.shape[-1]-nperseg+1, step)

        for k, ind in enumerate(indices):
            x_dt = detrend_func(x[..., ind:ind+nperseg])
            xft = fftpack.fft(x_dt*win, nfft)
            if same_data:
                yft = xft
            else:
                y_dt = detrend_func(y[..., ind:ind+nperseg])
                yft = fftpack.fft(y_dt*win, nfft)
            if k == 0:
                Pxy = (xft * yft.conj())
            else:
                Pxy *= k/(k+1.0)
                Pxy += (xft * yft.conj()) / (k+1.0)
        Pxy *= scale
        f = fftpack.fftfreq(nfft, 1.0/fs)

        if axis != -1:
            Pxy = np.rollaxis(Pxy, -1, axis)

        return f, Pxy

def cohere(x, y, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
              fs=1.0, window='hanning', nfft=None, detrend='constant',
              scaling='density', axis=-1):
    """
    Estimates power spectrum density (PSD) of a given AnalogSignal using
    Welch's method, which works in the following steps:
        1. cut the given data into several overlapping segments. The degree of
            overlap can be specified by parameter *overlap* (default is 0.5,
            i.e. segments are overlapped by the half of their length).
            The number and the length of the segments are determined according
            to parameter *num_seg*, *len_seg* or *freq_res*. By default, the
            data is cut into 8 segments.
        2. apply a window function to each segment. Hanning window is used by
            default. This can be changed by giving a window function or an
            array as parameter *window* (for details, see the docstring of
            `scipy.signal.welch()`)
        3. compute the periodogram of each segment
        4. average the obtained periodograms to yield PSD estimate
    These steps are implemented in `scipy.signal`, and this function is a
    wrapper which provides a proper set of parameters to
    `scipy.signal.welch()`. Some parameters for scipy.signal.welch(), such as
    `nfft`, `detrend`, `window`, `return_onesided` and `scaling`, also works
    for this function.

    Parameters
    ----------
    signal: Neo AnalogSignalArray or Quantity array or Numpy ndarray
        Time series data, of which PSD is estimated. When a Quantity array or
        Numpy ndarray is given, sampling frequency should be given through the
        keyword argument `fs`, otherwise the default value (`fs=1.0`) is used.
    num_seg: int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if *len_seg* or *freq_res* is given. Default is 8.
    len_seg: int, optional
        Length of segments. This parameter is ignored if *freq_res* is given.
        Default is None (determined from other parameters).
    freq_res: Quantity or float, optional
        Desired frequency resolution of the obtained PSD estimate in terms of
        the interval between adjacent frequency bins. When given as a float, it
        is taken as frequency in Hz. Default is None (determined from other
        parameters).
    overlap: float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap). Default is 0.5 (half-overlapped).
    fs: Quantity array or float, optional
        Specifies the sampling frequency of the input time series. When the
        input is given as an AnalogSignalArray, the sampling frequency is taken
        from its attribute and this parameter is ignored. Default is 1.0.
    window, nfft, detrend, return_onesided, scaling, axis: optional
        These arguments are directly passed on to scipy.signal.welch(). See the
        respective descriptions in the docstring of `scipy.signal.welch()` for
        usage.

    Returns
    -------
    freqs: Quantity array or Numpy ndarray
        Frequencies associated with the power estimates in `psd`. `freqs` is
        always a 1-dimensional array irrespective of the shape of the input
        data. Quantity array is returned if `signal` is AnalogSignalArray or
        Quantity array. Otherwise Numpy ndarray containing frequency in Hz is
        returned.
    psd: Quantity array or Numpy ndarray
        PSD estimates of the time series in `signal`. Quantity array is
        returned if `data` is AnalogSignalArray or Quantity array. Otherwise
        Numpy ndarray is returned.
    """

    # initialize a parameter dict (to be given to scipy.signal.welch()) with
    # the parameters directly passed on to scipy.signal.welch()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'scaling': scaling, 'axis': axis}

    # When the input is AnalogSignalArray, the axis for time index is rolled to
    # the last
    xdata = np.asarray(x)
    ydata = np.asarray(y)
    if isinstance(x, neo.AnalogSignalArray):
        xdata = np.rollaxis(xdata, 0, len(xdata.shape))
        ydata = np.rollaxis(ydata, 0, len(ydata.shape))

    # if the data is given as AnalogSignalArray, use its attribute to specify
    # the sampling frequency
    if hasattr(x, 'sampling_rate'):
        params['fs'] = x.sampling_rate.rescale('Hz').magnitude
    else:
        params['fs'] = fs

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if freq_res is not None:
        if freq_res <= 0:
            raise ValueError("freq_res must be positive")
        dF = freq_res.rescale('Hz').magnitude \
            if isinstance(freq_res, pq.quantity.Quantity) else freq_res
        nperseg = int(params['fs'] / dF)
        if nperseg > xdata.shape[axis]:
            raise ValueError("freq_res is too high for the given data size")
    elif len_seg is not None:
        if len_seg <= 0:
            raise ValueError("len_seg must be a positive number")
        elif xdata.shape[axis] < len_seg:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_seg
    else:
        if num_seg <= 0:
            raise ValueError("num_seg must be a positive number")
        elif xdata.shape[axis] < num_seg:
            raise ValueError("num_seg must be smaller than the data length")
        # when only *num_seg* is given, *nperseg* is determined by solving the
        # following equation:
        #  num_seg * nperseg - (num_seg-1) * overlap * nperseg = data.shape[-1]
        #  -----------------   ===============================   ^^^^^^^^^^^
        # summed segment lengths        total overlap            data length
        nperseg = int(xdata.shape[axis] / (num_seg - overlap * (num_seg - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, Pxy = welch(xdata, ydata, **params)
    freqs, Pxx = welch(xdata, xdata, **params)
    freqs, Pyy = welch(ydata, ydata, **params)
    coherency = np.abs(Pxy)**2 / (np.abs(Pxx) * np.abs(Pyy))
    phaselag = np.angle(Pxy)

    # attach proper units to return values
    if isinstance(x, pq.quantity.Quantity):
        freqs = freqs * pq.Hz

    return freqs, coherency, phaselag

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
