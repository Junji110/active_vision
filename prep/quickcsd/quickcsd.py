import os
import json

import numpy as np
import scipy.io as spio
import scipy.signal as spsig
import matplotlib.pyplot as plt

# from active_vision.fileio import lvdread
import lvdread


def find_lvdfilenames(datadir, session, rec):
    # identify the names of metadata files
    searchtoken = 'rec{0}_pc'.format(rec)
        
    filenames = os.listdir(datadir)
    fn_found = []
    for fn in filenames:
        if str(session) in fn and searchtoken in fn:
            fn_found.append("{0}/{1}".format(datadir, fn))
    if len(fn_found) == 0:
        raise IOError("LVD File not found.")

    return fn_found

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
        lower boundary of the pass-band.
    lowpassreq: float
        higher boundary of the pass-band.
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
    CSD = np.linalg.solve(F, LFP)
    
    return CSD

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


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    else:
        raise ValueError("Configuration file (conf.json) not found.")
    size = conf['quickcsd']['size']
    ori = conf['quickcsd']['ori']
    freq = conf['quickcsd']['freq']
    LFP_upfreq = conf['quickcsd']['LFP_freq_upper_bound']
    MUA_lowfreq = conf['quickcsd']['MUA_freq_lower_bound']
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--rawdir", default=conf['rawdir'])
    parser.add_argument("--prepdir", default=conf['prepdir'])
    parser.add_argument("--data", nargs=3, default=[20150714, 4, 1])
    parser.add_argument("--pc", type=int, default=1)
    parser.add_argument("--timerange", nargs=2, default=conf['quickcsd']['timerange'])
    parser.add_argument("--h", type=float, default=conf['quickcsd']['h'])
    parser.add_argument("--R", type=float, default=conf['quickcsd']['R'])
    parser.add_argument("--sigma", type=float, default=conf['quickcsd']['sigma'])
    parser.add_argument("--smooth", type=bool, default=False)
    parser.add_argument("--stimsize", type=float, default=None)
    parser.add_argument("--stimori", type=float, default=None)
    parser.add_argument("--stimfreq", type=float, default=None)
    parser.add_argument("--csdrange", nargs=2, type=float, default=None)
    parser.add_argument("--channels", nargs="*", default=conf['quickcsd']['channels'])
    arg = parser.parse_args()
    
    # set parameters
    sess, rec, blk = arg.data
    h = arg.h 
    R = arg.R 
    sigma = arg.sigma
    channels = arg.channels
    num_ch = len(channels)
    
    # set filenames
    for fn in find_lvdfilenames(arg.rawdir, sess, rec):
        if 'pc{0}'.format(arg.pc) in fn:
            fn_wideband = fn
            break
    else:
        raise IOError("Data file for {sess}_rec{rec}_blk{blk}_pc{pc} not found in {dir}".format(sess=sess, rec=rec, blk=blk, pc=arg.pc, dir=arg.rawdir))
    fn_taskinfo = "{dir}/{sess}_rec{rec}_blk{blk}_taskinfo.mat".format(dir=arg.prepdir, sess=sess, rec=rec, blk=blk)

    # extract stimulus presentation timings and stimulus IDs
    taskinfo = spio.loadmat(fn_taskinfo, struct_as_record=False, squeeze_me=True)
    infoL = taskinfo['L']
    infoS = taskinfo['S']
    success_trials = np.where(infoL.SF == 1)
    idx_stim_on = infoL.FIX_image_on_tmg[success_trials]
    imgIDs = infoS.imgID[infoL.t_tgt_data[success_trials]-1]

    # load parameters from the data file
    lvd_reader = lvdread.LVDReader(fn_wideband)
    header = lvd_reader.get_header()
    Fs = header['AISampleRate']
    idx_ini = int(arg.timerange[0] * Fs)
    idx_fin = int(arg.timerange[1] * Fs)
    times = np.linspace(arg.timerange[0], arg.timerange[1], idx_fin - idx_ini, endpoint=False)

    # compute event triggered average LFP
    ERP = np.zeros((num_ch, idx_fin - idx_ini))
    MUA = np.zeros((num_ch, idx_fin - idx_ini))
    cnt = 0
    for i_trial, indice_on in enumerate(idx_stim_on):
        for i_stim, idx_on in enumerate(indice_on):
            if np.isnan(idx_on):
                continue
            idx_on = long(idx_on)
            imgID = imgIDs[i_trial][i_stim]
            if (arg.stimsize is not None and size[imgID-1] != arg.stimsize)\
             or (arg.stimori is not None and ori[imgID-1] != arg.stimori)\
             or (arg.stimfreq is not None and freq[imgID-1] != arg.stimfreq) :
                continue
            data = lvd_reader.get_data(channel=channels, samplerange=[idx_on + idx_ini, idx_on + idx_fin])
            ERP += butterworth_filter(data, Fs, None, LFP_upfreq)
            MUA += butterworth_filter(np.square(butterworth_filter(data, Fs, MUA_lowfreq, None)), Fs, None, MUA_lowfreq/2)
            cnt += 1
    ERP = ERP / cnt
    MUA = MUA / cnt
    
    ### compute CSD
    CSD = estimateCSD(ERP, h, R, sigma)
    if arg.smooth:
        CSD = np.dot(smoothing_matrix_3p(num_ch), CSD)
    if arg.csdrange is None:
        csdmax = np.abs(CSD).max()
        csdrange = (-csdmax, csdmax)
    else:
        csdrange = arg.csdrange

    # plot results
    plt.subplot(141)
    scale = ERP.std() * 2
    for i_ch, chdata in enumerate(data):
        plt.plot(times, ERP[i_ch] / scale - i_ch, color='black', alpha=0.5)
    plt.xlim(arg.timerange)
    plt.xlabel("Time from stimulus onset (s)")
    plt.ylim(-num_ch, 1)
    plt.ylabel("Channel ID")
    plt.grid()
    plt.title("Local field potential (< {0:.1f} Hz)".format(LFP_upfreq))

    plt.subplot(142)
    baseline = MUA.mean()
    scale = MUA.std() * 4
    for i_ch, chdata in enumerate(data):
        plt.plot(times, (MUA[i_ch] - baseline) / scale - i_ch, color='black', alpha=0.5)
    plt.xlim(arg.timerange)
    plt.xlabel("Time from stimulus onset (s)")
    plt.ylim(-num_ch, 1)
    plt.ylabel("Channel ID")
    plt.grid()
    plt.title("Multi unit activity (> {0:.1f} Hz power)".format(MUA_lowfreq))

    plt.subplot(143)
    X, Y = np.meshgrid(times, range(num_ch+1))
    CSDplot = plt.pcolormesh(X, Y, CSD, vmin=csdrange[0], vmax=csdrange[1], cmap='jet_r')
    plt.xlim(arg.timerange)
    plt.xlabel("Time from stimulus onset (s)")
    plt.ylim(num_ch, 0)
    plt.ylabel("Channel ID")
    plt.grid()
    plt.title("Current source density")

    plt.subplot(1, 40, 31)
    cbar = plt.colorbar(CSDplot, cax=plt.gca())
    cbar.ax.set_ylabel("Current source density (uA/mm3)")

    title = "{sess}_rec{rec}_blk{blk}".format(sess=sess, rec=rec, blk=blk)
    if arg.stimsize > 0:
        title = ', '.join([title, "size: {0} deg".format(arg.stimsize)])
    if arg.stimori > 0:
        title = ', '.join([title, "ori: {0} deg".format(arg.stimori)])
    if arg.stimfreq > 0:
        title = ', '.join([title, "freq: {0} deg".format(arg.stimfreq)])
    title = ' '.join([title, "(N = {0})".format(cnt)])
    plt.suptitle(title)
    plt.show()