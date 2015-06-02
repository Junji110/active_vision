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


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    else:
        raise ValueError("Configuration file (conf.json) not found.")
    MUA_lowfreq = conf['quicksel']['MUA_freq_lower_bound']
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--rawdir", default=conf['rawdir'])
    parser.add_argument("--prepdir", default=conf['prepdir'])
    parser.add_argument("--data", nargs=3, default=[20140804, 4, 1])
    parser.add_argument("--resprange", nargs=2, default=conf['quicksel']['resprange'])
    parser.add_argument("--baselinerange", nargs=2, default=conf['quicksel']['baselinerange'])
    parser.add_argument("--smooth", type=bool, default=False)
    parser.add_argument("--stimsize", type=float, default=None)
    parser.add_argument("--stimori", type=float, default=None)
    parser.add_argument("--stimfreq", type=float, default=None)
    parser.add_argument("--csdrange", nargs=2, type=float, default=None)
    parser.add_argument("--channels", nargs="*", default=conf['quicksel']['channels'])
    parser.add_argument("--dump", type=bool, default=False)
    parser.add_argument("--dumpdir", default=conf['quicksel']['dumpdir'])
    arg = parser.parse_args()
    
    # set parameters
    sess, rec, blk = arg.data
    channels = arg.channels
    num_ch = len(channels)
    
    # set filenames
    for fn in find_lvdfilenames(arg.rawdir, sess, rec):
        if 'pc1' in fn:
            fn_wideband = fn
            break
    fn_taskinfo = "{dir}/{sess}_rec{rec}_blk{blk}_taskinfo.mat".format(dir=arg.prepdir, sess=sess, rec=rec, blk=blk)

    # load parameters from the data file
    lvd_reader = lvdread.LVDReader(fn_wideband)
    header = lvd_reader.get_header()
    Fs = header['AISampleRate']
    idx_ini = int(arg.resprange[0] * Fs)
    idx_fin = int(arg.resprange[1] * Fs)
    idx_bl_ini = int(arg.baselinerange[0] * Fs)
    idx_bl_fin = int(arg.baselinerange[1] * Fs)

    # extract stimulus presentation timings and stimulus image IDs
    taskinfo = spio.loadmat(fn_taskinfo, struct_as_record=False, squeeze_me=True)
    infoL = taskinfo['L']
    infoS = taskinfo['S']
    success_trials = np.where(infoL.SF == 1)
    idx_stim_on = infoL.FIX_image_on_tmg[success_trials]
    idx_stim_off = infoL.FIX_image_off_tmg[success_trials]
    imgIDs = infoS.imgID[infoL.t_tgt_data[success_trials]-1]
    num_stim = sum(len(x) for x in idx_stim_on)
    
    imgIDset = []
    for imgID in imgIDs:
        imgIDset.extend(imgID)
    imgIDset = set(imgIDset)
    
    # extract MUA responses channel by channel
    for i_ch, chID in enumerate(channels):
        # compute MUA from wideband signal
        data = lvd_reader.get_data(channel=[chID,])[0]
        MUA = butterworth_filter(np.square(butterworth_filter(data, Fs, MUA_lowfreq, None)), Fs, None, MUA_lowfreq/2)

        # extract MUA responses trial by trial
        responses = np.empty(num_stim, dtype=[('FIX_image_on_tmg', long), ('imgID', int), ('response_mean', float), ('baseline_mean', float), ('baseline_std', float)])
        i_stim = 0
        for i_trial, indice_on in enumerate(idx_stim_on):
            for i_stim_trial, idx_on in enumerate(indice_on):
                baseline_mean = MUA[idx_on + idx_bl_ini : idx_on + idx_bl_fin].mean()
                baseline_std = MUA[idx_on + idx_bl_ini : idx_on + idx_bl_fin].std()
                response_mean = MUA[idx_on + idx_ini : idx_on + idx_fin].mean()
                responses[i_stim] = idx_on, imgIDs[i_trial][i_stim_trial], response_mean, baseline_mean, baseline_std
                i_stim += 1
        
        # dump responses as a CSV file
        if arg.dump:
            fn_out = "{dir}/{sess}_rec{rec}_blk{blk}_{chID}_quicksel.csv".format(dir=arg.dumpdir, sess=sess, rec=rec, blk=blk, chID=chID)
            with open(fn_out, 'w') as fd:
                fd.write(",\t".join(responses.dtype.names) + "\n")
                for op in responses:
                    opstr = [str(x) for x in op]
                    fd.write(',\t'.join(opstr) + '\n')
        
        # plot results
        plt.subplot(6, 4, i_ch+1)
        for imgID in imgIDset:
            output_img = responses[responses['imgID'] == imgID]
            resp = (output_img['response_mean'] - output_img['baseline_mean']) / output_img['baseline_std']
            plt.bar(float(imgID) - 0.5, np.mean(resp), 1.0, color='red', alpha=0.5 + (imgID % 2) * 0.5) 
            plt.errorbar(float(imgID), np.mean(resp), np.std(resp, ddof=1)/np.sqrt(len(resp)-1), color='black')
        plt.grid()
        plt.xlim(min(imgIDset)-1, max(imgIDset)+1)
        plt.ylim(-2, 2)
        plt.title(chID)
        if i_ch / 4 == 5:
            plt.xlabel("Image ID")
        if i_ch % 4 == 0:
            plt.ylabel("MUA responses\n(z-score)")

    plt.show()