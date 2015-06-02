'''
active_vision/prep/eyevex.py

Module for extract eye events from active vision data sets

Written by Junji Ito (j.ito@fz-juelich.de) on 2013.06.18
'''
import os
import time
import numpy as np
import scipy.signal

import eyecalib2


def index_nearest(data, x):
    datalen = len(data)
    i_ini = 0
    i_fin = datalen
    while i_fin - i_ini > 1:
        i_mid = (i_ini + i_fin) / 2
        if data[i_mid] > x:
            i_fin = i_mid
        else:
            i_ini = i_mid
    return i_ini if x - data[i_ini] < data[i_fin] - x else i_fin
            
def bivariate_polynomial(order, coeffs):
    '''
    Return a bivariate polynomial function of given order and coefficients. For
    example, when order=2 and coeffs=[c0, c1, c2, ..., c5] are given, this
    function returns a function polynom(x, y) that evaluates the following
    polynomial:
        
            c0 + (c1 * x) + (c2 * y) + (c3 * x^2) + (c4 * xy) + (c5 * y^2)
    
    Arguments
    ---------
    order: int
        Order of polynomial
    coeffs: 1D float array-like
        An array that contains the coefficients of polynomial. The length of
        coeffs must be equal to or greater than
        
            N_terms = (order + 1) * (order + 2) / 2.
        
        When the length is greater than N_terms, only the first N_terms
        elements are used as coefficients.
        
    Returns
    -------
    polynom: function
        polynom() takes two arguments and returns the value of the specified
        polynomial evaluated with the given arguments. The arguments can be
        integers, floats, or arrays of an identical shape. When arrays are
        given, polynom() returns an array of the same shape as the arguments.
    '''
    # Bivariate polynomial of order N has (N + 1)(N + 2)/2 terms.
    # Length of coeffs must be equal to or greater than this number.
    if len(coeffs) < (order + 1) * (order + 2) / 2:
        raise ValueError("The number of coefficients is not enough for the specified order")
    
    def polynom(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("The shapes of x and y must be identical.")
        retval = np.zeros_like(x)
        n_term = 0
        for ord in range(order + 1):
            for subord in range(ord + 1):
                retval += coeffs[n_term] * x**(ord-subord) * y**subord
                n_term += 1
        return retval
        
    return polynom

def moving_average(data, smooth_len):
    fft = np.fft.fft
    ifft = np.fft.ifft
    N = len(data)
    # the least power of 2 greater than N
    N_pow2 = 2 ** (int(np.log2(N)) + 1)
    
    # linear trend
    trend = np.linspace(data[0], data[-1], len(data))
    
    # zero-padding to a power of 2 for efficient convolution
    tmpdata = np.zeros(N_pow2)              
    tmpdata[0:N] = np.asarray(data) - trend
    
    # generate boxcar kernel
    kernel = np.zeros(N_pow2)
    if smooth_len % 2 == 0:
        kernel[:smooth_len/2] = 1. / smooth_len
    else:
        kernel[:smooth_len/2+1] = 1. / smooth_len
    kernel[-smooth_len/2:] = 1. / smooth_len
    
    # convolution of the signal with the wavelet
    return ifft(fft(tmpdata) * fft(kernel))[0:N].real + trend

def eyecoil2eyepos_savgol(eyecoil, Fs, calib_coeffs, window_length=199, polyorder=2, calib_order=2, verbose=False):
    if callable(calib_coeffs[0]) and callable(calib_coeffs[1]):
        volt2deg_x, volt2deg_y = calib_coeffs
    else:
        calib_coeffs_arr = np.array(calib_coeffs).reshape((2,6)).T
        volt2deg_x = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 0])
        volt2deg_y = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 1])

    # calibrate eye position
    eyepos = np.array([volt2deg_x(eyecoil[:, 0], eyecoil[:, 1]), volt2deg_y(eyecoil[:, 0], eyecoil[:, 1])])
    if verbose: print "Preprocessing 1/2: volt --> deg transformation done."

    eyepos = scipy.signal.savgol_filter(eyepos, window_length, polyorder, deriv=0)
    eyevelo = scipy.signal.savgol_filter(eyepos, window_length, polyorder, deriv=1, delta=1.0/Fs)
    eyevelo_abs = np.hypot(*eyevelo)
    eyeaccl = scipy.signal.savgol_filter(eyepos, window_length, polyorder, deriv=2, delta=1.0/Fs)
    eyeaccl_abs = np.hypot(*eyeaccl)
    if verbose: print "Preprocessing 2/2: smoothing and derivation done."

    return eyepos, eyevelo_abs, eyeaccl_abs

def eyecoil2eyepos(eyecoil, Fs, calib_coeffs, smooth_width, calib_order=2, verbose=False):
    if callable(calib_coeffs[0]) and callable(calib_coeffs[1]):
        volt2deg_x, volt2deg_y = calib_coeffs
    else:
        calib_coeffs_arr = np.array(calib_coeffs).reshape((2,6)).T
        volt2deg_x = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 0])
        volt2deg_y = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 1])
    
    # calibrate eye position
    eyepos = np.array([volt2deg_x(eyecoil[:, 0], eyecoil[:, 1]), volt2deg_y(eyecoil[:, 0], eyecoil[:, 1])])
    if verbose: print "Preprocessing 1/4: eye calibration done."
    
    # smooth eye position signal
    smooth_len = int(smooth_width * Fs)
    eyepos[0] = moving_average(eyepos[0], smooth_len)
    eyepos[1] = moving_average(eyepos[1], smooth_len)
    if verbose: print "Preprocessing 2/4: smoothing done."

    # compute eye velocity
    dx = eyepos[0, 1:] - eyepos[0, :-1]
    dy = eyepos[1, 1:] - eyepos[1, :-1]
    eyevelo = np.sqrt(dx ** 2 + dy ** 2) * Fs
    eyevelo = np.append(eyevelo, eyevelo[-1])
    eyevelo = moving_average(eyevelo, smooth_len)
    if verbose: print "Preprocessing 3/4: eye velocity calculation done."

    # compute eye acceleration
    eyeaccl = (eyevelo[1:] - eyevelo[:-1]) * Fs
    eyeaccl = np.append(eyeaccl, eyeaccl[-1])
    eyeaccl = moving_average(eyeaccl, smooth_len)
    if verbose: print "Preprocessing 4/4: eye acceleration calculation done."
    
    return eyepos, eyevelo, eyeaccl
    
def detect_saccades(eyepos, eyevelo, eyeaccl, Fs, param):
    sacvelo_threshold = param['sacvelo_threshold']
    sacvelo_peak_min = param['sacvelo_peak_min']
    sacvelo_peak_max = param['sacvelo_peak_max']
    sacaccl_peak_min = param['sacaccl_peak_min']
    sacaccl_peak_max = param['sacaccl_peak_max']
    sacamp_threshold = param['sacamp_threshold']
    saclen_min = int(param['sacdur_min'] * Fs)
    saclen_max = int(param['sacdur_max'] * Fs)
    
    eyevelo_digitized = (eyevelo > sacvelo_threshold).astype(int)
    idx_onset = np.where(np.diff(eyevelo_digitized) == 1)[0] + 1
    idx_offset = np.where(np.diff(eyevelo_digitized) == -1)[0] + 1
    
    if idx_offset[0] < idx_onset[0]:
        idx_offset = idx_offset[1:]
    
    sac = []
    for ion, ioff in zip(idx_onset, idx_offset):
        saclen = ioff - ion
        sacvelo_peak = eyevelo[ion:ioff].max()
        sacaccl_peak = eyeaccl[ion:ioff].max()
        x_on = eyepos[0, ion]
        y_on = eyepos[1, ion]
        x_off = eyepos[0, ioff]
        y_off = eyepos[1, ioff]
        sacamp = np.sqrt((x_off - x_on) ** 2 + (y_off - y_on) ** 2)
        if (
            saclen_min <= saclen < saclen_max
            and sacvelo_peak_min < sacvelo_peak < sacvelo_peak_max
            and sacaccl_peak_min < sacaccl_peak < sacaccl_peak_max
            and sacamp > sacamp_threshold
            ):
            sac.append((ion, ioff, x_on, y_on, x_off, y_off, sacvelo_peak, sacaccl_peak))
    
    return np.array(sac, dtype=[('on', long), ('off', long), ('x_on', float), ('y_on', float), ('x_off', float), ('y_off', float), ('param1', float), ('param2', float)])

def detect_fixations(eyepos, eyevelo, Fs, sac, param):
    eyevelo_threshold = param['fixvelo_threshold']
    eyeshift_threshold = param['fixshift_threshold']
    fixlen_min = int(param['fixdur_min'] * Fs)
    fixlen_max = int(param['fixdur_max'] * Fs)
    
    fix = []
    for i in xrange(len(sac) - 1):
        # fixation duration must be within fixdur_range
        idx_onset = sac['off'][i]
        idx_offset = sac['on'][i + 1]
        if not (fixlen_min <= idx_offset - idx_onset < fixlen_max):
            continue
        
        # eye velocity during a fixation must be smaller than eyevelo_threshold
        eyevelo_fix = eyevelo[idx_onset:idx_offset]
        if eyevelo_fix.max() >= eyevelo_threshold:
            continue
        
        # eye position shifts during a fixation must be within
        # eyeshift_threshold measured from the mean eye position during the
        # fixation
        eyepos_fix = eyepos[:, idx_onset:idx_offset]
        x_on = eyepos[0, idx_onset]
        y_on = eyepos[1, idx_onset]
        x_off = eyepos[0, idx_offset]
        y_off = eyepos[1, idx_offset]
        center = np.mean(eyepos_fix, axis=1)
#        dev = np.sqrt((eyepos_fix[0] - center[0]) ** 2 + (eyepos_fix[1] - center[1]) ** 2)
        dev = np.sqrt((eyepos_fix[0] - x_on) ** 2 + (eyepos_fix[1] - y_on) ** 2)
#        dispersion = np.mean(dev)
        if dev.max() >= eyeshift_threshold:
            continue
            
#        fix.append((idx_onset, idx_offset, center[0], center[1], dispersion))
        fix.append((idx_onset, idx_offset, x_on, y_on, x_off, y_off, center[0], center[1]))
        
    return np.array(fix, dtype=[('on', long), ('off', long), ('x_on', float), ('y_on', float), ('x_off', float), ('y_off', float), ('param1', float), ('param2', float)])

def extract_eye_events(eyepos, eyevelo, eyeaccl, Fs, param, verbose=False):
    if verbose:
        print "Saccade extraction..."
    sac = detect_saccades(eyepos, eyevelo, eyeaccl, Fs, param)
    if verbose:
        print "\t...done."
        
    if verbose:
        print "Fixation extraction..."
    fix = detect_fixations(eyepos, eyevelo, Fs, sac, param)
    if verbose:
        print "\t...done."
        
    return sac, fix

def plot_summary(sac, fix, eyepos, eyevelo, eyeaccl, Fs, param, timerange=None):
    import matplotlib.pyplot as plt
    
    figure = plt.figure()
    
    # define axes
    ax1 = figure.add_subplot(411)
    ax2 = figure.add_subplot(412, sharex=ax1)
    ax3 = figure.add_subplot(413, sharex=ax1)
    ax4 = figure.add_subplot(414, sharex=ax1)
    
    # set axes range
    ax1.set_ylim(-40, 40)
    ax2.set_ylim(-40, 40)
    ax3.set_ylim(-100, 1100)
    ax4.set_ylim(-10000, 110000)
    
    # set axes label
    ax1.set_ylabel('X (deg)')
    ax2.set_ylabel('Y (deg)')
    ax3.set_ylabel('Velo (deg/s)')
    ax4.set_ylabel('Accl (deg/s2)')
    ax4.set_xlabel('Time (sec)')
    
    t = np.arange(eyepos.shape[1]) / Fs
    if timerange is not None:
        t += timerange[0]
    
    ax1.plot(t, eyepos[0], lw=1.5, color='black', alpha=0.2)
    ax2.plot(t, eyepos[1], lw=1.5, color='black', alpha=0.2)
    for sac_tmp in sac:
        sac_on = sac_tmp['on']
        sac_off = sac_tmp['off']
        ax1.plot(t[sac_on:sac_off], eyepos[0, sac_on:sac_off], lw=1.5, color='red')
        ax2.plot(t[sac_on:sac_off], eyepos[1, sac_on:sac_off], lw=1.5, color='red')
    for fix_tmp in fix:
        fix_on = fix_tmp['on']
        fix_off = fix_tmp['off']
        ax1.plot(t[fix_on:fix_off], eyepos[0, fix_on:fix_off], lw=1.5, color='green')
        ax2.plot(t[fix_on:fix_off], eyepos[1, fix_on:fix_off], lw=1.5, color='green')
    ax1.grid()
    ax2.grid()
    
    ax3.plot(t, eyevelo, color='black')
    ax3.axhline(y=param['sacvelo_threshold'])
    ax3.axhline(y=param['sacvelo_peak_min'])
    ax3.grid()
    
    ax4.plot(t, eyeaccl, color='black')
    ax4.axhline(y=param['sacaccl_peak_min'])
    ax4.grid()

    ax4.autoscale(axis='x', tight=True)
    
    plt.show()

def main(eyecoil, Fs, calib_coeffs, param, datalen_max=10000000, seg_overlap=100000, calib_order=2, ret_eyepos=False, verbose=False):
    # cut data into segments if it's too long (> datalen_max). Here indices of
    # the head and the tail of each segment are defined.
    datalen = eyecoil.shape[0]
    if datalen < datalen_max:
        seg_range = [[0, datalen]]
    else:
        seg_range = []
        for i in range(0, datalen - datalen_max, datalen_max - seg_overlap):
            seg_range.append([i, i + datalen_max])
        seg_range.append([i + datalen_max - seg_overlap, datalen])
        print "Data is longer than datalen_max (= %d); cut into %d segments." % (datalen_max, len(seg_range),)
            
    # extract eye events from each segment
    sacfix_buff = []
    for idx_ini, idx_fin in seg_range:
        eyecoil_seg = eyecoil[idx_ini:idx_fin]
    
        # compute eye position and its deliverables from eye coil signal
        # eyepos, eyevelo, eyeaccl = eyecoil2eyepos(eyecoil_seg, Fs, calib_coeffs, param['smooth_width'], calib_order=calib_order, verbose=verbose)
        eyepos, eyevelo, eyeaccl = eyecoil2eyepos_savgol(eyecoil_seg, Fs, calib_coeffs, calib_order=calib_order, verbose=verbose)

        # extract eye events
        sac_tmp, fix_tmp = extract_eye_events(eyepos, eyevelo, eyeaccl, Fs, param, verbose=verbose)
        sac_tmp['on'] += idx_ini; sac_tmp['off'] += idx_ini
        fix_tmp['on'] += idx_ini; fix_tmp['off'] += idx_ini
        sacfix_buff.append([sac_tmp, fix_tmp])
    
    # concatenate the eye events from segments
    sacfix = [sacfix_buff[0][0], sacfix_buff[0][1]]
    for sacfix_tmp in sacfix_buff[1:]:
        for i in [0, 1]:
            t_last = sacfix[i]['on'][-1]
            if t_last in sacfix_tmp[i]['on']:
                idx_ini = np.where(sacfix_tmp[i]['on'] == t_last)[0][0] + 1
            else:
                idx_ini = 0
            sacfix[i] = np.append(sacfix[i], sacfix_tmp[i][idx_ini:])
    sac = sacfix[0]
    fix = sacfix[1]
    
    if ret_eyepos is True:
        return sac, fix, eyepos, eyevelo, eyeaccl
    else:
        return sac, fix
    
def find_filenames(datadir, session, rec, filetype):
    import re

    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml', 'hdf5']:
        raise ValueError("Filetype {0} is not supported.".format(filetype))
    
    if filetype in ['daq', 'lvd', 'hdf5', 'odml']:
        searchdir = "{dir}/{sess}".format(dir=datadir, sess=session)
        re_filename = re.compile('{sess}.*_rec{rec}.*\.{filetype}$'.format(sess=session, rec=rec, filetype=filetype))
    else:
        searchdir = "{dir}/{sess}/{sess}_rec{rec}".format(dir=datadir, sess=session, rec=rec)
        re_filename = re.compile(".*{0}.*".format(filetype))
        
    filenames = os.listdir(searchdir)
    fn_found = []
    for fn in filenames:
        match = re_filename.match(fn)
        if match:
            fn_found.append("{0}/{1}".format(searchdir, fn))

    if len(fn_found) == 0:
        raise IOError("Files of type '{0}' not found.".format(filetype))
    else:
        return fn_found

if __name__ == '__main__':
    import json
    from argparse import ArgumentParser
    import lvdread
    import matplotlib.pyplot as plt

    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    eex_param = conf['eyevex']['eex_param']

    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=conf['datadir'])
    parser.add_argument("--sess", "--session")
    parser.add_argument("--rec")
    parser.add_argument("--data", nargs=2, default=None)
    parser.add_argument("--calib_sess", dest="calibsess")
    parser.add_argument("--calib_rec", dest="calibrec")
    parser.add_argument("--calib_blk", dest="calibblk")
    parser.add_argument("--calib", nargs=3, default=None)
    parser.add_argument("--calib_method", dest="calibmeth", default=conf['eyevex']['calib_method'])
    parser.add_argument("--calib_param", dest="calibparam", default=conf['eyevex']['calib_param'])
    parser.add_argument("--calib_ignore", dest="calibignore", nargs='*', type=int, default=[-1,])
    parser.add_argument("--timerange", nargs=2, type=float, default=None)
    parser.add_argument("--plot", action="store_true", default=False)

    arg = parser.parse_args()

    # copy commandline arguments to variables
    datadir = arg.datadir

    if arg.data is None:
        sess = arg.sess
        rec = arg.rec
    else:
        sess, rec = arg.data

    if arg.calib is None:
        calib_sess = arg.calibsess
        calib_rec = arg.calibrec
        calib_blk = arg.calibblk
    else:
        calib_sess, calib_rec, calib_blk = arg.calib
    calib_blk = int(calib_blk)
    calib_method = arg.calibmeth
    calib_param = arg.calibparam
    calib_ignore = arg.calibignore

    timerange = arg.timerange

    # identify the name of the eyecoil data file
    for fn in find_filenames(datadir, sess, rec, 'lvd'):
        if 'pc3' in fn:
            fn_eye = fn
            break
    else:
        raise ValueError("Eye coil data file not found in {}".format(datadir))
    reader_eye = lvdread.LVDReader(fn_eye)
    param = reader_eye.get_param()
    Fs = param['sampling_rate']

    # define calibration parameter
    print "Generating eye coil signal transform functions..."
    transform = eyecalib2.gen_transform_from_block(calib_method, calib_param, datadir, calib_sess, calib_rec, calib_blk, calib_ignore)
    print "...transform function generated."
    print
    
    # load eyecoil signal
    print "Loading data..."
    eyecoil = reader_eye.get_data(channel=('eyecoil_x', 'eyecoil_y'), timerange=timerange)
    eyecoil = eyecoil.swapaxes(0, 1)
    print "...data loaded."
    print
    
    # extract eye events
    print "Extracting eye events..."
    sac, fix = main(eyecoil, Fs, transform, eex_param, verbose=True)
    print "...eye event extraction done."
    print
    
    # format eye event data
    eye_event = np.append(sac, fix)
    offset = 0 if timerange is None else timerange[0]*Fs
    eye_event['on'] += offset
    eye_event['off'] += offset
    idx_sort = eye_event['on'].argsort()
    eye_event = eye_event[idx_sort]
    eventID = np.array([100] * len(sac) + [200] * len(fix))
    eventID = eventID[idx_sort]

    # save to file
    fn_eyeevent = "{sess}_rec{rec}_eyeevent.dat".format(sess=sess, rec=rec)
    with open(fn_eyeevent, 'w') as fd:
        fd.write("eventID\t" + "\t".join(eye_event.dtype.names) + "\n")
        for evid, ev in zip(eventID, eye_event):
            output = [str(evid)] + [str(x) for x in ev]
            fd.write('\t'.join(output) + '\n')
    print 'Eye event data saved in {0}'.format(fn_eyeevent)

    # summary plot
    if arg.plot:
        print "Generating summary plot..."
        # eyepos, eyevelo, eyeaccl = eyecoil2eyepos(eyecoil, Fs, transform, eex_param['smooth_width'])
        eyepos, eyevelo, eyeaccl = eyecoil2eyepos_savgol(eyecoil, Fs, transform)

        sac_amp = np.sqrt((sac['x_off'] - sac['x_on']) ** 2 + (sac['y_off'] - sac['y_on']) ** 2)
        sac_velo = sac['param1']
        sac_angle = np.arctan2(sac['y_off'] - sac['y_on'], sac['x_off'] - sac['x_on'])

        A = (1000.0-30.0)/9.0
        B = 30.0
        idx_mainseq = sac_velo < A * sac_amp + B

        idx_pat = np.ones_like(sac, bool)
        for i_sac, _ in enumerate(sac[1:]):
            if (sac[i_sac]['on'] - sac[i_sac-1]['on'] < 1000)\
                    and (0.9 < sac_velo[i_sac] / sac_velo[i_sac-1] < 1.1)\
                    and (np.abs(np.exp(1j*sac_angle[i_sac]) + np.exp(1j*sac_angle[i_sac-1])) / 2 < 0.1):
                idx_pat[i_sac] = idx_pat[i_sac-1] = False

        # sac = sac[np.logical_not(idx_mainseq)]
        sac = sac[np.logical_not(idx_pat)]
        print "# sac: {}".format(len(sac))

        # plt.subplot(211)
        # plt.plot(eyepos[0])
        # plt.plot(eyepos[1])
        # plt.grid()
        # plt.subplot(212, sharex=plt.gca())
        # plt.plot(eyecoil[:, 0])
        # plt.plot(eyecoil[:, 1])
        # plt.grid()
        # plt.show()

        plot_summary(sac, fix, eyepos, eyevelo, eyeaccl, Fs, eex_param, timerange=timerange)
        print "...done."

