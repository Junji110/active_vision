'''
active_vision/prep/eyevex.py

Module for extract eye events from active vision data sets

Written by Junji Ito (j.ito@fz-juelich.de) on 2013.06.18
'''
import os
import time
import numpy as np


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

def eyecoil2eyepos(eyecoil, Fs, calib_coeffs, smooth_width, calib_order=2, verbose=False):
    calib_coeffs_arr = np.array(calib_coeffs).reshape((2,6)).T
    
    # calibrate eye position
    volt2deg_x = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 0])
    volt2deg_y = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 1])
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

def summary_plot(sac, fix, eyepos, eyevelo, eyeaccl, Fs, param):
    import matplotlib.pyplot as plt
    
    figure = plt.figure()
    
    # define axes
    ax1 = figure.add_subplot(411)
    ax2 = figure.add_subplot(412, sharex=ax1)
    ax3 = figure.add_subplot(413, sharex=ax1)
    ax4 = figure.add_subplot(414, sharex=ax1)
    
    # set axes range
    ax1.set_ylim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax3.set_ylim(0, 600)
    ax4.set_ylim(-50000, 50000)
    
    # set axes label
    ax1.set_ylabel('X (deg)')
    ax2.set_ylabel('Y (deg)')
    ax3.set_ylabel('Velo (deg/s)')
    ax4.set_ylabel('Accl (deg/s2)')
    ax4.set_xlabel('Time (sec)')
    
    t = np.arange(eyepos.shape[1]) / Fs
    
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
        eyepos, eyevelo, eyeaccl = eyecoil2eyepos(eyecoil_seg, Fs, calib_coeffs, param['smooth_width'], calib_order=calib_order, verbose=verbose)
        
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
    

if __name__ == '__main__':
    import daqread
    import eyecalib_old
    
    # define parameters
    datadir = "C:/Users/ito/datasets/osaka/behavior"
    datasess = "20130306"
    rec = "1"
#    datasess = "20130306"
#    rec = "2"
#    datasess = "20130315"
#    rec = "1"
#    datafile = "20130315_124319_rec1_pc3.daq"
    calibsess = "20130306"
    calibrec = "1"
    eex_param = {
                'smooth_width': 0.002,
#                'sacamp_threshold': 1.0,
#                'sacvelo_threshold': 50.0,
#                'sacvelo_peak_range': [100.0, 1000.0],    # medium sensitivity
                'sacamp_threshold': 0.1,
                'sacvelo_threshold': 30.0,
                'sacvelo_peak_min': 50.0,
                'sacvelo_peak_max': 1000.0,
                'sacaccl_peak_min': 8000.0,
                'sacaccl_peak_max': 100000.0,
                'sacdur_min': 0.005,
                'sacdur_max': 1.0,
                'fixvelo_threshold': 70.0,
                'fixshift_threshold': 1.0,
                'fixdur_min': 0.1,
                'fixdur_max': 1.0,
                }
    #datalen_max = 2000000    # to be used for test
    datalen_max = 10000000
    
    # identify the name of the eyecoil data file
    sessdir = datadir + "/" + datasess
    filenames = os.listdir(sessdir)
    datafile = None
    for fn in filenames:
        if 'rec%s_pc3' % rec in fn:
            datafile = fn
    if not datafile:
        raise IOError("Eyecoil data file not found.")
    
    # define calibration parameter
    print "Defining calibration parameters..."
    # calib_coeffs = np.array([[4.80495885e+00, 1.09999438e+01], [-4.54751460e+00, 2.17502356e+00], [6.81946928e-01, -7.37863592e+00], [-2.93593904e-02, 8.32437864e-03], [-6.96825031e-02, 8.47741602e-02], [-1.64500961e-02, -2.67035858e-01]])    # to be used for test
    calib_coeffs = eyecalib_old.main(datadir, calibsess, calibrec, "polynomial_fit", 2.0)
    print "...calibration parameters defined."
    print
    
    # load eyecoil signal
    print "Loading data..."
    filename = '/'.join([datadir, datasess, datafile])
    objinfo = daqread.daqread(filename, 'info')
    Fs = objinfo.ObjInfo['SampleRate']
    chname = [x['ChannelName'] for x in objinfo.ObjInfo['Channel']]
    chID = [x['Index'] for x in objinfo.ObjInfo['Channel']]
    ch_eyecoil_x = chname.index('eyecoil_x')
    ch_eyecoil_y = chname.index('eyecoil_y')
    eyecoil, time_eyecoil = daqread.daqread(filename, 'data2', Channels=[chID[ch_eyecoil_x], chID[ch_eyecoil_y]])
    print "...data loading done."
    print
    
    # extract eye events
    print "Extracting eye events..."
    sac, fix = main(eyecoil, Fs, calib_coeffs, eex_param, datalen_max=datalen_max, verbose=True)
    print "...eye event extraction done."
    print
    
    # output result
    event = np.append(sac, fix)
    eventID = np.array([100] * len(sac) + [200] * len(fix))
    idx_sort = event['on'].argsort()
    event = event[idx_sort]
    eventID = eventID[idx_sort]
    clk2smpl = np.arange(0, (event['off'][-1] + 1) * 4, 4)
    
    # convert sample count to clock count
    for key in ['on', 'off']:
        for i,smpl in enumerate(event[key]):
            event[key][i] = index_nearest(clk2smpl, smpl)
    outfile = open("%s_rec%s_eyeevent.dat" % (datasess, rec), 'w')
    outfile.write("eventID\t" + "\t".join(event.dtype.names) + "\n")
    for evid, ev in zip(eventID, event):
        output = [str(evid)] + [str(x) for x in ev]
        outfile.write('\t'.join(output) + '\n')
    outfile.close()

    # summary plot
    print "Generating summary plot..."
    if eyecoil.shape[0] >= datalen_max:
        eyecoil = eyecoil[0:datalen_max]
    eyepos, eyevelo, eyeaccl = eyecoil2eyepos(eyecoil, Fs, calib_coeffs, eex_param['smooth_width'], verbose=True)
    sac_seg = sac[sac['off'] < datalen_max]
    fix_seg = fix[fix['off'] < datalen_max]
    summary_plot(sac_seg, fix_seg, eyepos, eyevelo, eyeaccl, Fs, eex_param)
    print "...summary plot generated."
    