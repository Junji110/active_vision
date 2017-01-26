import os
import numpy as np

import eyecalib2


def eyecoil2eyepos(eyecoil, calib_coeffs, calib_order=2, verbose=False):
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

    if callable(calib_coeffs[0]) and callable(calib_coeffs[1]):
        volt2deg_x, volt2deg_y = calib_coeffs
    else:
        calib_coeffs_arr = np.array(calib_coeffs).reshape((2,6)).T
        volt2deg_x = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 0])
        volt2deg_y = bivariate_polynomial(calib_order, calib_coeffs_arr[:, 1])

    # transformation from eye coil signal to eye position
    if verbose:
        print "Eye coil signal transformation..."
    eyepos = np.array([volt2deg_x(eyecoil[:, 0], eyecoil[:, 1]), volt2deg_y(eyecoil[:, 0], eyecoil[:, 1])])
    if verbose:
        print "\t...done."

    return eyepos

def eyepos_derivation(eyepos, derivorder, window_length, polyorder, fs, verbose=False):
    def derivation_savgol(data, derivorder, window_length, polyorder, fs=1.0):
        from scipy.signal import savgol_filter
        delta = 1.0 / fs
        derivs = []
        for dord in derivorder:
            derivs.append(savgol_filter(data, window_length, polyorder, dord, delta))
        return derivs

    if verbose:
        print "Eye velocity and eye acceleration derivation..."
    eyepos, eyevelo, eyeaccl = derivation_savgol(eyepos, derivorder, window_length, polyorder, fs)
    eyevelo = np.hypot(*eyevelo)
    eyeaccl = np.hypot(*eyeaccl)
    if verbose:
        print "\t...done."

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
        # dev = np.sqrt((eyepos_fix[0] - center[0]) ** 2 + (eyepos_fix[1] - center[1]) ** 2)
        dev = np.sqrt((eyepos_fix[0] - x_on) ** 2 + (eyepos_fix[1] - y_on) ** 2)
        # dispersion = np.mean(dev)
        if dev.max() >= eyeshift_threshold:
            continue
            
        # fix.append((idx_onset, idx_offset, center[0], center[1], dispersion))
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

def save_eye_events(fn, sac, fix, samplerange, param):
    # format eye event data
    eye_event = np.append(sac, fix)
    eventID = np.array([100] * len(sac) + [200] * len(fix))
    idx_sort = eye_event['on'].argsort()
    eye_event = eye_event[idx_sort]
    eventID = eventID[idx_sort]

    offset = samplerange[0]
    eye_event['on'] += offset
    eye_event['off'] += offset

    # save to file
    with open(fn, 'w') as fd:
        fd.write("eventID\t{0}\n".format("\t".join(eye_event.dtype.names)))
        for key, val in param.items():
            fd.write("# {0}: {1}\n".format(key, val))
        for evid, ev in zip(eventID, eye_event):
            output = [str(evid)] + [str(x) for x in ev]
            fd.write('\t'.join(output) + '\n')

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
    n_seg = len(seg_range)

    # extract eye events from each segment
    sacfix_buff = []
    for i_seg in range(n_seg):
        if verbose and n_seg > 1:
            print "Processing {0} of {1} segments...".format(i_seg+1, n_seg)

        idx_ini, idx_fin = seg_range[i_seg]
        eyecoil_seg = eyecoil[idx_ini:idx_fin]
    
        # compute eye position and its derivatives from eye coil signal
        # eyepos = eyecoil2eyepos(eyecoil_seg, Fs, calib_coeffs, calib_order=calib_order, verbose=verbose)
        eyepos = eyecoil2eyepos(eyecoil_seg, calib_coeffs, calib_order=calib_order, verbose=verbose)
        eyepos, eyevelo, eyeaccl = eyepos_derivation(eyepos, derivorder=(0, 1, 2), window_length=param['savgol_window_length'], polyorder=param['savgol_polyorder'], fs=Fs, verbose=verbose)

        # extract eye events
        sac_tmp, fix_tmp = extract_eye_events(eyepos, eyevelo, eyeaccl, Fs, param, verbose=verbose)
        sac_tmp['on'] += idx_ini; sac_tmp['off'] += idx_ini
        fix_tmp['on'] += idx_ini; fix_tmp['off'] += idx_ini
        sacfix_buff.append([sac_tmp, fix_tmp])

        if verbose and n_seg > 1:
            print "...Segment {0} done.".format(i_seg+1)

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
    from argparse import ArgumentParser
    import json
    import lvdread

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

    def load_task(fn_task, blk):
        convfunc = lambda x: long(x)
        converters = {'INTERVAL': convfunc, 'TIMING_CLOCK': convfunc, 'GL_TIMER_VAL': convfunc}
        taskdata = np.genfromtxt(fn_task, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
        blockdata = taskdata[taskdata['g_block_num'] == blk]

        evID = blockdata['log_task_ctrl']
        evtime = blockdata['TIMING_CLOCK']
        trial = blockdata['TRIAL_NUM']

        num_trials = max(trial)
        success = []
        stimID = []
        for i_trial in range(num_trials):
            trialID = i_trial + 1
            trialdata = blockdata[blockdata['TRIAL_NUM'] == trialID]
            success.append(trialdata[-1]['SF_FLG'])
            stimID.append(trialdata[-1]['t_tgt_data'])

        events = np.array(zip(evID, evtime, trial), dtype=[('evID', int), ('evtime', long), ('trial', int)])
        param = dict(num_trials=num_trials, success=success, stimID=stimID)
        return events, param

    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    eex_param = conf['eyevex']['eex_param']

    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=conf['datadir'])
    parser.add_argument("--savedir", default=conf['savedir'])
    parser.add_argument("--sbj", "--subject")
    parser.add_argument("--sess", "--session")
    parser.add_argument("--rec", "--recording")
    parser.add_argument("--blk", "--block")
    parser.add_argument("--data", nargs=3, default=None)
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

    datadir = "{dir}/{sbj}".format(dir=arg.datadir, sbj=arg.sbj)
    savedir = arg.savedir

    # copy commandline arguments to variables
    if arg.data is None:
        sess, rec, blk = arg.sess, arg.rec, arg.blk
    else:
        sess, rec, blk = arg.data
    blk = int(blk)

    if arg.calib is None:
        calib_sess, calib_rec, calib_blk = arg.calibsess, arg.calibrec, arg.calibblk
    else:
        calib_sess, calib_rec, calib_blk = arg.calib
    calib_blk = int(calib_blk)
    calib_method, calib_param, calib_ignore = arg.calibmeth, arg.calibparam, arg.calibignore

    timerange = arg.timerange

    # identify the name of the eyecoil data file
    fn_eye = [fn for fn in find_filenames(datadir, sess, rec, 'lvd') if 'pc3' in fn]
    if len(fn_eye) == 0:
        raise ValueError("Eye coil data file not found in {0}".format(datadir))
    else:
        fn_eye = fn_eye[0]
    reader_eye = lvdread.LVDReader(fn_eye)
    param = reader_eye.get_param()
    Fs = param['sampling_rate']
    data_length = param['data_length']

    # define sample range
    if blk == 0:
        if timerange is not None:
            samplerange = np.array((timerange[0]*Fs, timerange[1]*Fs), long)
        else:
            samplerange = np.array((0, data_length), long)
    else:
        fn_task = find_filenames(datadir, sess, rec, 'task')[0]
        task_events, task_param = load_task(fn_task, blk)
        samplerange = task_events['evtime'][[0, -2]]
        if timerange is not None:
            samplerange[1] = samplerange[0] + long(timerange[1]*Fs)
            samplerange[0] = samplerange[0] + long(timerange[0]*Fs)

    # define calibration parameter
    print "Generating eye coil signal transform functions..."
    transform = eyecalib2.gen_transform_from_block(calib_method, calib_param, datadir, calib_sess, calib_rec, calib_blk, calib_ignore)
    print "...transform function generated."
    print
    
    # load eyecoil signal
    print "Loading data..."
    eyecoil = reader_eye.get_data(channel=('eyecoil_x', 'eyecoil_y'), samplerange=samplerange)
    eyecoil = eyecoil.swapaxes(0, 1)
    print "...data loaded."
    print

    # extract eye events
    print "Extracting eye events..."
    if arg.plot:
        sac, fix, eyepos, eyevelo, eyeaccl = main(eyecoil, Fs, transform, eex_param, verbose=True, ret_eyepos=True)
    else:
        sac, fix = main(eyecoil, Fs, transform, eex_param, verbose=True)
    print "...eye events extracted."
    print
    
    # save eye events in a file
    fn_eyeevent = "{dir}/{sess}_rec{rec}_blk{blk}_eyeevent.dat".format(dir=savedir, sess=sess, rec=rec, blk=blk)
    params = {
        'subject': arg.sbj,
        'session': sess, 'recording': rec, 'block': blk,
        'calib_sess': calib_sess, 'calib_rec': calib_rec, 'calib_blk': calib_blk,
        'calib_method': calib_method, 'calib_param': calib_param,
        'calib_ignore': calib_ignore,
        'sampling_rate': Fs,
    }
    params.update(eex_param)
    save_eye_events(fn_eyeevent, sac, fix, samplerange, params)
    print 'Eye event data saved in {0}'.format(fn_eyeevent)

    # summary plot
    if arg.plot:
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

        print "Generating summary plot..."
        plot_summary(sac, fix, eyepos, eyevelo, eyeaccl, Fs, eex_param, timerange=timerange)
        print "\t...done."

