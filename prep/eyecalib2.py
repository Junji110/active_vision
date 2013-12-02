'''
active_vision/prep/eyecalib2.py

Module for calibration of eye tracking data for LVD data format

Written by Junji Ito (j.ito@fz-juelich.de) on 2013.09.26
'''
import os
import json

import numpy as np
from scipy import linalg, interpolate
from scipy.io import loadmat
import odml
from odml.tools.xmlparser import XMLWriter, XMLReader

#from active_vision.fileio import daqread
import daqread
import lvdread

def polynomial(order, coeffs):
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
    order = int(order)
    
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
    
def polynomial_fit(ideal, actual, order=2):
    order = int(order)
    A = []
    for obs in actual:
        terms = []
        for ord in range(order + 1):
            for yord in range(ord + 1):
                terms.append((obs[0] ** (ord - yord)) * (obs[1] ** yord))
        A.append(terms)
    return linalg.lstsq(A, ideal)
            
class TransformFactory(object):
    '''
    A class for generating functions that transform eye tracker signal into
    horizontal and vertical gaze position on the display.
    
    Arguments
    ---------
    method: str
        Specifies the method for the transform generation. The following methods
        are implemented:
        
            'polynomial_fit':
                Least square fit with bivariate polynomial. The order of the
                polynomial must be specified via the argument param.
                
            'multiquadric':
            'inverse':
            'gaussian':
                Interpolation with radial basis function of different types.
                The scale of the basis function must be specified via the
                argument param.
            'linear':
            'cubic':
            'quintic':
            'thin_plate':
                Interpolation with radial basis function of different types.
                No parameter needs to be specified for these basis
                functions.
    
    param: int or float
        Parameters required for the specified method.
    
    Methods
    -------
    gen_transform(signal, gazepos):
        Return two functions that transform eye tracker signal into horizontal
        and vertical gaze position on the display, based on calibration
        recordings.
    '''
    def __init__(self, method, param=None):
        self.method = method
        self.param = param
        self.coeffs = None
        self.residues = None
        self.rank = None
        self.singular = None
    
    def get_param(self):
        return self.param
    
    def get_coeffs(self):
        return self.coeffs
    
    def get_residues(self):
        return self.residues
    
    def get_rank(self):
        return self.rank
    
    def get_singular(self):
        return self.singular
        
    def gen_transform(self, signal, gazepos):
        '''
        Return two functions that transform eye tracker signal into horizontal
        and vertical gaze position on the display, based on calibration
        recordings.
        
        Arguments
        ---------
        signal: float array of shape (N, 2)
            Eye tracker signals at N calibration points. The signals for
            horizontal and vertical directions should be stored in the first and
            the second elements of the second dimension, respectively.
        
        gazepos: float array of shape (N, 2)
            Positions of the calibration points used for each element of signal.
            Horizontal and vertical positions should be stored in the first and
            the second elements of the second dimension, respectively.
        
        Returns
        -------
        transform_h, transform_v: functions
            Functions that transform eye tracker signal into gaze position on
            the display. Each function takes two arguments, representing eye
            tracker signal from two channels, and returns the horizontal or
            vertical gaze position corresponding to the given eye tracker
            signal. The arguments can be single numbers or arrays of an
            identical shape. When arrays are given, the function returns an
            array of the same shape as the arguments.
        '''
        if self.method == 'polynomial_fit':
            if not (isinstance(self.param, int) or isinstance(self.param, float)):
                raise ValueError("Specify the order of the polynomial via keyword argument 'param'")
            order = int(self.param)
            
            self.coeffs, self.residues, self.rank, self.singular = polynomial_fit(gazepos, signal, order)
            transform_h = polynomial(order, self.coeffs[:, 0])
            transform_v = polynomial(order, self.coeffs[:, 1])
        elif self.method in ['multiquadric', 'inverse', 'gaussian']:
            if not (isinstance(self.param, float) or isinstance(self.param, int)):
                raise ValueError("Specify the parameter of the radial basis function via keyword argument 'param'")
            transform_h = interpolate.Rbf(signal[:, 0], signal[:, 1], gazepos[:, 0], function=self.method, epsilon=self.param)
            transform_v = interpolate.Rbf(signal[:, 0], signal[:, 1], gazepos[:, 1], function=self.method, epsilon=self.param)
        elif self.method in ['linear', 'cubic', 'quintic', 'thin_plate']:
            transform_h = interpolate.Rbf(signal[:, 0], signal[:, 1], gazepos[:, 0], function=self.method)
            transform_v = interpolate.Rbf(signal[:, 0], signal[:, 1], gazepos[:, 1], function=self.method)
            
        return transform_h, transform_v

def find_filenames(datadir, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'data']:
        raise ValueError("filetype must be either of 'imginfo', 'stimtiming', 'param', 'parameter', 'task', or 'daq'.")
    
    # identify the names of metadata files
    if filetype in ['daq', 'lvd', 'data']:
        searchdir = "{0}/{1}".format(datadir, session)
        searchtoken = '_rec{0}_pc'.format(rec)
    else:
        searchdir = "{ddir}/{sess}/{sess}_rec{rec}".format(ddir=datadir, sess=session, rec=rec)
        if not os.path.exists(searchdir):
            searchdir = "{ddir}/{sess}/{sess}_{rec}".format(ddir=datadir, sess=session, rec=rec) # old directory naming scheme
        searchtoken = filetype
        
    filenames = os.listdir(searchdir)
    fn_found = []
    for fn in filenames:
        if searchtoken in fn:
            fn_found.append("{0}/{1}".format(searchdir, fn))
    if len(fn_found) == 0:
        raise IOError("Files of type '{0}' not found.".format(filetype))
    
    return fn_found

def get_eyecalib_trialinfo(filename, blk=-1, fix_off_id=210):
    data = np.genfromtxt(filename, skip_header=1, delimiter=',', names=True, dtype=None)
    
    if 2 not in data['g_task_switch']:
        raise ValueError("No eye calibration trials in the specified recording.")
    
    if blk > 0:
        if 'g_block_num' not in data.dtype.names:
            raise ValueError("The task data file '{0}' does not contain block data.".format(filename))
        data = data[data['g_block_num'] == blk]
    
    num_trial = data['TRIAL_NUM'].max() + 1
    
    fix_tgt, fix_on, fix_off, success = [[] for dummy in range(4)]
    for i_trial in range(1, num_trial):   # trial 0 is skipped because it's not a proper trial
        data_trial = data[data['TRIAL_NUM'] == i_trial]
        fix_tgt.append(data_trial['t_tgt_data'][0])
        fix_on.append(data_trial['TIMING_CLOCK'][0])
        fix_off.append(data_trial['TIMING_CLOCK'][data_trial['log_task_ctrl']==fix_off_id][0])
        success.append(np.all(data_trial['SF_FLG']))
    
    return np.array(fix_tgt), np.array(fix_on), np.array(fix_off), np.array(success)
            
def get_eyecalib_fixpos(filename):
    imginfo = loadmat(filename)
    t_info = imginfo['t_info'][0,0]
    task_type = t_info['task_type'][0,0]
    if task_type != 2:
        raise ValueError("The imginfo file '{0}' is not from an eye calibration session.".format(filename))
    f_posx = t_info['f_posx'].reshape(-1)
    f_posy = t_info['f_posy'].reshape(-1)
    return np.array((f_posx, f_posy)).T

def extract_eyecalib_signal_lvd(filename, blk, fix_off):
    lvd_reader = lvdread.LVDReader(filename)
    header = lvd_reader.get_header()
    
    ch_eyecoil_x = header['AIUsedChannelName'].index('eyecoil_x')
    ch_eyecoil_y = header['AIUsedChannelName'].index('eyecoil_y')
    
    eyecoil = np.empty((len(fix_off), 2))
    for i_trial, idx in enumerate(fix_off):
        # get eye coil data at fixation offset
        eyecoil[i_trial][0] = lvd_reader.get_data(ch_eyecoil_x, samplerange=[idx, idx+1])[0]
        eyecoil[i_trial][1] = lvd_reader.get_data(ch_eyecoil_y, samplerange=[idx, idx+1])[0]
    
    return eyecoil

def extract_eyecalib_signal_daq(filename, fix_off):
    data, time, objinfo = daqread.daqread(filename, 'data-info')
    
    # identify the channels for necessary data
    chnames = [x['ChannelName'] for x in objinfo.ObjInfo['Channel']]
    ch_timing = chnames.index('timing')
    ch_eyecoil_x = chnames.index('eyecoil_x')
    ch_eyecoil_y = chnames.index('eyecoil_y')
    
    timing = data[:, ch_timing]
    eyecoil_x = data[:, ch_eyecoil_x]
    eyecoil_y = data[:, ch_eyecoil_y]
    
    # mapping from timing clock number to sample number
    clk2smpl = np.where(np.diff((timing < 2).astype(int)) == 1)[0] + 1
#    clk2smpl = np.array([i for i in range(data.shape[0] - 1) if timing[i] < 2 and timing[i+1] > 2])
    
    # pick up the eyecoil signals at the offsets of fixation spot
    idx = clk2smpl[fix_off]
    
    return np.array((eyecoil_x[idx], eyecoil_y[idx])).T

def extract_eyecalib_signal(data_type, filename, fix_off, blk=-1):
    if data_type == 'daq':
        return extract_eyecalib_signal_daq(filename, fix_off)
    elif data_type == 'lvd':
        return extract_eyecalib_signal_lvd(filename, blk, fix_off)

def extract_eyecalib_data(datadir, session, rec, blk):   
    # identify the format of the data file from its extension
    fnlist_data = find_filenames(datadir, session, rec, 'data')
    if len(fnlist_data) == 0:
        raise IOError("Data file(s) not found.")
    fnlist_data = filter(lambda x: '_pc3' in x.split('/')[-1], fnlist_data)
    if len(fnlist_data) == 0:
        raise ValueError("No eye coil recording data file was found in {1}.".format(datadir))
    else:
        fn_data = fnlist_data[0]
    data_ext = fn_data.split('.')[-1]
    
    # set datatype specific parameters
    if data_ext == 'daq':
        fix_off_id = 10
        fn_imginfo = find_filenames(datadir, session, rec, 'imginfo')[0]
    elif data_ext == 'lvd':
        fix_off_id = 210
        fnlist_imginfo = find_filenames(datadir, session, rec, 'imginfo')
        fnlist_imginfo = filter(lambda x: '_blk{0}'.format(blk) in x.split('/')[-1], fnlist_imginfo)
        if len(fnlist_imginfo) == 0:
            raise ValueError("Specified block number {0} is not valid for session {1}_rec{2}.".format(blk, session, rec))
        else:
            fn_imginfo = fnlist_imginfo[0]
    
    # read trial information from task file
    fn_task = find_filenames(datadir, session, rec, 'task')[0]
    fix_tgt, fix_on, fix_off, success = get_eyecalib_trialinfo(fn_task, blk, fix_off_id)
    
    # read fixation positions from imginfo file
    fixpos = get_eyecalib_fixpos(fn_imginfo)
    # - take only the success trials
    fix_tgt = fix_tgt[success] # stimulus IDs of success trials only
    fixpos = fixpos[fix_tgt-1] # stimulus ID = n corresponds to the (n-1)-th entry of fixpos
    
    # read eyecoil signal from the data file
    signal = extract_eyecalib_signal(data_ext, fn_data, fix_off, blk)
    # - take only the success trials
    signal = signal[success]
    
    return fixpos, signal
   
def saveparams_odml(datadir, sess, rec, blk, ignore, coeffs):
    fn_odML = "%(datadir)s/%(sess)s/%(sess)s_rec%(rec)s.odml" % {'datadir':datadir, 'sess':sess, 'rec':rec}
    
    # load metadata from odML file
    with open(fn_odML, 'r') as fd:
        metadata = XMLReader().fromFile(fd)
    
    sect = metadata['Dataset']['AnalogData3']
    if sect.find_related(key='CalibParams') is None:
        #sect.append(odml.Section(name="CalibParams", type="parameter"))
        raise ValueError("Section for CalibParam not found in {0}.".format(fn_odML))
    
    # add calibration paramters to the odML structure
    sect_calib = sect['CalibParams']
    props = [
             {'name':"blk{0}_calib_sess".format(blk), 'value':sess, 'unit':"", 'dtype':"string"},
             {'name':"blk{0}_calib_rec".format(blk), 'value':rec, 'unit':"", 'dtype':"string"},
             {'name':"blk{0}_calib_blk".format(blk), 'value':blk, 'unit':"", 'dtype':"int"},
             {'name':"blk{0}_Ignore".format(blk), 'value':ignore, 'unit':"", 'dtype':"int"},
             {'name':"blk{0}_Coeffs".format(blk), 'value':coeffs.T.reshape(12).tolist(), 'unit':"", 'dtype':"float"},
             ]
    for prop in props:
        if sect_calib.contains(odml.Property(prop['name'], None)):
            sect_calib.remove(sect_calib.properties[prop['name']])
        sect_calib.append(odml.Property(**prop))
    
    # save metadata back to odml file
    XMLWriter(metadata).write_file(fn_odML)
    
def plot_summary(gazepos, signal, transform, xrange, yrange, vrange, ignore, method, method_param):
    import matplotlib.pyplot as plt
   
    # define a mask rejecting ignored signal points
    mask_goodsig = np.array([(x not in ignore) for x in range(len(signal))])
    
    plt.figure(1)
    plt.subplot(1, 1, 1, aspect=True)
    for i in range(len(signal)):
        if i in ignore:
            plt.text(signal[i, 0], signal[i, 1], str(i), color='red', alpha=0.5, weight=1000)
        else:
            plt.text(signal[i, 0], signal[i, 1], str(i), color='blue', alpha=0.5, weight=1000)
            
        plt.plot(signal[i, 0], signal[i, 1], 'x')
    plt.xlabel("eyecoil_x (V)")
    plt.ylabel("eyecoil_y (V)")
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.grid()
    plt.suptitle("Eye coil signal at each fixation")
    
    plt.figure(2)
    # interpolate values on a mesh using the obtained transform
    xs = np.linspace(xrange[0], xrange[1], 100)
    ys = np.linspace(yrange[0], yrange[1], 100)
    X, Y = np.meshgrid(xs, ys)
    Z_fit_x = transform[0](X, Y)
    Z_fit_y = transform[1](X, Y)
    
    # plot the result
    plt.subplot(211, aspect=True)
    plt.pcolormesh(X, Y, Z_fit_x, vmin=vrange[0], vmax=vrange[1], cmap='jet')
    plt.scatter(signal[mask_goodsig, 0], signal[mask_goodsig, 1], 30, gazepos[mask_goodsig, 0], vmin=vrange[0], vmax=vrange[1], cmap='jet')
    plt.xlabel("eyecoil_x (V)")
    plt.ylabel("eyecoil_y (V)")
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.colorbar().set_label('Horizontal gaze position (deg)')
    plt.grid()
    
    plt.subplot(212, aspect=True)
    plt.pcolormesh(X, Y, Z_fit_y, vmin=vrange[0], vmax=vrange[1], cmap='jet')
    plt.scatter(signal[mask_goodsig, 0], signal[mask_goodsig, 1], 30, gazepos[mask_goodsig, 1], vmin=vrange[0], vmax=vrange[1], cmap='jet')
    plt.xlabel("eyecoil_x (V)")
    plt.ylabel("eyecoil_y (V)")
    plt.xlim(xrange)
    plt.ylim(yrange)
    plt.colorbar().set_label('Vertical gaze position (deg)')
    plt.grid()
    
    plt.suptitle("Transform obtained by %s (param: %.2f) with %d samples" % (method, method_param, np.sum(mask_goodsig)))
    plt.show()
    
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=conf['datadir'])
    parser.add_argument("--sess", "--session")
    parser.add_argument("--rec")
    parser.add_argument("--blk", "--block", type=int, default=-1)
    parser.add_argument("--method", default=conf['eyecalib']['method'])
    parser.add_argument("--method_param", type=float, default=conf['eyecalib']['method_param'])
    parser.add_argument("--ignore", nargs='*', type=int, default=[-1,])
    parser.add_argument("--nosave", action='store_false', dest="odml", default=True)
    parser.add_argument("--xrange", nargs=2, type=float, default=None)
    parser.add_argument("--yrange", nargs=2, type=float, default=None)
    parser.add_argument("--vrange", nargs=2, type=float, default=None)
    parser.add_argument("--eyepos_x", nargs='*', type=float, default=None)
    parser.add_argument("--eyepos_y", nargs='*', type=float, default=None)
    parser.add_argument("--eyecoil_x", nargs='*', type=float, default=None)
    parser.add_argument("--eyecoil_y", nargs='*', type=float, default=None)
    arg = parser.parse_args()
    
    if None in (arg.eyecoil_x, arg.eyecoil_y, arg.eyepos_x, arg.eyepos_y):
        ideal, actual = extract_eyecalib_data(arg.datadir, arg.sess, arg.rec, arg.blk)
    else:
        ideal = np.asarray(zip(arg.eyepos_x, arg.eyepos_y))
        actual = np.asarray(zip(arg.eyecoil_x, arg.eyecoil_y))
    # create True/False mask to reject the trials specified to be ignored
    mask_ignore = np.array([(x not in arg.ignore) for x in range(len(actual))])
    # get coefficients of fitting polinomial
    coeffs, residuals, rank, s = polynomial_fit(ideal[mask_ignore], actual[mask_ignore], arg.method_param)
    
    # save the parameters and the results of fitting
    if arg.odml is True and None not in [arg.sess, arg.rec]:
        saveparams_odml(arg.datadir, arg.sess, arg.rec, arg.blk, arg.ignore, coeffs)
    
    # output the results
    if None in (arg.xrange, arg.yrange, arg.vrange):
        print coeffs
    else:
        tf_x = polynomial(arg.method_param, coeffs[:, 0])
        tf_y = polynomial(arg.method_param, coeffs[:, 1])
        plot_summary(ideal, actual, (tf_x, tf_y), arg.xrange, arg.yrange, arg.vrange, arg.ignore, arg.method, arg.method_param)
