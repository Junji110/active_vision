'''
active_vision/prep/eyevex_gui2.py

Graphical user interface for active_vision/prep/eyevex.py

Written by Junji Ito (j.ito@fz-juelich.de) on 2013.09.26
'''
import os
import json
import re
import copy

import numpy as np
import wx
import odml
from odml.tools.xmlparser import XMLWriter, XMLReader
import matplotlib
matplotlib.use('WXAgg') # This needs to be declared before importing sub-modules of matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, NavigationToolbar2WxAgg

import lvdread
import eyevex
import eyecalib2


def load_odml(datadir, sess, rec, blk):
    fn_odML = "{0}/{1}/{1}_rec{2}.odml".format(datadir, sess, rec)
    with open(fn_odML, 'r') as fd:
        metadata = XMLReader().fromFile(fd)
    
    if metadata['Dataset']['EventData'].properties['blk{0}_task_type'.format(blk)].value.data != 3:
        raise ValueError("No free viewing trials in the specified sess/rec/blk.")
        
    prefix = 'blk{0}_'.format(blk)
    
    # store all relevant metadata parameters in one dictionary
    param = {
             'Fs': metadata['Recording']['HardwareSettings']['DataAcquisition3'].properties['AISampleRate'].value.data,
#             'pxlperdeg': metadata['Setup']['HardwareProperties']['Monitor'].properties['PixelPerDegree'].value.data,
             'pxlperdeg': metadata['Recording']['HardwareSettings']['Monitor'].properties['PixelPerDegree'].value.data,
             'datafile': os.path.basename(metadata['Dataset']['AnalogData3'].properties['File'].value.data),
             'taskfile': os.path.basename(metadata['Dataset']['EventData'].properties['File'].value.data),
             'evID': [x.data for x in metadata['Experiment']['Behavior']['Task3'].properties['EventID'].value],
             'evtype': [x.data for x in metadata['Experiment']['Behavior']['Task3'].properties['EventType'].value],
             
             'stimsetname': metadata['Dataset']['StimulusData'].properties[prefix+'setname'].value.data,
             'stimID2imgID': [x.data for x in metadata['Dataset']['StimulusData'].properties[prefix+'imgID'].value],
             'success': [x.data for x in metadata['Dataset']['EventData'].properties[prefix+'success'].value],
             'stimID': [x.data for x in metadata['Dataset']['EventData'].properties[prefix+'stimID'].value],
             }
    
    # if eye event extraction parameters are found in the odML file, store them
    sect = metadata['Dataset']['AnalogData3']
    if sect.find_related(key='EyevexParams') is not None:
        sect_eex = sect['EyevexParams']
        param_eex = {}
        re_propname = re.compile('blk{0}_(.*)'.format(blk))
        for prop in sect_eex.properties:
            match = re_propname.match(prop.name)
            if match:
                paramkey = match.group(1)
                if isinstance(prop.value, list):
                    param_eex[paramkey] = [x.data for x in prop.value]
                else:
                    param_eex[paramkey] = prop.value.data
        if len(param_eex) > 0:
            param['eex'] = param_eex
    
    return param

def load_calibparam(datadir, sess, rec, blk):
    fn_odML = "{0}/{1}/{1}_rec{2}.odml".format(datadir, sess, rec)
    with open(fn_odML, 'r') as fd:
        metadata = XMLReader().fromFile(fd)
    
    sect = metadata['Dataset']['AnalogData3']
    if sect.find_related(key='CalibParams') is None:
        raise ValueError("Section for CalibParam not found in {0}".format(fn_odML))
    
    param = {}
    
    sect_calib = sect['CalibParams']
    re_propname = re.compile('blk{0}_(.*)'.format(blk))
    for prop in sect_calib.properties:
        match = re_propname.match(prop.name)
        if match:
            paramkey = match.group(1)
            if isinstance(prop.value, list):
                param[paramkey] = [x.data for x in prop.value]
            else:
                param[paramkey] = prop.value.data
    
    if len(param) == 0:
        raise ValueError("No eye calibration parameters for block {0} found in {1}".format(blk, fn_odML))
    else:
        return param
        
def get_events(param):
    convfunc = lambda x: long(x)
    converters = {'INTERVAL': convfunc, 'TIMING_CLOCK': convfunc, 'GL_TIMER_VAL': convfunc}
    fn_task = "{datadir}/{sess}/{sess}_rec{rec}/{taskfile}".format(**param)
    taskdata = np.genfromtxt(fn_task, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
    blockdata = taskdata[taskdata['g_block_num'] == param['blk']]
    
    evID = blockdata['log_task_ctrl']
    evtime = blockdata['TIMING_CLOCK']
    trial = blockdata['TRIAL_NUM']
    return np.array(zip(evID, evtime, trial), dtype=[('evID', int), ('evtime', long), ('trial', int)])

def save_eexparam(fn_odML, param):
    # load metadata from odML file
    with open(fn_odML, 'r') as fd:
        metadata = XMLReader().fromFile(fd)
    
    sect = metadata['Dataset']['AnalogData3']
    if sect.find_related(key='EyevexParams') is None:
        raise ValueError("Section for EyevexParams not found in {0}.".format(fn_odML))
    
    # write eex paramters in
    sect_eex = sect['EyevexParams']
    for key, val in param['eex'].items():
        propname = 'blk{0}_{1}'.format(param['blk'], key)
        propdtype = type(val[0]).__name__ if isinstance(val, list) else type(val).__name__
        if propdtype == 'unicode': propdtype = 'string'
        prop = {'name': propname, 'value': val, 'unit': "", 'dtype': propdtype}
        if sect_eex.contains(odml.Property(propname, None)):
            sect_eex.remove(sect_eex.properties[propname])
        sect_eex.append(odml.Property(**prop))
        
    # save metadata back to odml file
    XMLWriter(metadata).write_file(fn_odML)
    
def save_calibparam(fn_odML, param):
    # load metadata from odML file
    with open(fn_odML, 'r') as fd:
        metadata = XMLReader().fromFile(fd)
    
    sect = metadata['Dataset']['AnalogData3']
    if sect.find_related(key='CalibParams') is None:
        raise ValueError("Section for CalibParam not found in {0}.".format(fn_odML))
    
    # write calibration paramters in
    sect_calib = sect['CalibParams']
    for key, val in param['calib'].items():
        propname = 'blk{0}_{1}'.format(param['blk'], key)
        propdtype = type(val[0]).__name__ if isinstance(val, list) else type(val).__name__
#        if propdtype == 'unicode': propdtype = 'string'
        if propdtype in ('str', 'unicode'): propdtype = 'string'
        prop = {'name': propname, 'value': val, 'unit': "", 'dtype': propdtype}
        if sect_calib.contains(odml.Property(propname, None)):
            sect_calib.remove(sect_calib.properties[propname])
        sect_calib.append(odml.Property(**prop))
        
    # save metadata back to odml file
    XMLWriter(metadata).write_file(fn_odML)
    
def drawgraph(figure, img_stim, sac, fix, eyepos, eyevelo, eyeaccl, Fs, param, t_mark=None):
    figure.clf()
    figure.subplots_adjust(bottom=0.08, top=0.955, left=0.045, right=0.99, wspace=0.15)
    
    # define axes
    ax0 = figure.add_subplot(121, aspect=True)
    ax1 = figure.add_subplot(422)
    ax2 = figure.add_subplot(424, sharex=ax1)
    ax3 = figure.add_subplot(426, sharex=ax1)
    ax4 = figure.add_subplot(428, sharex=ax1)
    
    # set axes range
    ax0.set_xlim(-20, 20)
    ax0.set_ylim(-15, 15)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(-20, 20)
    ax2.set_ylim(-20, 20)
    ax3.set_ylim(0, 600)
    ax4.set_ylim(-50000, 50000)
    
    # set axes label
    ax0.set_xlabel('X (deg)')
    ax0.set_ylabel('Y (deg)')
    ax1.set_ylabel('X (deg)')
    ax2.set_ylabel('Y (deg)')
    ax3.set_ylabel('Velo (deg/s)')
    ax4.set_ylabel('Accl (deg/s2)')
    ax4.set_xlabel('Time (sec)')
    
    width = img_stim.shape[1]
    height = img_stim.shape[0]
    pxlperdeg = param['pxlperdeg']
    ax0.imshow(img_stim, extent=[-width/2/pxlperdeg, width/2/pxlperdeg, -height/2/pxlperdeg, height/2/pxlperdeg])
    ax0.plot(eyepos[0], eyepos[1], lw=3, color='white', alpha=0.4)
#    ax0.plot(eyepos[0] - eyepos[0][0], eyepos[1] - eyepos[1][0], lw=3, color='white', alpha=0.4)
    for fix_tmp in fix:
        fix_on = fix_tmp['on']
        fix_off = fix_tmp['off']
        ax0.plot(eyepos[0, fix_on:fix_off], eyepos[1, fix_on:fix_off], lw=4, color='white')
    ax0.plot(eyepos[0, 0], eyepos[1, 0], 'g.', ms=15)
    ax0.plot(eyepos[0, -1], eyepos[1, -1], 'r.', ms=15)
    
    t = np.arange(eyepos.shape[1]) / Fs
    
    ax1.plot(t, eyepos[0], color='black', lw=1.5, alpha=0.2)
    ax2.plot(t, eyepos[1], color='black', lw=1.5, alpha=0.2)
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
    ax3.axhline(y=param['eex']['fixvelo_threshold'], ls='-', color='green')
    ax3.axhline(y=param['eex']['sacvelo_threshold'], ls='-', color='red')
    ax3.axhline(y=param['eex']['sacvelo_peak_min'], ls='--', color='red')
    ax3.grid()
    
    ax4.plot(t, eyeaccl, color='black')
    ax4.axhline(y=param['eex']['sacaccl_peak_min'], ls='-', color='red')
    ax4.grid()
    
    if t_mark:
        idx_mark = int(t_mark * Fs)
        if idx_mark < eyepos.shape[1]:
            ax0.plot(eyepos[0, idx_mark], eyepos[1, idx_mark], 'bx', mew=4, ms=15)
        ax1.axvline(t_mark, color='blue', linestyle='-')
        ax2.axvline(t_mark, color='blue', linestyle='-')
        ax3.axvline(t_mark, color='blue', linestyle='-')
        ax4.axvline(t_mark, color='blue', linestyle='-')

class PlotPanel(wx.Panel):
    def __init__(self, parent, id):
        self.parent = parent
        
        wx.Panel.__init__(self, self.parent, id)
        
        # define widgets
        self.figure = matplotlib.figure.Figure()
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.toolbar = NavigationToolbar2WxAgg(self.canvas)        
        
        self.canvas.mpl_connect('button_press_event', self.OnClick)
        
        # layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.canvas, 1, wx.EXPAND)
        vbox.Add(self.toolbar, 0, wx.EXPAND)
        self.SetSizer(vbox)
    
    def OnClick(self, event):
        if event.button == 3 and event.inaxes is not None:
            if event.inaxes.get_position().xmax > 0.8:
                self.parent.controlpanel.OnDraw(None, event.xdata)
        
class ControlPanel(wx.Panel):
    def __init__(self, parent, id, events, lvd_reader, param):
        self.parent = parent
        self.task_events = events
        self.lvd_reader = lvd_reader
        self.param = copy.deepcopy(param)
        self.param_default = copy.deepcopy(param)
        self.Fs = param['Fs']
        self.calib_coeffs = param['calib']['Coeffs']
        
        wx.Panel.__init__(self, parent, id)
        
        # define widgets
        self.tc = {}
        for key, val in self.param['eex'].items():
            self.tc[key] = wx.TextCtrl(self, -1, str(val))

        self.sc_trial = wx.SpinCtrl(self, -1, str(1), min=1, max=self.task_events['trial'].max())
        
        self.button = {}
        self.button['draw'] = wx.Button(self, 1, 'Draw')
        self.button['prev'] = wx.Button(self, 2, '< Prev')
        self.button['next'] = wx.Button(self, 3, 'Next >')
        self.button['reset'] = wx.Button(self, 4, 'Reset to Default')
        self.button['apply'] = wx.Button(self, 5, 'Save Eye Events')
        self.button['save'] = wx.Button(self, 6, 'Save Parameters')
        
        self.Bind(wx.EVT_BUTTON, self.OnDraw, id=1)
        self.Bind(wx.EVT_BUTTON, self.OnPrev, id=2)
        self.Bind(wx.EVT_BUTTON, self.OnNext, id=3)
        self.Bind(wx.EVT_BUTTON, self.OnReset, id=4)
        self.Bind(wx.EVT_BUTTON, self.OnApply, id=5)
        self.Bind(wx.EVT_BUTTON, self.OnSave, id=6)
        
        # layout
        self.Layout()
        
    def Layout(self):
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        
        # left half
        vbox1 = wx.BoxSizer(wx.VERTICAL)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Trial:'))
        hbox_tmp.Add(self.sc_trial)
        hbox_tmp.Add(self.button['draw'])
        hbox_tmp.Add(self.button['prev'])
        hbox_tmp.Add(self.button['next'])
        vbox1.Add(hbox_tmp)
        
        vbox1.AddSpacer(20)
        
        # saccade parameters
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Saccade velocity threthold:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacvelo_threshold'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'deg/sec'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox1.Add(hbox_tmp)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Saccade velocity peak range:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacvelo_peak_min'])
        hbox_tmp.Add(wx.StaticText(self, -1, ' - '), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacvelo_peak_max'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'deg/sec'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox1.Add(hbox_tmp)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Saccade acceleration peak range:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacaccl_peak_min'])
        hbox_tmp.Add(wx.StaticText(self, -1, ' - '), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacaccl_peak_max'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'deg/sec2'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox1.Add(hbox_tmp)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Saccade duration range:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacdur_min'])
        hbox_tmp.Add(wx.StaticText(self, -1, ' - '), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacdur_max'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'sec'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox1.Add(hbox_tmp)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Saccade amplitude threthold:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['sacamp_threshold'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'deg'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox1.Add(hbox_tmp)
        
        # right half
        vbox2 = wx.BoxSizer(wx.VERTICAL)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(self.button['reset'])
        hbox_tmp.AddSpacer(20)
        hbox_tmp.Add(self.button['save'])
        hbox_tmp.Add(self.button['apply'])
        vbox2.Add(hbox_tmp)
        
        vbox2.AddSpacer(20)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Data smoothing width:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['smooth_width'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'sec'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox2.Add(hbox_tmp)
        
        vbox2.AddSpacer(20)
        
        # fixation parameters
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Fixation velocity threthold:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['fixvelo_threshold'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'deg/sec'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox2.Add(hbox_tmp)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Fixation shift threthold:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['fixshift_threshold'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'deg'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox2.Add(hbox_tmp)
        
        hbox_tmp = wx.BoxSizer(wx.HORIZONTAL)
        hbox_tmp.Add(wx.StaticText(self, -1, 'Fixation duration range:'), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['fixdur_min'])
        hbox_tmp.Add(wx.StaticText(self, -1, ' - '), flag=wx.ALIGN_CENTER_VERTICAL)
        hbox_tmp.Add(self.tc['fixdur_max'])
        hbox_tmp.Add(wx.StaticText(self, -1, 'sec'), flag=wx.ALIGN_CENTER_VERTICAL)
        vbox2.Add(hbox_tmp)
        
        hbox.Add(vbox1, 0, wx.ALL, 10)
        hbox.Add(vbox2, 0, wx.ALL, 10)
        self.SetSizer(hbox)
        
    def OnDraw(self, event, t_mark=None):
        n_trial = int(self.sc_trial.GetValue())
        figure = self.parent.plotpanel.figure
        evID_on = self.param['on_event']
        evID_off = self.param['off_event']
        
        # check if the trial is a success one
        if self.param['success'][n_trial - 1] not in [1, -303]:
            self.parent.statusbar.SetStatusText('No free viewing period in Trial {0}'.format(n_trial))
            figure.clf()
            figure.canvas.draw()
            return
        
        # check if the trial onset and offset are properly defined
        task_events = self.task_events[self.task_events['trial'] == n_trial]
        evIDs = task_events['evID']
        evtimes = task_events['evtime']
        if evID_on not in evIDs or evID_off not in evIDs:
            self.parent.statusbar.SetStatusText('Trial onset or offset is missing in Trial {0}'.format(n_trial))
            figure.clf()
            figure.canvas.draw()
            return
        
        # check if there is a free viewing period of finite duration
        img_on = evtimes[evIDs == evID_on][0]
        img_off = evtimes[evIDs == evID_off][0]
        if img_on == img_off:
            self.parent.statusbar.SetStatusText('The duration of free viewing period is zero in Trial {0}'.format(n_trial))
            figure.clf()
            figure.canvas.draw()
            return
        
        # set the parameter values in the control panel to the param dict
        for key in self.tc.keys():
            self.param['eex'][key] = float(self.tc[key].GetValue())
        
        # read the trial segments of X and Y eye coil signals
        eyecoil = self.lvd_reader.get_data(['eyecoil_x', 'eyecoil_y'], [img_on, img_off])
        eyecoil = eyecoil.T
        
        # extract eye events
        if isinstance(self.calib_coeffs, (str, unicode)):
            transform = eyecalib2.gen_transform_from_block(self.calib_coeffs, self.param['datadir'], self.param['calib']['sess'], self.param['calib']['rec'], self.param['calib']['blk'])
            sac, fix, eyepos, eyevelo, eyeaccl = eyevex.main(eyecoil, self.Fs, transform, self.param['eex'], ret_eyepos=True)
        else:
            sac, fix, eyepos, eyevelo, eyeaccl = eyevex.main(eyecoil, self.Fs, self.calib_coeffs, self.param['eex'], ret_eyepos=True)
        
        # call plot function
        imgID = self.param['stimID2imgID'][self.param['stimID'][n_trial - 1] - 1]
        fn_stim = "{0}/{1}/{2}.png".format(self.param['stimdir'], self.param['stimsetname'], imgID)
        img_stim = matplotlib.image.imread(fn_stim) * 0.7   # dimmed by 0.7 times the original luminance
        
        self.parent.statusbar.SetStatusText('Draw Trial {0}'.format(n_trial))
        drawgraph(figure, img_stim, sac, fix, eyepos, eyevelo, eyeaccl, self.Fs, self.param, t_mark)
        figure.canvas.draw()

    def OnPrev(self, event):
        n_trial = int(self.sc_trial.GetValue())
        self.sc_trial.SetValue(n_trial - 1)
        self.OnDraw(None)
        
    def OnNext(self, event):
        n_trial = int(self.sc_trial.GetValue())
        self.sc_trial.SetValue(n_trial + 1)
        self.OnDraw(None)
    
    def OnReset(self, event):
        self.param = copy.deepcopy(self.param_default)
        for key, val in self.param['eex'].items():
            self.tc[key].SetValue(str(val))
        self.parent.statusbar.SetStatusText('Parameter values reset to default')

    def OnApply(self, event):
        # save parameter values in odML file
        self.OnSave(None)
        
        # extract eye events
        self.parent.statusbar.SetStatusText('Applying the current parameters to the whole recording...')
        task_fin = self.task_events['trial'].max()
        idx_ini = self.task_events[self.task_events['trial'] == 1]['evtime'][0]
        idx_fin = self.task_events[self.task_events['trial'] == task_fin]['evtime'][-1]
        eyecoil = self.lvd_reader.get_data(['eyecoil_x', 'eyecoil_y'], [idx_ini, idx_fin])
        eyecoil = eyecoil.T
        sac, fix = eyevex.main(eyecoil, self.Fs, self.calib_coeffs, self.param['eex'], verbose=True)
        
        # format data
        sac['on'] += idx_ini; sac['off'] += idx_ini
        fix['on'] += idx_ini; fix['off'] += idx_ini
        eye_event = np.append(sac, fix)
        eventID = np.array([100] * len(sac) + [200] * len(fix))
        idx_sort = eye_event['on'].argsort()
        eye_event = eye_event[idx_sort]
        eventID = eventID[idx_sort]
        
        # save to file
        fn_eyeevent = "{sess}_rec{rec}_blk{blk}_eyeevent.dat".format(**self.param)
        with open(fn_eyeevent, 'w') as fd:
            fd.write("eventID\t" + "\t".join(eye_event.dtype.names) + "\n")
            for evid, ev in zip(eventID, eye_event):
                output = [str(evid)] + [str(x) for x in ev]
                fd.write('\t'.join(output) + '\n')
        self.parent.statusbar.SetStatusText('Eye event data saved in {0}'.format(fn_eyeevent))
    
    def OnSave(self, event):
        # copy the parameter values in the control panel into the param dict
        for key in self.tc.keys():
            self.param['eex'][key] = float(self.tc[key].GetValue())
            
        # save parameter values in odML file
        fn_odML = "{datadir}/{sess}/{sess}_rec{rec}.odml".format(**self.param)
        save_eexparam(fn_odML, self.param)
        save_calibparam(fn_odML, self.param)
        self.param_default = copy.deepcopy(self.param)
        self.parent.statusbar.SetStatusText('Parameter values saved in {0}'.format(fn_odML))
        
class MainFrame(wx.Frame):
    def __init__(self, parent, id, title, param):
        wx.Frame.__init__(self, parent, id, title)

        # load task event data from task.csv file
        events = get_events(param)
        
        # initialize lvd file reader
        fn_data = "{datadir}/{sess}/{datafile}".format(**param)
        lvd_reader = lvdread.LVDReader(fn_data)
        
        # define widgets
        self.plotpanel = PlotPanel(self, -1)
        self.controlpanel = ControlPanel(self, -1, events, lvd_reader, param)
        self.statusbar = self.CreateStatusBar()
        
        # layout
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(self.plotpanel, 1, wx.EXPAND)
        vbox.Add(self.controlpanel, 0, wx.EXPAND)
        vbox.Add(self.statusbar, 0, wx.EXPAND)
        self.SetSizer(vbox)
        
        # initialize
        self.Maximize()
        self.Show(True)


if __name__ == '__main__':
    from argparse import ArgumentParser
    
    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=conf['datadir'])
    parser.add_argument("--stimdir", default=conf['stimdir'])
    parser.add_argument("--sess", "--session")
    parser.add_argument("--rec")
    parser.add_argument("--blk", "--block")
    parser.add_argument("--data", nargs=3, default=None)
    parser.add_argument("--calib_sess", dest="calibsess")
    parser.add_argument("--calib_rec", dest="calibrec")
    parser.add_argument("--calib_blk", dest="calibblk")
    parser.add_argument("--calib", nargs=3, default=None)
    parser.add_argument("--calib_method", dest="calibmeth")
    parser.add_argument("--calib_ignore", dest="calibignore", nargs='*', type=int, default=[-1,])
    parser.add_argument("--on_event", default=conf['eyevex_gui2']['on_event'])
    parser.add_argument("--off_event", default=conf['eyevex_gui2']['off_event'])
    arg = parser.parse_args()
    
    # store commandline arguments in local variables
    datadir = arg.datadir
    if arg.data is None:
        sess = arg.sess
        rec = arg.rec
        blk = arg.blk
    else:
        sess, rec, blk = arg.data
    blk = int(blk)
        
    if arg.calib is None:
        if None in (arg.calibsess, arg.calibrec, arg.calibblk):
            calibsess = sess
            calibrec = rec
            calibblk = blk
        else:
            calibsess = arg.calibsess
            calibrec = arg.calibrec
            calibblk = arg.calibblk
    else:
        calibsess, calibrec, calibblk = arg.calib
    calibblk = int(calibblk)
        
    # load metadata parameters from odML file
    param = load_odml(datadir, sess, rec, blk)
    
    # add parameters from command line arguments
    param.update(
                 {
                  'datadir': datadir,
                  'sess': sess, 'rec': rec, 'blk': blk,
#                  'calib_sess': calibsess, 'calib_rec': calibrec, 'calib_blk': calibblk,
                  'stimdir': arg.stimdir,
                  'on_event': arg.on_event, 'off_event': arg.off_event,
                  }
                 )
    
    # load eye event extraction parameters from configuration file, if they are
    # not found in odML
    if 'eex' not in param:
        param['eex'] = conf['eyevex_gui2']['eex_param']
        
    # load calibration parameters
    if arg.calibmeth in ['linear', 'cubic', 'quintic', 'thin_plate']:
        param['calib'] = {'Coeffs': arg.calibmeth,
                          'Ignore': arg.calibignore,
                          'sess': calibsess,
                          'rec': calibrec,
                          'blk': calibblk
                          }
    else:
        param['calib'] = load_calibparam(datadir, calibsess, calibrec, calibblk)
        
#    print arg
#    print param['eex']
#    print param['calib']
    
    # initiate GUI
    matplotlib.interactive(True)
    app = wx.App(redirect=False)
    title = "EYEVEX: Eye Event Extractor (%(sess)s_rec%(rec)s)" % {'sess':sess, 'rec':rec}
    mainframe = MainFrame(None, -1, title, param)
    mainframe.controlpanel.OnDraw(None)
    app.MainLoop()
