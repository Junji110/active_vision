'''
active_vision/prep/odml_av3.py

Module for generating odML file for active vision data sets containing .lvd data files

Written by:
    Junji Ito (j.ito@fz-juelich.de) 
        Richard Meyes (r.meyes@fz-juelich.de),
            Lyuba Zehl (l.zehl@fz-juelich.de) and
    on 2013.12.12
'''
# import packages
import os
import re
import copy
import json
import datetime

import numpy as np
import scipy.io as sio

import active_vision.fileio.lvdread as lvdread

### =========== metadata loading functions ===========
def load_lvd(datadir, session, rec):
    """ loads the metadata (e.g. header information) from an .lvd file """
    fn_lvd = find_filenames(datadir, session, rec, 'lvd')
        
    props = {}

    for fn in fn_lvd:
        # identify PC number
        valid_fn = re.compile('[0-9]+_[0-9]+_rec[0-9]+_pc([0-9])\.lvd')
        match = valid_fn.match(fn.split('/')[-1])
        if match is None:
            raise ValueError("'{0}' is not a valid filename.".format(fn))
        pcnum = match.group(1)
        
        # load metadata from the file
        lvd_reader = lvdread.LVDReader(fn)
        lvd_header = lvd_reader.get_header()
        lvd_param = lvd_reader.get_param()
        
        # add Dataset/AnalogData* properties
        sectname = '/Dataset/AnalogData{0}'.format(pcnum)
        props[sectname] = [
            {"_type": "Property", "name": "File", "value": fn, "unit": "", "dtype": "string"},
            {"_type": "Property", 'name': 'SamplesAcquired', "value": lvd_param['data_length'], "unit": "", "dtype": "int"},
            {"_type": "Property", 'name': 'ChannelName', "value": lvd_header['AIUsedChannelName'], "unit": "", "dtype": "string"},
            ]
        
        # add Recording/HardwareSettings/DataAcquisiiton* properties
        sectname = '/Recording/HardwareSettings/DataAcquisition{0}'.format(pcnum)
        props[sectname] = [
            {"_type": "Property", 'name': 'AISampleRate', "value": lvd_header['AISampleRate'], "unit": "Hz", "dtype": "float"},
            {"_type": "Property", 'name': 'AIUsedChannelCount', "value": lvd_header['AIUsedChannelCount'], "unit": "", "dtype": "int"},
            {"_type": "Property", 'name': 'HwChannel', "value": lvd_header['HwChannel'], "unit": "", "dtype": "string"},
            {"_type": "Property", 'name': 'InputRangeLow', "value": lvd_header['InputRangeLow'], "unit": "", "dtype": "float"},
            {"_type": "Property", 'name': 'InputRangeHigh', "value": lvd_header['InputRangeHigh'], "unit": "", "dtype": "float"},
            {"_type": "Property", 'name': 'SensorRangeLow', "value": lvd_header['SensorRangeLow'], "unit": "", "dtype": "float"},
            {"_type": "Property", 'name': 'SensorRangeHigh', "value": lvd_header['SensorRangeHigh'], "unit": "", "dtype": "float"},
            ]
            
    return props

def load_task(datadir, session, rec):
    filename = find_filenames(datadir, session, rec, 'task')[0]
    converters = genfromtxt_converters()
    taskdata = np.genfromtxt(filename, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters) # structured array
    block_info = parse_taskdata(taskdata)
    props = {}
    
    # add Recording properties
    sectname = '/Recording'
    props[sectname] = []                                         
    Start = (taskdata['DATE'][0] + " " + taskdata['TIME'][0]).replace('/', '-')
    End = (taskdata['DATE'][-1] + " " + taskdata['TIME'][-1]).replace('/', '-')
    
    block_count = len(block_info) 
    block_task_type = [x['task_type'] for x in block_info]
    block_start = [x['block_start'] for x in block_info]
    props[sectname].extend([
        {"_type": "Property", "name": "Start", "value": Start, "unit": "", "dtype": "datetime"},
        {"_type": "Property", "name": "End", "value": End, "unit": "", "dtype": "datetime"},
        {"_type": "Property", "name": "block_count", "value": block_count, "unit": "", "dtype": "int"},
        {"_type": "Property", "name": "block_task_type", "value": block_task_type, "unit": "", "dtype": "int"},
        {"_type": "Property", "name": "block_start", "value": block_start, "unit": "", "dtype": "int"}
        ])
        
    # add Dataset/EventData properties
    sectname_event = '/Dataset/EventData'
    props[sectname_event] = []
    props[sectname_event].append({"_type": "Property", "name": "File", "value": filename, "unit": "", "dtype": "string"})
    block_info = parse_taskdata(taskdata)
    for i_block, info in enumerate(block_info):
        for key, val in info.items():
            name = "blk{0}_{1}".format(i_block + 1, key)
            props[sectname_event].append({"_type": "Property", "name": name, "value": val, "unit": "", "dtype": "int"})
            
    return props

def load_imginfo(datadir, session, rec):
    filenames = find_filenames(datadir, session, rec, 'imginfo')
    num_block = len(filenames)
    
    props = {}
    
    # add Dataset/StimulusData properties
    sectname = '/Dataset/StimulusData'
    props[sectname] = []
    for fn in filenames:
        # identify block number
        valid_fn = re.compile('imginfo_blk([0-9]+)_tsk([0-9])_.*\.mat')
        match = valid_fn.match(fn.split('/')[-1])
        if match is None:
            raise ValueError("'{0}' is not a valid filename.".format(fn))
        block = match.group(1)
        
        propname = "blk{0}_File".format(block)
        props[sectname].append({"_type": "Property", "name": propname, "value": fn, "unit": "", "dtype": "string"})
        
        imginfo = sio.loadmat(fn)
        t_info = imginfo['t_info'][0,0]
        for name in t_info.dtype.names:
            value, dtype, ndim = matdata2value(t_info[name])
            if ndim == 2:
                for i, val in enumerate(value):
                    propname = "blk{0}_{1}{2}".format(block, name, i+1)
                    props[sectname].append({"_type": "Property", "name": propname, "value": val, "unit": "", "dtype": dtype})
            else:
                propname = "blk{0}_{1}".format(block, name)
                props[sectname].append({"_type": "Property", "name": propname, "value": value, "unit": "", "dtype": dtype})
        
    return props

def load_param(datadir, session, rec):
    filenames = find_filenames(datadir, session, rec, 'param')
    num_task = len(filenames)
    
    props = {}
    
    # add Experiment/Behavior properties
    sectname = '/Experiment/Behavior'
    for fn in filenames:
        # identify task type
        valid_fn = re.compile('.*_param_tsk([0-9])_.*\.csv')
        match = valid_fn.match(fn.split('/')[-1])
        if match is None:
            raise ValueError("'{0}' is not a valid filename.".format(fn))
        task_type = match.group(1)
        subsectname = '/{0}/Task{1}'.format(sectname, task_type)
        props[subsectname] = []
        
        props[subsectname].append({"_type": "Property", "name": "File", "value": fn, "unit": "", "dtype": "string"})
        
        # add properties in param file
        converters = genfromtxt_converters()
        params = np.genfromtxt(fn, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
        for name in params.dtype.names:
            if name[0:2] in ['pf', 'pe', 'pv', 'g_'] or name is 'TIMING_CLOCK':
                if isinstance(params[name], np.ndarray) and len(params[name].shape) > 0:
                    value = params[name].tolist()
                    dtype = type(np.asscalar(params[name][0])).__name__
                else:
                    value = params[name]
                    dtype = type(np.asscalar(params[name])).__name__
                props[subsectname].append({"_type": "Property", "name": name, "value": value, "unit": "", "dtype": dtype})
    return props

### =========== utility functions ===========
def genfromtxt_converters():
    """ converts strings or numbers into long integers """
    convfunc = lambda x: long(x)
    return {'INTERVAL': convfunc, 'TIMING_CLOCK': convfunc, 'GL_TIMER_VAL': convfunc}
    
def find_filenames(datadir, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd']:
        raise ValueError("filetype must be either of 'imginfo', 'stimtiming', 'param', 'parameter', 'task', or 'daq'.")
    
    # identify the names of metadata files
    if filetype in ['daq', 'lvd']:
        searchdir = "{0}/{1}".format(datadir, session)
        searchtoken = 'rec{0}_pc'.format(rec)
    else:
        searchdir = "{ddir}/{sess}/{sess}_rec{rec}".format(ddir=datadir, sess=session, rec=rec)
        searchtoken = filetype
        
    filenames = os.listdir(searchdir)
    fn_found = []
    for fn in filenames:
        if searchtoken in fn:
            fn_found.append("{0}/{1}".format(searchdir, fn))
    if len(fn_found) == 0:
        raise IOError("Files of type '{0}' not found.".format(filetype))
    
    return fn_found

def matdata2value(matdata):
    # in matlab data structure different data types are stored in arrays of different shapes
    # string: [1,] array
    # number: [1, 1] array
    # n-list: [n, 1] array
    # (m, n)-list: [m, n] array
    if matdata.size == 0:
        value = "(*empty*)"
        dtype = "string"
        ndim = -1
    elif matdata.ndim < 2:
        value = matdata[0]
        dtype = 'string'
        ndim = -1
    elif matdata.shape[0] == 1:
        value = matdata[0, 0]
        dtype = type(np.asscalar(matdata)).__name__
        ndim = 0
    elif matdata.shape[1] == 1:
        value = matdata[:, 0].tolist()
        dtype = type(np.asscalar(matdata[0])).__name__
        ndim = 1
    else:
        value = [matdata[:, x].tolist() for x in range(matdata.shape[1])]
        dtype = type(np.asscalar(matdata[0, 0])).__name__
        ndim = 2
    return value, dtype, ndim

def parse_taskdata(taskdata):
    block = taskdata['g_block_num']
    trial = taskdata['TRIAL_NUM']
    task_type_all = taskdata['g_task_switch']
    timing_all = taskdata['TIMING_CLOCK']
    stimID_all = taskdata['t_tgt_data']
    success_all = taskdata['SF_FLG']
    eventID_all = taskdata['log_task_ctrl']
    
    num_block = block.max() + 1
    block_info = []
    for i_block in range(1, num_block):
        num_trial = trial[block == i_block].max() + 1
        task_type = task_type_all[block == i_block][0]
        block_start = timing_all[block == i_block][0]
        info = {'block_start': block_start, 'task_type': task_type, 'num_trials': num_trial - 1, 'trial_start': [], 'trial_end': [], 'stimID': [], 'success': []}
        for i_trial in range(1, num_trial):
#        for i_trial in range(0, num_trial):
            trial_mask = (block == i_block) & (trial == i_trial)
            timing = timing_all[trial_mask]
            stimID = stimID_all[trial_mask]
            success = success_all[trial_mask]
            eventID = eventID_all[trial_mask]
            
            info['trial_start'].append(timing.min())
            info['trial_end'].append(timing.max())
            info['stimID'].append(stimID[0])
            if 0 in success:
                info['success'].append(eventID.min())
            else:
                info['success'].append(1)
        block_info.append(info)
    
    return block_info
    
def print_metadata(metadata):
    def print_section(sect, ntab=0, tabstr='    '):
        tabs = tabstr * ntab
        print("{0}{1} (type: {2})".format(tabs, sect.name, sect.type))
        tabs = tabstr * (ntab + 1)
        for prop in sect.properties:
            if isinstance(prop.value, list):
                data = [str(x.data) for x in prop.value]
                unit = "" if prop.value[0].unit is None else prop.value[0].unit
                print("{0}{1}: [{2}] {3} (dtype: {4})".format(tabs, prop.name, ', '.join(data), unit, prop.value[0].dtype))
            else:
                unit = "" if prop.value.unit is None else prop.value.unit
                print("{0}{1}: {2} {3} (dtype: {4})".format(tabs, prop.name, prop.value.data, unit, prop.value.dtype))
        print
        
        for subsect in sect.sections:
            print_section(subsect, ntab+1)
            
    print("Version {0}, Created by {1} on {2}".format(metadata.version, metadata.author, metadata.date))
    print
    for sect in metadata.sections:
        print_section(sect)
        print
        
# def load_props_from_file(datadir, session, rec, default_props):
#     props = copy.deepcopy(default_props)
#     props.update(load_task(datadir, session, rec))
#     props.update(load_imginfo(datadir, session, rec))
#     props.update(load_param(datadir, session, rec, default_props))
#     props.update(load_lvd(datadir, session, rec))
#     return props

if __name__ == '__main__':
    
    # import packages
    import ox.factory as oxf
    from argparse import ArgumentParser
    
    # default parameters:
    defauth = "R.Meyes"
    # recording parameters
    defsess = "20131111"
    defrec = "1"
    # template parameters
    defsect = "odml_sections_awake_v{version}.json"
    defprop = "odml_properties_awake_v{version}.json"
    defsuppl = "odml_suppl_v{version}.json"
    
    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    template_version = conf["odml_av"]["template_version"]
    # print template_version
    # quit()
    
    # parse command line options    
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=conf['datadir'])
    parser.add_argument("--session", "--sess", default=defsess)
    parser.add_argument("--rec", default=defrec)
    parser.add_argument("--suppl", "--conf", default=defsuppl)
    parser.add_argument("--template_dir", default=conf['odml_av']["template_dir"])
    parser.add_argument("--odml", default=None)
    parser.add_argument("--author", default=defauth)
    arg = parser.parse_args()
    
    # load .json templates
    # sections
    fn_template = "/".join([arg.template_dir, defsect.format(version=template_version)])
    with open(fn_template, 'r') as fd_template:
        section_info = json.load(fd_template)
    # properties   
    fn_template = "/".join([arg.template_dir, defprop.format(version=template_version)])
    with open(fn_template, 'r') as fd_template:
        props = json.load(fd_template)
    # supplementary info    
    fn_template = "/".join([arg.template_dir, defsuppl.format(version=template_version)])
    with open(fn_template, 'r') as fd_template:
        suppl_info = json.load(fd_template)
    
    # generate instance of odml.Factory
    fobj = oxf.Factory(suppl_info["Author"])
    # add sections tree
    fobj.add_section_dict(section_info)
    
    props.update(load_task(arg.datadir, arg.session, arg.rec))
    props.update(load_imginfo(arg.datadir, arg.session, arg.rec))
    props.update(load_param(arg.datadir, arg.session, arg.rec))
    props.update(load_lvd(arg.datadir, arg.session, arg.rec))

    
    # put properties from template
    for sect_path in props.keys():
        for prop in props[sect_path]:
            fobj.add_property(prop, sect_path)
    
    # put supplementary info        
    for sect_path in suppl_info.keys():
        if sect_path not in ["Author", "Co-Author", "Version"]: #TODO: these are entries for document
            for prop in suppl_info[sect_path]:
                prop_path = ":".join([sect_path, prop["name"]])
                # TODO: talk to Lyuba about implementation in the factory
                try:
                    fobj.replace_property(copy.deepcopy(prop), prop_path)
                except KeyError:
                    fobj.add_property(copy.deepcopy(prop), sect_path)
    
    # save .odML file
    fn_odML_dir = "{datadir}".format(datadir=arg.datadir)
    fn_odML_sess = "{sess}_rec{rec}_v{ver}".format(sess=defsess, rec=defrec, ver=template_version)     
    fobj.save(fn_odML_dir, fn_odML_sess, fileformat='odml')
    print("odML structure saved in {0} as {1}.odml".format(fn_odML_dir, fn_odML_sess))


