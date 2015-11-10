'''
active_vision/prep/odml_av2.py

Module for generating odML file for active vision datasets containing .lvd data files

Written by Junji Ito (j.ito@fz-juelich.de) on 2013.09.20
'''
import os
import re
import datetime
import json

import numpy as np
import scipy.io as sio
import odml

import lvdread


class odMLFactory(object):
    def __init__(self, section_info={}, default_props={}, filename='', strict=True):
        self.sect_info = section_info
        self.def_props = default_props
        self.strict = strict
        if filename:
            self._sections = self.__get_sections_from_file(filename)
        else:
            self._sections = {}
            for sectname in self.__get_top_section_names():
                self._sections[sectname] = self.__gen_section(sectname)
    
    def __get_sections_from_file(self, filename):
        # load odML from file
        with open(filename, 'r') as fd_odML:
            metadata = odml.tools.xmlparser.XMLReader().fromFile(fd_odML)
        sections = {}
        for sect in metadata.sections:
            sections[sect.name] = sect
        return sections
    
    def __get_top_section_names(self):
        topsectnames = [] 
        for key in self.sect_info:
            if '/' not in key:
                topsectnames.append(key)
        return topsectnames
            
    def __add_property(self, sect, prop, strict=True):
        if sect.contains(odml.Property(prop['name'], None)):
            sect.remove(sect.properties[prop['name']])
        elif strict is True:
            raise ValueError("Property '{0}' does not exist in section '{1}'.".format(prop['name'], sect.name))
        sect.append(odml.Property(**prop))
        
    def __gen_section(self, name, parent=''):
        longname = parent + name
        sect = odml.Section(name=name, type=self.sect_info[longname]['type'])
        
        # add properties
        if longname in self.def_props:
            for prop in self.def_props[longname]:
                self.__add_property(sect, prop, strict=False)
            
        # add subsections
        if 'subsections' in self.sect_info[longname]:
            for subsectname in self.sect_info[longname]['subsections']:
                sect.append(self.__gen_section(subsectname, longname+'/'))
                
        return sect
            
    def __get_section_from_longname(self, sectname):
        def get_subsect(sect, names):
            if len(names) == 0:
                return sect
            else:
                return get_subsect(sect.sections[names[0]], names[1:])
            
        names = sectname.split('/')
        if names[0] not in self._sections:
            return None
        else:
            return get_subsect(self._sections[names[0]], names[1:])

    def put_values(self, properties):
        for sectname, sectprops in properties.items():
            sect = self.__get_section_from_longname(sectname)
            if sect is None:
                raise ValueError("Invalid section name '{0}'".format(sectname))
            else:
                for prop in sectprops:
                    self.__add_property(sect, prop, self.strict)
    
    def get_odml(self, author, version=None):
        metadata = odml.Document(author, datetime.date.today(), version)
        for sect in self._sections.values():
            metadata.append(sect)
        return metadata
    
    def save_odml(self, filename, author, version=None):
        metadata = self.get_odml(author, version)
        odml.tools.xmlparser.XMLWriter(metadata).write_file(filename)
    

### =========== metadata loading functions ===========
def load_lvd(datadir, session, rec):
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
        sectname = 'Dataset/AnalogData{0}'.format(pcnum)
        props[sectname] = []
        props[sectname].extend([
            {"name": "File", "value": fn, "unit": "", "dtype": "string"},
            {'name': 'SamplesAcquired', "value": lvd_param['data_length'], "unit": "", "dtype": "int"},
            {'name': 'ChannelName', "value": lvd_header['AIUsedChannelName'], "unit": "", "dtype": "string"},
            ])
        
        # add Recording/HardwareSettings/DataAcquisiiton* properties
        sectname = 'Recording/HardwareSettings/DataAcquisition{0}'.format(pcnum)
        props[sectname] = []
        props[sectname].extend([
            {'name': 'AISampleRate', "value": lvd_header['AISampleRate'], "unit": "Hz", "dtype": "float"},
            {'name': 'AIUsedChannelCount', "value": lvd_header['AIUsedChannelCount'], "unit": "", "dtype": "int"},
            {'name': 'HwChannel', "value": lvd_header['HwChannel'], "unit": "", "dtype": "string"},
            {'name': 'InputRangeLow', "value": lvd_header['InputRangeLow'], "unit": "", "dtype": "float"},
            {'name': 'InputRangeHigh', "value": lvd_header['InputRangeHigh'], "unit": "", "dtype": "float"},
            {'name': 'SensorRangeLow', "value": lvd_header['SensorRangeLow'], "unit": "", "dtype": "float"},
            {'name': 'SensorRangeHigh', "value": lvd_header['SensorRangeHigh'], "unit": "", "dtype": "float"},
            ])
    
    return props

def load_task(datadir, session, rec):
    filename = find_filenames(datadir, session, rec, 'task')[0]
    converters = genfromtxt_converters()
    taskdata = np.genfromtxt(filename, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
    
    props = {}
    
    # add Recording properties
    sectname = 'Recording'
    props[sectname] = []
    Start = (taskdata['DATE'][0] + " " + taskdata['TIME'][0]).replace('/', '-')
    End = (taskdata['DATE'][-1] + " " + taskdata['TIME'][-1]).replace('/', '-')
    props['Recording'].extend([
        {"name": "Start", "value": Start, "unit": "", "dtype": "datetime"},
        {"name": "End", "value": End, "unit": "", "dtype": "datetime"},
        ])
        
    # add Dataset/EventData properties
    sectname_event = 'Dataset/EventData'
    props[sectname_event] = []
    props[sectname_event].append({"name": "File", "value": filename, "unit": "", "dtype": "string"})
    block_info = parse_taskdata(taskdata)
    for i_block, info in enumerate(block_info):
        for key, val in info.items():
            name = "blk{0}_{1}".format(i_block + 1, key)
            props[sectname_event].append({"name": name, "value": val, "unit": "", "dtype": "int"})
    
    return props

def load_imginfo(datadir, session, rec):
    filenames = find_filenames(datadir, session, rec, 'imginfo')
    num_block = len(filenames)
    
    props = {}
    
    # add Dataset/StimulusData properties
    sectname = 'Dataset/StimulusData'
    props[sectname] = []
    for fn in filenames:
        # identify block number
        valid_fn = re.compile('imginfo_blk([0-9]+)_tsk([0-9])_.*\.mat')
        match = valid_fn.match(fn.split('/')[-1])
        if match is None:
            raise ValueError("'{0}' is not a valid filename.".format(fn))
        block = match.group(1)
        
        propname = "blk{0}_File".format(block)
        props[sectname].append({"name": propname, "value": fn, "unit": "", "dtype": "string"})
        
        imginfo = sio.loadmat(fn)
        t_info = imginfo['t_info'][0,0]
        for name in t_info.dtype.names:
            value, dtype, ndim = matdata2value(t_info[name])
            if ndim == 2:
                for i, val in enumerate(value):
                    propname = "blk{0}_{1}{2}".format(block, name, i+1)
                    props[sectname].append({"name": propname, "value": val, "unit": "", "dtype": dtype})
            else:
                propname = "blk{0}_{1}".format(block, name)
                props[sectname].append({"name": propname, "value": value, "unit": "", "dtype": dtype})
    
    return props

def load_param(datadir, session, rec, default_props):
    filenames = find_filenames(datadir, session, rec, 'param')
    num_task = len(filenames)
    
    props = {}
    
    # add Experiment/Behavior properties
    sectname = 'Experiment/Behavior'
    for fn in filenames:
        # identify task type
        valid_fn = re.compile('.*_param_tsk([0-9])_.*\.csv')
        match = valid_fn.match(fn.split('/')[-1])
        if match is None:
            raise ValueError("'{0}' is not a valid filename.".format(fn))
        task_type = match.group(1)
        subsectname = '{0}/Task{1}'.format(sectname, task_type)
        props[subsectname] = []
        
        props[subsectname].append({"name": "File", "value": fn, "unit": "", "dtype": "string"})
        
        # add properties from suppl_prop
        subsectname_suppl = '{0}@task{1}'.format(subsectname, task_type)
        if subsectname_suppl in default_props:
            props[subsectname].extend(default_props[subsectname_suppl])
        
        # add properties in param file
        converters = genfromtxt_converters()
        params = np.genfromtxt(fn, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
        for name in params.dtype.names:
            if name[0:2] in ['pf', 'pe', 'pv', 'g_']:
                if isinstance(params[name], np.ndarray) and len(params[name].shape) > 0:
                    value = params[name].tolist()
                    dtype = type(np.asscalar(params[name][0])).__name__
                else:
                    value = params[name]
                    dtype = type(np.asscalar(params[name])).__name__
                props[subsectname].append({"name": name, "value": value, "unit": "", "dtype": dtype})
            
    return props

### =========== utility functions ===========
def genfromtxt_converters():
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
    # n-list: [n, 1] or [1, n] array
    # (m, n)-list: [m, n] array
    if matdata.size == 0:
        value = "(*empty*)"
        dtype = "string"
        ndim = -1
    elif matdata.ndim < 2:
        value = matdata[0]
        dtype = 'string'
        ndim = -1
    elif matdata.shape == (1, 1):
        value = matdata[0, 0]
        dtype = type(np.asscalar(matdata)).__name__
        ndim = 0
    elif matdata.shape[0] == 1:
        value = matdata[0, 0]
        dtype = type(np.asscalar(matdata[0, 0])).__name__
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
        info = {'task_type': task_type, 'num_trials': num_trial - 1}
        if num_trial > 1:
            info['trial_start'] = []
            info['trial_end'] = []
            info['stimID'] = []
            info['success'] = []
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
        
def load_props_from_file(datadir, session, rec, default_props):
    props = []
    props.append(load_task(datadir, session, rec))
    props.append(load_imginfo(datadir, session, rec))
    props.append(load_param(datadir, session, rec, default_props))
    props.append(load_lvd(datadir, session, rec))
    return props
            
    
if __name__ == '__main__':
    from argparse import ArgumentParser
    
    # load configuration file
    scriptdir = os.path.abspath(os.path.dirname(__file__))
    if os.path.exists(scriptdir + "/conf.json"):
        conf = json.load(open(scriptdir + "/conf.json"))
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--datadir", default=conf['datadir'])
    parser.add_argument("--session", "--sess")
    parser.add_argument("--rec")
    parser.add_argument("--suppl", "--conf")
    parser.add_argument("--template_dir", default=conf['odml_av']["template_dir"])
    parser.add_argument("--odml", default=None)
    arg = parser.parse_args()
    
    # load supplementary information
    with open(arg.suppl, 'r') as fd_suppl:
        suppl_info = json.load(fd_suppl)
    
    # generate an instance of odMLFactory
    if arg.odml:
        # if a filename is given, generate an odMLFactory instance from the file
        fn_odML = arg.odml
        odml_factory = odMLFactory(filename=fn_odML)
    else:
        # otherwise, generate an odMLFactory instance from templates
        # - load section template
        fn_sections = arg.template_dir + "/" + suppl_info["Template"]["sect_template"]
        with open(fn_sections, 'r') as fd_sections:
            section_info = json.load(fd_sections)
        
        # - load property template
        fn_props = arg.template_dir + "/" + suppl_info["Template"]["prop_template"]
        with open(fn_props, 'r') as fd_props:
            default_props = json.load(fd_props)
        
        # - initialize an odMLFactory instance with the templates
        odml_factory = odMLFactory(section_info, default_props, strict=False)
        
        # - put property values contained in metadata files
        props_in_file = load_props_from_file(arg.datadir, arg.session, arg.rec, default_props)
        for props in props_in_file:
            odml_factory.put_values(props)
        
        fn_odML = "{0}_rec{1}.odml".format(arg.session, arg.rec)
        
    # put supplementary property values
    suppl_props = suppl_info["SupplProp"]
    odml_factory.put_values(suppl_props)
    
    # save the odML structure in a file
    author = suppl_info["Author"]
    version = suppl_info["Version"]
    odml_factory.save_odml(fn_odML, author, version)
    print("odML structure saved in {0}".format(fn_odML))
    print
    
    # print out the odML structure for a check
    print_metadata(odml_factory.get_odml(author, version))
    