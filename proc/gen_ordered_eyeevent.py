import os
import sys
import re

import numpy as np
import scipy as sp
import odml.tools.xmlparser as odmlparser

sys.path.append("/users/ito/toolbox")
import active_vision.utils as avutils


# import parameters
from parameters.gen_ordered_eyeevent import *


def load_odml(fn_odml, blk):
    with open(fn_odml, 'r') as fd:
        metadata = odmlparser.XMLReader().fromFile(fd)

    prefix = 'blk{0}_'.format(blk)

    if metadata['Dataset']['EventData'].properties[prefix+'task_type'].value.data != 3:
        raise ValueError("No free viewing trials in the specified sess/rec/blk.")

    # store all relevant metadata parameters in one dictionary
    param = {
        'sampling_rate':
            metadata['Recording']['HardwareSettings']['DataAcquisition1'].properties['AISampleRate'].value.data,
        'pxlperdeg': metadata['Recording']['HardwareSettings']['Monitor'].properties['PixelPerDegree'].value.data,

        'depth':  metadata['Setup']['Electrode1'].properties['Depth'].value.data,

        'evID': [x.data for x in metadata['Experiment']['Behavior']['Task3'].properties['EventID'].value],
        'evtype': [x.data for x in metadata['Experiment']['Behavior']['Task3'].properties['EventType'].value],

        'fn_V1': os.path.basename(metadata['Dataset']['AnalogData1'].properties['File'].value.data),
        'ChannelName_V1': [str(x.data) for x in metadata['Dataset']['AnalogData1'].properties['ChannelName'].value],

        'taskfile': os.path.basename(metadata['Dataset']['EventData'].properties['File'].value.data),
        'num_trials': metadata['Dataset']['EventData'].properties[prefix+'num_trials'].value.data,
        'success': [x.data for x in metadata['Dataset']['EventData'].properties[prefix+'success'].value],
        'stimID': [x.data for x in metadata['Dataset']['EventData'].properties[prefix+'stimID'].value],
        'trial_start': [x.data for x in metadata['Dataset']['EventData'].properties[prefix+'trial_start'].value],
        'trial_end': [x.data for x in metadata['Dataset']['EventData'].properties[prefix+'trial_end'].value],

        'stimsetname': metadata['Dataset']['StimulusData'].properties[prefix+'setname'].value.data,
        'imgformat': metadata['Dataset']['StimulusData'].properties[prefix+'imgformat'].value.data,

        'stimID2imgID': [x.data for x in metadata['Dataset']['StimulusData'].properties[prefix+'imgID'].value],
    }

    return param


def load_calibparam(datadir, sbj, sess, rec, blk):
    fn_odml = "{0}/{1}/{2}/{2}_rec{3}.odml".format(datadir, sbj, sess, rec)
    with open(fn_odml, 'r') as fd:
        metadata = odmlparser.XMLReader().fromFile(fd)

    sect = metadata['Dataset']['AnalogData3']
    if sect.find_related(key='CalibParams') is None:
        raise ValueError("Section for CalibParam not found in {0}".format(fn_odml))

    param = {
        'calib_sess': sess,
        'calib_rec': rec,
        'calib_blk': blk,
        'Ignore': [-1],
    }

    sect_calib = sect['CalibParams']
    re_propname = re.compile('blk{0}_(.*)'.format(blk))
    for prop in sect_calib.properties:
        match = re_propname.match(prop.name)
        if match:
            paramkey = match.group(1)
            if isinstance(prop.value, list):
                param[paramkey] = [x.data for x in prop.value]
            elif paramkey == "Ignore":
                param[paramkey] = [prop.value.data, ]
            else:
                param[paramkey] = prop.value.data

    return param


def load_dataset_info(dataset):
    _, sbj, sess, rec, blk, _, _ = dataset

    # --- find filenames
    fn_odml = avutils.find_filenames(rawdir, sbj, sess, rec, 'odml')[0]
    fn_task = avutils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]
    # fn_V1 = avutils.find_filenames(rawdir, sbj, sess, rec, 'hdf5', 'pc1')[0]
    # fn_IT = avutils.find_filenames(rawdir, sbj, sess, rec, 'hdf5', 'pc2')[0]

    # construct params dict
    params = {'sbj': sbj, 'sess': sess, 'rec': rec, 'blk': blk}

    # --- load parameters from odML file
    odml_params = load_odml(fn_odml, blk)
    params.update(odml_params)
    calib_params = load_calibparam(rawdir, sbj, sess, rec, blk-1)
    params.update(calib_params)

    # # --- load parameters from downsampled data file
    # data_reader = {'V1': hdf5read.HDF5Reader(fn_V1), 'IT': hdf5read.HDF5Reader(fn_IT)}
    # data_params = data_reader['V1'].get_param()
    # params.update({key: data_params[key] for key in ['downsample_factor', ]})
    data_reader = None

    # load task events from task file
    task_events, task_params = avutils.load_task(fn_task, int(blk))
    params.update(task_params)

    print("Dataset {}:{}_rec{}_blk{}, information successfully loaded.".format(sbj, sess, rec, blk))

    return data_reader, params, task_events


def load_eyeevent_data(dataset):
    _, sbj, sess, rec, blk, _, _ = dataset
    fn_eyevex = "{}/{}/eyeevents/{}_rec{}_blk{}_eyeevent.dat".format(prepdir, sbj, sess, rec, blk)
    return avutils.load_eyevex(fn_eyevex)


def load_stimulus_info(params):
    stimsetdir = "/".join([stimdir, params['stimsetname']])
    imgIDs = params['stimID2imgID']
    objIDs = {}
    objpos = {}
    objsize = {}
    bgID = {}
    objdeg = {}
    objnum = {}
    for imgID in imgIDs:
        fn_imgmat = "{dir}/{id}.mat".format(dir=stimsetdir, id=imgID)
        if os.path.exists(fn_imgmat):
            imginfo = sp.io.loadmat(fn_imgmat, squeeze_me=True, struct_as_record=False)
            info = imginfo['information']
            bgID[imgID] = "{0:03d}".format(info.backgroundid)
            objIDs[imgID] = np.array(info.objectid, int)
            if objIDs[imgID].size == 1 and np.isnan(objIDs):
                continue
            objpos[imgID] = np.array(zip(info.object_x_position, info.object_y_position), int)
            objsize[imgID] = np.array(zip(info.object_x_size, info.object_y_size), int)
            objdeg[imgID] = 2.0
            objnum[imgID] = len(objIDs[imgID])
        else:
            print("{}: .mat file for {}.png not found".format(stimsetdir, imgID))
        objpos[imgID][:, 0] = objpos[imgID][:, 0] - stim_size[0] / 2
        objpos[imgID][:, 1] = -objpos[imgID][:, 1] + stim_size[1] / 2
        objpos[imgID] = objpos[imgID] / pxlperdeg
        objsize[imgID] = objsize[imgID] / pxlperdeg
    return objIDs, objpos, objsize, bgID, objdeg, objnum


def convert_trial_time_to_clock_count(sacinfo, fixinfo, task_events, params, evID_on=311):
    sampling_rate = params['sampling_rate']
    for i in range(len(sacinfo['trialID'])):
        trialID = sacinfo['trialID'][i]
        task_events_trial = task_events[task_events['trial'] == trialID]
        idx_on = task_events_trial['evtime'][task_events_trial['evID'] == evID_on][0]
        sacinfo['on'][i] = long(sacinfo['on'][i] * sampling_rate) + idx_on
        sacinfo['off'][i] = long(sacinfo['off'][i] * sampling_rate) + idx_on
        fixinfo['on'][i] = long(fixinfo['on'][i] * sampling_rate) + idx_on
        fixinfo['off'][i] = long(fixinfo['off'][i] * sampling_rate) + idx_on


if __name__ == "__main__":
    for dataset in datasets:
        species, sbj, sess, rec, blk, stimset, _ = dataset
        fn_eyevex_in = "{dir}/{sbj}/eyeevents/{sess}_rec{rec}_blk{blk}_eyeevent.dat".format(
            dir=prepdir, sbj=sbj, sess=sess, rec=rec, blk=blk)
        fn_eyevex_out = "{dir}/{sess}_rec{rec}_blk{blk}_eyeevent_ordered.dat".format(
            dir=savedir, sbj=sbj, sess=sess, rec=rec, blk=blk)

        # load parameters and eye event data
        data_readers, params, task_events = load_dataset_info(dataset)
        eye_events = load_eyeevent_data(dataset)
        stiminfo = load_stimulus_info(params)
        sacinfo, fixinfo = avutils.get_eyeevent_info(eye_events, stiminfo, task_events, params,
                                                     sampling_rate=params['sampling_rate'],
                                                     pairing=eyeevent_pairing,)
        convert_trial_time_to_clock_count(sacinfo, fixinfo, task_events, params)

        # derive type and order information
        sactypes = avutils.get_sactype(sacinfo, obj_dist_threshold)
        sacorder = np.array(avutils.get_intra_obj_order(sacinfo, fixinfo, sactypes))
        sacorder_rev = np.array(avutils.get_intra_obj_order_rev(sacinfo, fixinfo, sactypes))
        
        # add the derived information to the info dicts
        sacinfo['type'] = sactypes
        sacinfo['order'] = sacorder
        sacinfo['rev_order'] = sacorder_rev
        fixinfo['type'] = np.array([0 if x in (2, 4) else 1 for x in sactypes])
        fixinfo['order'] = sacorder
        fixinfo['rev_order'] = np.array([x if y >= 0 else -1 for x, y in zip(np.roll(sacorder_rev, -1), sacorder_rev)])

        # Generate database for output
        # --- create an empty eye event array
        num_sac = len(sacinfo['on'])
        num_fix = len(fixinfo['on'])
        stimsetname = "/".join([stimdir, params['stimsetname']])
        print("{} saccades, {} fixations".format(num_sac, num_fix))
        print("stimset: {}".format(stimsetname))
        num_eyeevent = num_sac + num_fix
        dtype_eyeevent = [('eventID', int),
                          ('on', long), ('off', long),
                          ('x_on', float), ('y_on', float),
                          ('x_off', float), ('y_off', float),
                          ('param1', float), ('param2', float),
                          ('type', int),
                          ('obj_dist_on', float), ('obj_dist_off', float),
                          ('objID_on', int), ('objID_off', int),
                          ('objpos_x_on', float), ('objpos_y_on', float),
                          ('objpos_x_off', float), ('objpos_y_off', float),
                          ('order', int), ('rev_order', int),
                          ]
        eye_events_typed = np.recarray((num_sac + num_fix,), dtype=dtype_eyeevent)

        # --- fill the array with saccade info
        eye_events_typed['eventID'][:num_sac] = 100
        for key in sacinfo:
            if key in eye_events_typed.dtype.names:
                eye_events_typed[key][:num_sac] = sacinfo[key]

        # --- fill the array with fixation info
        eye_events_typed['eventID'][num_sac:num_eyeevent] = 200
        for key in fixinfo:
            if key in eye_events_typed.dtype.names:
                eye_events_typed[key][num_sac:num_eyeevent] = fixinfo[key]
            elif key in ['obj_dist', 'objID', 'objpos_x', 'objpos_y']:
                value = fixinfo[key]
                eye_events_typed[key+'_on'][num_sac:num_eyeevent] = value
                eye_events_typed[key+'_off'][num_sac:num_eyeevent] = value

        eye_events_typed.sort(order='on')

        # Format outputs
        # --- start with a header line
        output_lines = []
        output_lines.append("\t".join(eye_events_typed.dtype.names))

        # --- add comment lines
        output_lines.append("# original_file: {0}".format(fn_eyevex_in))
        output_lines.append("# subject: {0}".format(sbj))
        output_lines.append("# session: {0}".format(sess))
        output_lines.append("# recording: {0}".format(rec))
        output_lines.append("# block: {0}".format(blk))
        output_lines.append("# stimulus_set: {0}".format(stimsetname))
        output_lines.append("# obj_dist_threshold: {0}".format(obj_dist_threshold))

        # --- add data lines
        for eye_event in eye_events_typed:
            output_lines.append("\t".join(map(str, eye_event)))

        # Save the output lines in file
        with open(fn_eyevex_out, "w") as f:
            f.write("\n".join(output_lines))
        print("data saved in {0}\n".format(fn_eyevex_out))
