import os
import sys
import re

import numpy as np
import scipy as sp
import odml.tools.xmlparser as odmlparser

sys.path.append("/users/ito/toolbox")
import active_vision.utils as avutils


class active_vision_io(object):
    def __init__(self, sbj, sess, rec, blk, datadirs):
        self.fn_odml = avutils.find_filenames(datadirs['rawdir'], sbj, sess, rec, 'odml')[0]
        self.fn_task = avutils.find_filenames(datadirs['rawdir'], sbj, sess, rec, 'task')[0]
        self.fn_eyevex = "{}/{}/eyeevents/{}_rec{}_blk{}_eyeevent.dat".format(datadirs['prepdir'], sbj, sess, rec, blk)
        self.params = {'sbj': sbj, 'sess': sess, 'rec': rec, 'blk': blk}
        self.params.update(self.load_odml())
        self.params.update(self.load_calibparam(blk-1))  # eye calibration block is assumed to be 1 block before the free viewing block
        self.stimsetdir = "/".join([datadirs['stimdir'], self.params['stimsetname']])

    def load_odml(self):
        with open(self.fn_odml, 'r') as fd:
            metadata = odmlparser.XMLReader().fromFile(fd)

        prefix = 'blk{0}_'.format(self.params['blk'])

        if metadata['Dataset']['EventData'].properties[prefix+'task_type'].value.data != 3:
            raise ValueError("No free viewing trials in the specified sess/rec/blk.")

        # store all relevant metadata parameters in one dictionary
        params = {
            'sampling_rate':
                metadata['Recording']['HardwareSettings']['DataAcquisition3'].properties['AISampleRate'].value.data,
            'pxlperdeg': metadata['Recording']['HardwareSettings']['Monitor'].properties['PixelPerDegree'].value.data,

            'depth':  metadata['Setup']['Electrode1'].properties['Depth'].value.data,

            'evID': [x.data for x in metadata['Experiment']['Behavior']['Task3'].properties['EventID'].value],
            'evtype': [x.data for x in metadata['Experiment']['Behavior']['Task3'].properties['EventType'].value],
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

        return params

    def load_calibparam(self, blk):
        with open(self.fn_odml, 'r') as fd:
            metadata = odmlparser.XMLReader().fromFile(fd)

        sect = metadata['Dataset']['AnalogData3']
        if sect.find_related(key='CalibParams') is None:
            raise ValueError("Section for CalibParam not found in {0}".format(self.fn_odml))

        params = {
            'calib_sess': self.params['sess'],
            'calib_rec': self.params['rec'],
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
                    params[paramkey] = [x.data for x in prop.value]
                elif paramkey == "Ignore":
                    params[paramkey] = [prop.value.data, ]
                else:
                    params[paramkey] = prop.value.data

        return params

    def get_params(self):
        return self.params

    def get_task_events(self):
        task_events, task_params = avutils.load_task(self.fn_task, int(self.params['blk']))
        self.params.update(task_params)
        return task_events

    def get_eye_events(self):
        return avutils.load_eyevex(self.fn_eyevex)

    def get_stimulus_info(self):
        imgIDs = self.params['stimID2imgID']
        objIDs = {}
        objpos = {}
        objsize = {}
        bgID = {}
        objdeg = {}
        objnum = {}
        for imgID in imgIDs:
            fn_imgmat = "{dir}/{id}.mat".format(dir=self.stimsetdir, id=imgID)
            if os.path.exists(fn_imgmat):
                imginfo = sp.io.loadmat(fn_imgmat, squeeze_me=True, struct_as_record=False)
                info = imginfo['information']
                if np.isnan(info.backgroundid) or np.any([np.isnan(x) for x in info.objectid]):
                    bgID[imgID] = None
                    objIDs[imgID] = None
                    continue
                bgID[imgID] = "{0:03d}".format(info.backgroundid)
                objIDs[imgID] = np.array(info.objectid, int)
                objpos[imgID] = np.array(zip(info.object_x_position, info.object_y_position), int)
                objsize[imgID] = np.array(zip(info.object_x_size, info.object_y_size), int)
                objdeg[imgID] = 2.0
                objnum[imgID] = len(objIDs[imgID])
            else:
                print("{}: .mat file for {}.png not found".format(self.stimsetdir, imgID))
            objpos[imgID][:, 0] = objpos[imgID][:, 0] - stim_size[0] / 2
            objpos[imgID][:, 1] = -objpos[imgID][:, 1] + stim_size[1] / 2
            objpos[imgID] = objpos[imgID] / pxlperdeg
            objsize[imgID] = objsize[imgID] / pxlperdeg
        return objIDs, objpos, objsize, bgID, objdeg, objnum


def convert_eyeevent_info_to_recarray(sacinfo, fixinfo):
    # --- first, create an empty database array
    num_sac = len(sacinfo['on'])
    num_fix = len(fixinfo['on'])
    num_eyeevent = num_sac + num_fix
    dtype_eyeevent = [('eventID', int),
                      ('on', long), ('off', long),
                      ('x_on', float), ('y_on', float),
                      ('x_off', float), ('y_off', float),
                      ('param1', float), ('param2', float),
                      ('type', int),
                      ('obj_dist_on', float), ('obj_dist_off', float),
                      ('objID_on', int), ('objID_off', int),
                      ('obj_pos_x_on', float), ('obj_pos_y_on', float),
                      ('obj_pos_x_off', float), ('obj_pos_y_off', float),
                      ('order', int), ('rev_order', int),
                      ]
    eye_events_arr = np.recarray((num_sac + num_fix,), dtype=dtype_eyeevent)
    # --- second, fill the array with saccade info
    eye_events_arr['eventID'][:num_sac] = 100
    for key in sacinfo:
        if key in eye_events_arr.dtype.names:
            eye_events_arr[key][:num_sac] = sacinfo[key]
        elif key == 'velo':
            eye_events_arr['param1'][:num_sac] = sacinfo[key]
        elif key == 'accl':
            eye_events_arr['param2'][:num_sac] = sacinfo[key]
    # --- third, append the fixation info
    eye_events_arr['eventID'][num_sac:num_eyeevent] = 200
    for key in fixinfo:
        if key in eye_events_arr.dtype.names:
            eye_events_arr[key][num_sac:num_eyeevent] = fixinfo[key]
        elif key in ['x', 'y', 'obj_dist', 'objID', 'obj_pos_x', 'obj_pos_y']:
            value = fixinfo[key]
            eye_events_arr[key+'_on'][num_sac:num_eyeevent] = value
            eye_events_arr[key+'_off'][num_sac:num_eyeevent] = value
            if key == 'x':
                eye_events_arr['param1'][num_sac:num_eyeevent] = value
            if key == 'y':
                eye_events_arr['param2'][num_sac:num_eyeevent] = value
    # --- finally, sort the array in the descending order of event onset time
    eye_events_arr.sort(order='on')

    return eye_events_arr


if __name__ == "__main__":
    from parameters.gen_ordered_eyeevent import *

    for dataset in datasets:
        _, sbj, sess, rec, blk, _, tasktype = dataset
        if tasktype in ('fv_stripes', 'eye_calibration'):
            print("Dataset {}:{}_rec{}_blk{} skipped (tasktype = {})\n".format(
                sbj, sess, rec, blk, tasktype))
            continue

        # load data and metadata
        avio = active_vision_io(sbj, sess, rec, blk, datadirs)
        task_events = avio.get_task_events()
        eye_events = avio.get_eye_events()
        stiminfo = avio.get_stimulus_info()
        params = avio.get_params()

        if artsac_coeffs is not None:
            idx_artsac = avutils.identify_artifact_saccades(eye_events, artsac_coeffs)
            eye_events_orig = eye_events
            eye_events = avutils.remove_artifact_saccades(eye_events_orig, idx_artsac)

        # collect eye event information in relation to the viewed images
        sacinfo, fixinfo = avutils.get_eyeevent_info(eye_events, stiminfo, task_events, params,
                                                     sampling_rate=params['sampling_rate'],
                                                     pairing=pairing_order,
                                                     use_trial_time=False)

        # identify the type of saccades and fixations
        sactypes = avutils.get_sactype(sacinfo, obj_dist_threshold)
        sacinfo['type'] = sactypes
        fixinfo['type'] = np.array([0 if x in (2, 4) else 1 for x in sactypes])

        # identify the order of saccades and fixations
        sacorder = np.array(avutils.get_intra_obj_order(sacinfo, fixinfo, sactypes))
        sacorder_rev = np.array(avutils.get_intra_obj_order_rev(sacinfo, fixinfo, sactypes))
        sacinfo['order'] = sacorder
        sacinfo['rev_order'] = sacorder_rev
        fixinfo['order'] = sacorder
        fixinfo['rev_order'] = np.array([x if y >= 0 else -1 for x, y in zip(np.roll(sacorder_rev, -1), sacorder_rev)])

        # organize the info into a recarray with the same field names as output
        eye_event_array = convert_eyeevent_info_to_recarray(sacinfo, fixinfo)

        # Generate output lines
        # --- start with a header line
        output_lines = []
        output_lines.append("\t".join(eye_event_array.dtype.names))
        # --- add comment lines
        fn_eyevex_in = "{dir}/{sbj}/eyeevents/{sess}_rec{rec}_blk{blk}_eyeevent.dat".format(
            dir=prepdir, sbj=sbj, sess=sess, rec=rec, blk=blk)
        stimsetname = "/".join([stimdir, params['stimsetname']])
        output_lines.append("# original_file: {0}".format(fn_eyevex_in))
        output_lines.append("# subject: {0}".format(sbj))
        output_lines.append("# session: {0}".format(sess))
        output_lines.append("# recording: {0}".format(rec))
        output_lines.append("# block: {0}".format(blk))
        output_lines.append("# stimulus_set: {0}".format(stimsetname))
        output_lines.append("# obj_dist_threshold: {0}".format(obj_dist_threshold))
        output_lines.append("# pairing_order: {0}".format(pairing_order))
        if artsac_coeffs is None:
            output_lines.append("# artifact_saccade_removal: 0")
        else:
            output_lines.append("# artifact_saccade_removal: 1")
            output_lines.append("# coeff_amp2velo: {0}".format(artsac_coeffs['amp2velo']))
            output_lines.append("# coeff_amp2accl: {0}".format(artsac_coeffs['amp2accl']))
            output_lines.append("# coeff_velo2accl: {0}".format(artsac_coeffs['velo2accl']))
        # --- add data lines
        for eye_event in eye_event_array:
            output_lines.append("\t".join(map(str, eye_event)))

        # Save the output lines in file
        fn_eyevex_out = "{dir}/{sess}_rec{rec}_blk{blk}_eyeevent_ordered.dat".format(
            dir=savedir, sbj=sbj, sess=sess, rec=rec, blk=blk)
        with open(fn_eyevex_out, "w") as f:
            f.write("\n".join(output_lines))
        print("Original eyeevent data: {}".format(fn_eyevex_in))
        print("Stimulus set: {}".format(stimsetname))
        if artsac_coeffs is not None:
            print("{} (out of {}) artifact saccades removed".format(len(idx_artsac), (eye_events_orig['eventID']==100).sum()))
        print("{} {} pairs found during trial periods".format(len(sacinfo['on']), pairing_order))
        print("Data saved in {0}".format(fn_eyevex_out))
        print("")

