import os.path
import re

import numpy as np
import scipy.io as sio
import odml.tools.xmlparser as odmlparser

# import parameters
from parameters.gen_typed_eyeevent import *


def find_filenames(datadir, subject, session, rec, filetype):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml', 'hdf5', 'RF']:
        raise ValueError("Filetype {0} is not supported.".format(filetype))

    if filetype in ['daq', 'lvd', 'hdf5', 'odml']:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile('{sess}.*_rec{rec}.*\.{filetype}$'.format(sess=session, rec=rec, filetype=filetype))
    elif filetype in ['RF',]:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        re_filename = re.compile("{0}{1}.*".format(filetype, session))
    else:
        searchdir = "{dir}/{sbj}/{sess}/{sess}_rec{rec}".format(dir=datadir, sbj=subject, sess=session, rec=rec)
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

def load_odml(fn_odml, blk):
    with open(fn_odml, 'r') as fd:
        metadata = odmlparser.XMLReader().fromFile(fd)

    prefix = 'blk{0}_'.format(blk)

    if metadata['Dataset']['EventData'].properties[prefix+'task_type'].value.data != 3:
        raise ValueError("No free viewing trials in the specified sess/rec/blk.")

    # store all relevant metadata parameters in one dictionary
    param = {
        'stimsetname': metadata['Dataset']['StimulusData'].properties[prefix+'setname'].value.data,
    }

    return param

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
        stimID.append(trialdata[0]['t_tgt_data'])

    events = np.array(zip(evID, evtime, trial), dtype=[('evID', int), ('evtime', long), ('trial', int)])
    param = dict(num_trials=num_trials, success=success, stimID=stimID)
    return events, param

def load_eyevex(fn_eyevex):
    convfunc = lambda x: long(x)
    converters = {'on': convfunc, 'off': convfunc}
    eyeevents = np.genfromtxt(fn_eyevex, names=True, dtype=None, converters=converters)
    return eyeevents

def load_imgmat_monkey(stimsetdir, imgIDs):
    objID = {}
    objpos = {}
    objsize = {}
    bgID = {}
    for imgID in imgIDs:
        fn_imgmat = "{dir}/{id}.mat".format(dir=stimsetdir, id=imgID)
        if not os.path.exists(fn_imgmat):
            bgID[imgID] = np.array(np.nan)
            objID[imgID] = np.array(np.nan)
            continue

        imginfo = sio.loadmat(fn_imgmat, squeeze_me=True, struct_as_record=False)
        info = imginfo['information']
        # bgID[imgID] = "{0:03d}".format(info.backgroundid)
        bgID[imgID] = np.array(info.backgroundid)
        objID[imgID] = np.array(info.objectid)
        if objID[imgID].size == 1 and np.isnan(objID[imgID]):
            continue
        else:
            objID[imgID] = objID[imgID].astype(int)
        objpos[imgID] = np.array(zip(info.object_x_position, info.object_y_position), int)
        objsize[imgID] = np.array(zip(info.object_x_size, info.object_y_size), int)

    return objID, objpos, objsize, bgID

def load_imgmat_human(stimsetdir, imgIDs):
    objID = {}
    objpos = {}
    objsize = {}
    bgID = {}
    for imgID in imgIDs:
        fn_imgmat = "{dir}/{id}.mat".format(dir=stimsetdir, id=imgID)
        if not os.path.exists(fn_imgmat):
            bgID[imgID] = np.array((np.nan,))
            objID[imgID] = np.array((np.nan,))
            continue

        imginfo = sio.loadmat(fn_imgmat, squeeze_me=True, struct_as_record=False)
        if tasktype == "Free":
            info = imginfo["information"]
            bgID[imgID] = os.path.splitext(info.BGname)[0]
            objID[imgID] = np.array([int(os.path.splitext(x)[0]) for x in info.OBname])
            if objID[imgID].size == 1 and np.isnan(objID[imgID][0]):
                continue
            objsize[imgID] = np.vstack(info.OBxySize)[:, ::-1]  # object size is stored in the order of (y_size, x_size)
            objpos[imgID] = np.array(zip(info.LUpositionH, info.LUpositionV), int) + objsize[imgID] / 2
        elif tasktype == "Memory":
            info = imginfo["infomation"]
            bgID[imgID] = "{:03d}".format(info.backgroundname)
            objID[imgID] = np.array([int(os.path.splitext(x)[0]) for x in info.objectname])
            if objID[imgID].size == 1 and np.isnan(objID[imgID][0]):
                continue
            objsize[imgID] = np.vstack(info.objectxysize)[:, ::-1]  # object size is stored in the order of (y_size, x_size)
            objpos[imgID] = np.array(info.objectposition, int)[:, ::-1]  # object position is stored in the order of (y_size, x_size)

    return objID, objpos, objsize, bgID

def get_stiminfo(species, stimsetdir, imgIDs, stim_size=None, pxlperdeg=None):
    if not os.path.exists(stimsetdir):
        raise ValueError("Stimulus set directory {} not found.".format(stimsetdir))
    if species == 'Human':
        load_imgmat = load_imgmat_human
    else:
        load_imgmat = load_imgmat_monkey
    objID, objpos, objsize, bgID = load_imgmat(stimsetdir, imgIDs)

    # convert from top-left origin- to center origin coordinate when stim_size is given
    if stim_size is not None:
        for imgID in imgIDs:
            if objID[imgID].size == 1 and np.isnan(objID[imgID]):
                continue
            objpos[imgID][:, 0] = objpos[imgID][:, 0] - stim_size[species][0] / 2
            objpos[imgID][:, 1] = -objpos[imgID][:, 1] + stim_size[species][1] / 2

    # convert from pixel to degree in visual angle when pxlperdeg is given
    if pxlperdeg is not None:
        for imgID in imgIDs:
            if objID[imgID].size == 1 and np.isnan(objID[imgID]):
                continue
            objpos[imgID] = objpos[imgID] / pxlperdeg[species]
            objsize[imgID] = objsize[imgID] / pxlperdeg[species]

    return objID, objpos, objsize, bgID

def add_focus_of_attraction(stiminfo, focus_of_attraction):
    objID, objpos, objsize, bgID = stiminfo
    imgIDs = objID.keys()
    for imgID in imgIDs:
        if focus_of_attraction[bgID[imgID]] is None:
            continue
        foas = np.array(focus_of_attraction[bgID[imgID]])
        num_foas = foas.shape[0]
        objpos[imgID] = np.vstack((objpos[imgID], np.array(foas)))
        objID[imgID] = np.hstack((objID[imgID], np.arange(-1, -num_foas-1, -1)))

def discard_artifact_saccade(eye_events, A, B, plot=False):
    idx_fix_all = np.where(eye_events['eventID'] == 200)[0]
    idx_sac_all = np.where(eye_events['eventID'] == 100)[0]
    amp = np.sqrt((eye_events['x_off'] - eye_events['x_on']) ** 2 + (eye_events['y_off'] - eye_events['y_on']) ** 2)
    velo = eye_events['param1']
    idx_sac = np.array([x for x in idx_sac_all if velo[x] < A * amp[x] + B])
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(amp[idx_sac_all], velo[idx_sac_all], 'r.')
        plt.plot(amp[idx_sac], velo[idx_sac], 'b.')
        x = np.linspace(0, 30, 1000)
        plt.plot(x, A * x + B)
        plt.show()
    idx_all = np.hstack((idx_sac, idx_fix_all))
    idx_all.sort()
    return eye_events[idx_all]

def extract_fixsac_pair(eye_events):
    idx_fix_all = np.where(eye_events['eventID'] == 200)[0]
    idx_sac_all = np.where(eye_events['eventID'] == 100)[0]
    idx_sac = np.array([x for x in idx_sac_all if x - 1 in idx_fix_all])
    idx_fix = idx_sac - 1
    idx_all = np.hstack((idx_sac, idx_fix))
    idx_all.sort()
    return eye_events[idx_all]

def get_fixinfo(eye_events, stiminfo, task_events, param):
    objID, objpos, objsize, bgID = stiminfo
    mask_fix = eye_events['eventID'] == 200

    fixinfo = {'trialID': [], 'imgID': [], 'bgID': [], 'objID': [], 'objpos_x': [], 'objpos_y': [],
               'obj_dist': [], 'type': [], 'on': [], 'off': [], 'x_on': [],
               'y_on': [], 'x_off': [], 'y_off': [], 'param1': [], 'param2': []}

    for i_trial in range(param['num_trials']):
        # reject failure trials
        if param['success'][i_trial] <= 0:
            continue

        # pick up fixations during free viewing
        trialID = i_trial + 1
        imgID = param['stimID'][trialID-1]
        taskev_trial = task_events[task_events['trial'] == trialID]

        # reject trials with missing image-onset or offset events
        if (taskev_trial['evID'] != 311).all() or (taskev_trial['evID'] != 312).all():
            continue

        # reject trials with stimuli without embedded objects
        if objID[imgID].size == 1 and np.isnan(objID[imgID]):
            continue

        clkcnt_img_on = taskev_trial['evtime'][taskev_trial['evID'] == 311][0]
        clkcnt_img_off = taskev_trial['evtime'][taskev_trial['evID'] == 312][0]
        mask_fv = (clkcnt_img_on <= eye_events['on']) & (eye_events['off'] < clkcnt_img_off)
        fix_trial = eye_events[mask_fix & mask_fv]
        fixpos = np.array((fix_trial['param1'], fix_trial['param2'])).swapaxes(0, 1)
        obj_dist_all = np.hypot(*np.rollaxis(fixpos[:, None, :] - objpos[imgID][None, :, :], -1))
        idx_fixobj = obj_dist_all.argmin(axis=1)

        # store parameters in the buffer
        fixinfo['trialID'].extend([trialID] * len(fix_trial))
        fixinfo['imgID'].extend([imgID] * len(fix_trial))
        fixinfo['bgID'].extend([bgID[imgID]] * len(fix_trial))
        fixinfo['objID'].extend(objID[imgID][idx_fixobj])
        fixinfo['objpos_x'].extend(objpos[imgID][idx_fixobj, 0])
        fixinfo['objpos_y'].extend(objpos[imgID][idx_fixobj, 1])
        fixinfo['obj_dist'].extend(obj_dist_all.min(axis=1))
        fixinfo['on'].extend(fix_trial['on'])
        fixinfo['off'].extend(fix_trial['off'])
        fixinfo['x_on'].extend(fix_trial['x_on'])
        fixinfo['y_on'].extend(fix_trial['y_on'])
        fixinfo['x_off'].extend(fix_trial['x_off'])
        fixinfo['y_off'].extend(fix_trial['y_off'])
        fixinfo['param1'].extend(fix_trial['param1'])
        fixinfo['param2'].extend(fix_trial['param2'])
    obj_dist = fixinfo['obj_dist']
    fixinfo['type'].extend([0 if x > obj_dist_threshold else 1 for x in obj_dist])

    return fixinfo

def get_sacinfo(eye_events, stiminfo, task_events, param):
    objID, objpos, objsize, bgID = stiminfo
    mask_sac = eye_events['eventID'] == 100

    sacinfo = {'trialID': [], 'imgID': [], 'bgID': [], 'objID_on': [],
               'objpos_x_on': [], 'objpos_y_on': [], 'objpos_x_off': [], 'objpos_y_off': [],
               'objID_off': [], 'obj_dist_on': [], 'obj_dist_off': [],
               'type': [], 'on': [], 'off': [], 'x_on': [], 'y_on': [],
               'x_off': [], 'y_off': [], 'param1': [], 'param2': []}

    for i_trial in range(param['num_trials']):
        # reject failure trials
        if param['success'][i_trial] <= 0:
            continue

        # pick up saccades during free viewing
        trialID = i_trial + 1
        imgID = param['stimID'][trialID-1]
        taskev_trial = task_events[task_events['trial'] == trialID]

        # reject trials with missing image-onset or offset events
        if (taskev_trial['evID'] != 311).all() or (taskev_trial['evID'] != 312).all():
            continue

        # reject trials with stimuli without embedded objects
        if objID[imgID].size == 1 and np.isnan(objID[imgID]):
            continue

        clkcnt_img_on = taskev_trial['evtime'][taskev_trial['evID'] == 311][0]
        clkcnt_img_off = taskev_trial['evtime'][taskev_trial['evID'] == 312][0]
        mask_fv = (clkcnt_img_on <= eye_events['on']) & (eye_events['off'] < clkcnt_img_off)
        sac_trial = eye_events[mask_sac & mask_fv]
        sacpos_on = np.array((sac_trial['x_on'], sac_trial['y_on'])).swapaxes(0, 1)
        obj_dist_on_all = np.hypot(*np.rollaxis(sacpos_on[:, None, :] - objpos[imgID][None, :, :], -1))
        idx_sacobj_on = obj_dist_on_all.argmin(axis=1)
        sacpos_off = np.array((sac_trial['x_off'], sac_trial['y_off'])).swapaxes(0, 1)
        obj_dist_off_all = np.hypot(*np.rollaxis(sacpos_off[:, None, :] - objpos[imgID][None, :, :], -1))
        idx_sacobj_off = obj_dist_off_all.argmin(axis=1)

        # store parameters in the buffer
        sacinfo['trialID'].extend([trialID] * len(sac_trial))
        sacinfo['imgID'].extend([imgID] * len(sac_trial))
        sacinfo['bgID'].extend([bgID[imgID]] * len(sac_trial))
        sacinfo['objID_on'].extend(objID[imgID][idx_sacobj_on])
        sacinfo['objID_off'].extend(objID[imgID][idx_sacobj_off])
        sacinfo['objpos_x_on'].extend(objpos[imgID][idx_sacobj_on, 0])
        sacinfo['objpos_y_on'].extend(objpos[imgID][idx_sacobj_on, 1])
        sacinfo['objpos_x_off'].extend(objpos[imgID][idx_sacobj_off, 0])
        sacinfo['objpos_y_off'].extend(objpos[imgID][idx_sacobj_off, 1])
        sacinfo['obj_dist_on'].extend(obj_dist_on_all.min(axis=1))
        sacinfo['obj_dist_off'].extend(obj_dist_off_all.min(axis=1))
        sacinfo['on'].extend(sac_trial['on'])
        sacinfo['off'].extend(sac_trial['off'])
        sacinfo['x_on'].extend(sac_trial['x_on'])
        sacinfo['y_on'].extend(sac_trial['y_on'])
        sacinfo['x_off'].extend(sac_trial['x_off'])
        sacinfo['y_off'].extend(sac_trial['y_off'])
        sacinfo['param1'].extend(sac_trial['param1'])
        sacinfo['param2'].extend(sac_trial['param2'])
    objID_on = sacinfo['objID_on']
    objID_off = sacinfo['objID_off']
    obj_dist_on = sacinfo['obj_dist_on']
    obj_dist_off = sacinfo['obj_dist_off']
    sacinfo['type'].extend([get_sactype(x, obj_dist_threshold) for x in zip(objID_on, objID_off, obj_dist_on, obj_dist_off)])

    return sacinfo

def get_sactype(params, threshold):
    objID_on, objID_off, obj_dist_on, obj_dist_off = params
    if obj_dist_on > threshold and obj_dist_off > threshold:
        return 4
    elif obj_dist_on > threshold:
        return 3
    elif obj_dist_off > threshold:
        return 2
    elif objID_on != objID_off:
        return 1
    else:
        return 0



#   =============================
#            MAIN ROUTINE
#   =============================
for species, sbj, sess, rec, blk, stimsetname, tasktype in datasets:
    if species != selected_species:
        continue
    if sbj not in selected_subjects:
        continue

    dataset_name = "{0}:{1}_rec{2}_blk{3}".format(sbj, sess, rec, blk)
    stim_size_deg = np.array(stim_size[species]) / float(pxlperdeg[species])
    stim_extent = (-stim_size_deg[0]/2, stim_size_deg[0]/2, -stim_size_deg[1]/2, stim_size_deg[1]/2)

    # set species specific variables
    if species == "Human":
        fn_task = "{dir}/{task}/{sbj}/{sbj}{rec}00{blk}{sym}_task.csv".format(
            dir=rawdir[species], task=tasktype, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
        fn_eyevex_in = "{dir}/eyeevents/{sbj}{rec}00{blk}{sym}_eyeevent.dat".format(dir=prepdir[species], sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
        fn_eyevex_out = "{dir}/{sbj}{rec}00{blk}{sym}_eyeevent_typed.dat".format(dir=savedir, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
    else:
        fn_task = find_filenames(rawdir[species], sbj, sess, rec, 'task')[0]
        fn_eyevex_in = "{dir}/{sbj}/eyeevents/{sess}_rec{rec}_blk{blk}_eyeevent.dat".format(dir=prepdir[species], sbj=sbj, sess=sess, rec=rec, blk=blk)
        fn_eyevex_out = "{dir}/{sess}_rec{rec}_blk{blk}_eyeevent_typed.dat".format(dir=savedir, sbj=sbj, sess=sess, rec=rec, blk=blk)

    # =====
    # ===== load data from files
    # =====
    # load parameters from odML file
    if species == "Monkey" and stimsetname is None:
        fn_odml = find_filenames(rawdir[species], sbj, sess, rec, 'odml')[0]
        param = load_odml(fn_odml, blk)
        stimsetname = param['stimsetname']
    stimsetdir = "{0}/{1}".format(stimdir, stimsetname)

    # load task events and parameters from task file
    task_events, param = load_task(fn_task, blk)
    imgIDs = set(param['stimID'])

    # load object information form stimulus files
    stiminfo = get_stiminfo(species, stimsetdir, imgIDs, stim_size, pxlperdeg)

    # load eye events from eyevex data file
    eye_events = load_eyevex(fn_eyevex_in)

    # collect fixation info and saccade info regarding objects
    fixinfo = get_fixinfo(eye_events, stiminfo, task_events, param)
    sacinfo = get_sacinfo(eye_events, stiminfo, task_events, param)

    # create an empty eye event array
    num_sac = len(sacinfo['type'])
    num_fix = len(fixinfo['type'])
    print "{} saccades, {} fixations".format(num_sac, num_fix)
    print "stimset: {}".format(stimsetdir)
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
                      ]
    eye_events_typed = np.recarray((num_sac + num_fix,), dtype=dtype_eyeevent)

    # fill saccade info
    eye_events_typed['eventID'][:num_sac] = 100
    for key in sacinfo:
        if key in eye_events_typed.dtype.names:
            eye_events_typed[key][:num_sac] = sacinfo[key]

    # fill fixation info
    eye_events_typed['eventID'][num_sac:num_eyeevent] = 200
    for key in fixinfo:
        if key in eye_events_typed.dtype.names:
            eye_events_typed[key][num_sac:num_eyeevent] = fixinfo[key]
        elif key in ['obj_dist', 'objID', 'objpos_x', 'objpos_y']:
            value = fixinfo[key]
            eye_events_typed[key+'_on'][num_sac:num_eyeevent] = value
            eye_events_typed[key+'_off'][num_sac:num_eyeevent] = value

    eye_events_typed.sort(order='on')

    # generate output lines
    output_lines = []

    # add header line
    output_lines.append("\t".join(eye_events_typed.dtype.names))

    # add comment lines
    output_lines.append("# original_file: {0}".format(fn_eyevex_in))
    output_lines.append("# subject: {0}".format(sbj))
    output_lines.append("# session: {0}".format(sess))
    output_lines.append("# recording: {0}".format(rec))
    output_lines.append("# block: {0}".format(blk))
    output_lines.append("# stimulus_set: {0}".format(stimsetname))
    output_lines.append("# obj_dist_threshold: {0}".format(obj_dist_threshold))

    # add data lines
    for eye_event in eye_events_typed:
        output_lines.append("\t".join(map(str, eye_event)))

    # write to output file
    with open(fn_eyevex_out, "w") as f:
        f.write("\n".join(output_lines))
    print "data saved in {0}\n".format(fn_eyevex_out)


