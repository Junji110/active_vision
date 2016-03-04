import os
import re

import numpy as np
import scipy.io as spio
import h5py

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

def get_imgID(stimdir, stimsetname):
    imgIDs = []
    # for i in range(1, 61):
    i_img = 1
    while True:
        fn_img = "{0}/{1}/{2}.mat".format(stimdir, stimsetname, i_img)
        if os.path.exists(fn_img):
            img_mat = spio.loadmat(fn_img, struct_as_record=False, squeeze_me=True)
            imgIDs.append(img_mat['information'].backgroundid)
            i_img += 1
        else:
            break
    return imgIDs

def get_objID(stimdir, stimsetname):
    objIDs = []
    # for i in range(1, 61):
    i_img = 1
    while True:
        fn_img = "{0}/{1}/{2}.mat".format(stimdir, stimsetname, i_img)
        if os.path.exists(fn_img):
            img_mat = spio.loadmat(fn_img, struct_as_record=False, squeeze_me=True)
            objIDs.extend(img_mat['information'].objectid)
            i_img += 1
        else:
            break
    return sorted(list(set(objIDs)))

def load_task(fn_task, blk=0):
    convfunc = lambda x: long(x)
    converters = {'INTERVAL': convfunc, 'TIMING_CLOCK': convfunc, 'GL_TIMER_VAL': convfunc}
    taskdata = np.genfromtxt(fn_task, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
    if blk == 0:
        blockdata = taskdata
    else:
        blockdata = taskdata[taskdata['g_block_num'] == blk]

    evID = blockdata['log_task_ctrl']
    evtime = blockdata['TIMING_CLOCK']
    trial = blockdata['TRIAL_NUM']

    num_trials = max(trial)
    success = []
    stimID = []
    task = []
    for i_trial in range(num_trials):
        trialID = i_trial + 1
        trialdata = blockdata[blockdata['TRIAL_NUM'] == trialID]
        success.append(trialdata[-1]['SF_FLG'])
        stimID.append(trialdata[-1]['t_tgt_data'])
        task.append(trialdata[-1]['g_task_switch'])

    events = np.array(zip(evID, evtime, trial), dtype=[('evID', int), ('evtime', long), ('trial', int)])
    param = dict(num_trials=num_trials, success=success, stimID=stimID, task=task)
    return events, param

def load_eyevex(fn_eyevex):
    convfunc = lambda x: long(x)
    converters = {'on': convfunc, 'off': convfunc}
    eyeevents = np.genfromtxt(fn_eyevex, names=True, dtype=None, converters=converters)
    return eyeevents

def load_imgh5_human(stimsetdir, imgIDs):
    objID = {}
    objpos = {}
    objsize = {}
    bgID = {}
    objdeg = {}
    objnum = {}
    for imgID in imgIDs:
        fn_imgh5 = "{dir}/{id}.h5".format(dir=stimsetdir, id=imgID)
        with h5py.File(fn_imgh5, 'r') as f:
            info = f['information']
            # bgID[imgID] = "{0:03d}".format(info['backgroundid'][...])
            bgID[imgID] = info['backgroundid'][...]
            objID[imgID] = info['objectid'][...]
            if objID[imgID].size == 1 and np.isnan(objID):
                continue
            objpos[imgID] = np.array(zip(info['object_x_position'][...], info['object_y_position'][...]))
            objsize[imgID] = np.array(zip(info['object_x_size'][...], info['object_y_size'][...]))
            objdeg[imgID] = info['object_deg_size'][...]
            objnum[imgID] = info['objectnumber'][...]
    return objID, objpos, objsize, bgID, objdeg, objnum

def load_imgmat_monkey(stimsetdir, imgIDs, tasktype="Free"):
    objID = {}
    objpos = {}
    objsize = {}
    bgID = {}
    objdeg = {}
    objnum = {}
    for imgID in imgIDs:
        fn_imgmat = "{dir}/{id}.mat".format(dir=stimsetdir, id=imgID)
        if os.path.exists(fn_imgmat):
            imginfo = spio.loadmat(fn_imgmat, squeeze_me=True, struct_as_record=False)
            info = imginfo['information']
            bgID[imgID] = "{0:03d}".format(info.backgroundid)
            objID[imgID] = np.array(info.objectid, int)
            if objID[imgID].size == 1 and np.isnan(objID):
                continue
            objpos[imgID] = np.array(zip(info.object_x_position, info.object_y_position), int)
            objsize[imgID] = np.array(zip(info.object_x_size, info.object_y_size), int)
            objdeg[imgID] = 2.0
            objnum[imgID] = len(objID[imgID])
        else:
            print ".mat file for {}.png not found".format(imgID)
    return objID, objpos, objsize, bgID, objdeg, objnum

def load_imgmat_human(stimsetdir, imgIDs, tasktype="Free"):
    objID = {}
    objpos = {}
    objsize = {}
    bgID = {}
    objdeg = {}
    objnum = {}
    for imgID in imgIDs:
        fn_imgmat = "{dir}/{id}.mat".format(dir=stimsetdir, id=imgID)
        if not os.path.exists(fn_imgmat):
            print ".mat file for {0}.png not found".format(imgID)

        imginfo = spio.loadmat(fn_imgmat, squeeze_me=True, struct_as_record=False)
        if tasktype == "Free":
            info = imginfo["information"]
            bgID[imgID] = os.path.splitext(info.BGname)[0]
            objID[imgID] = np.array([int(os.path.splitext(x)[0]) for x in info.OBname])
            if objID[imgID].size == 1 and np.isnan(objID):
                continue
            objsize[imgID] = np.vstack(info.OBxySize)[:, ::-1]  # object size is stored in the order of (y_size, x_size)
            objpos[imgID] = np.array(zip(info.LUpositionH, info.LUpositionV), int) + objsize[imgID] / 2
            objdeg[imgID] = 2.0
            objnum[imgID] = len(objID[imgID])
        elif tasktype == "Memory":
            info = imginfo["infomation"]
            bgID[imgID] = "{:03d}".format(info.backgroundname)
            objID[imgID] = np.array([int(os.path.splitext(x)[0]) for x in info.objectname])
            if objID[imgID].size == 1 and np.isnan(objID):
                continue
            objsize[imgID] = np.vstack(info.objectxysize)[:, ::-1]  # object size is stored in the order of (y_size, x_size)
            objpos[imgID] = np.array(info.objectposition, int)[:, ::-1]  # object position is stored in the order of (y_size, x_size)
            objdeg[imgID] = info.objectsize
            objnum[imgID] = info.objectnumber

    return objID, objpos, objsize, bgID, objdeg, objnum

def get_stiminfo(species, stimsetdir, imgIDs, tasktype, stim_size=None, pxlperdeg=None):
    if species == 'Human':
        load_imgmat = load_imgmat_human
    else:
        load_imgmat = load_imgmat_monkey
    objID, objpos, objsize, bgID, objdeg, objnum = load_imgmat(stimsetdir, imgIDs, tasktype)

    if stim_size is not None:
        for imgID in imgIDs:
            objpos[imgID][:, 0] = objpos[imgID][:, 0] - stim_size[species][0] / 2
            objpos[imgID][:, 1] = -objpos[imgID][:, 1] + stim_size[species][1] / 2

    if pxlperdeg is not None:
        for imgID in imgIDs:
            objpos[imgID] = objpos[imgID] / pxlperdeg[species]
            objsize[imgID] = objsize[imgID] / pxlperdeg[species]

    return objID, objpos, objsize, bgID, objdeg, objnum

def get_eyeevent_info(eye_events, stiminfo, task_events, param, minlat=0, objdeg=None, objnum=None, pairing=None):
    objID, objpos, objsize, bgID, objdeg_stim, objnum_stim = stiminfo

    fixinfo = {'trialID': [], 'imgID': [], 'bgID': [], 'on': [], 'off': [], 'dur': [], 'x': [], 'y': [], 'objID': [], 'obj_dist': [], 'obj_pos_x': [], 'obj_pos_y': []}
    sacinfo = {'trialID': [], 'imgID': [], 'bgID': [], 'on': [], 'off': [], 'dur': [], 'x_on': [], 'y_on': [], 'x_off': [], 'y_off': [], 'amp': [], 'velo': [], 'accl': [], 'objID_on': [], 'objID_off': [], 'obj_dist_on': [], 'obj_dist_off': [], 'obj_pos_x_on': [], 'obj_pos_y_on': [], 'obj_pos_x_off': [], 'obj_pos_y_off': []}

    mask_fix = eye_events['eventID'] == 200
    mask_sac = eye_events['eventID'] == 100

    for i_trial in range(param['num_trials']):
        # reject failure trials
        if param['success'][i_trial] <= 0:
            continue

        trialID = i_trial + 1
        imgID = param['stimID'][trialID-1]
        taskev_trial = task_events[task_events['trial'] == trialID]

        # reject trials with non-specified object size and/or object number
        if objdeg is not None and objdeg_stim[imgID] != objdeg:
            continue
        if objnum is not None and objnum_stim[imgID] != objnum:
            continue

        # reject trials with missing image-onset or offset events
        if (taskev_trial['evID'] != 311).all() or (taskev_trial['evID'] != 312).all():
            continue

        # pick up fixations during the free viewing period
        clkcnt_img_on = taskev_trial['evtime'][taskev_trial['evID'] == 311][0]
        clkcnt_img_off = taskev_trial['evtime'][taskev_trial['evID'] == 312][0]
        mask_fv = (clkcnt_img_on + minlat <= eye_events['on']) & (eye_events['on'] < clkcnt_img_off)
        fix_trial = eye_events[mask_fix & mask_fv]

        # store fixation parameters in the buffer
        fixinfo['trialID'].extend([trialID] * len(fix_trial))
        fixinfo['imgID'].extend([imgID] * len(fix_trial))
        fixinfo['bgID'].extend([bgID[imgID]] * len(fix_trial))
        fixinfo['on'].extend(fix_trial['on'] - clkcnt_img_on)
        fixinfo['off'].extend(fix_trial['off'] - clkcnt_img_on)
        fixinfo['dur'].extend(fix_trial['off'] - fix_trial['on'])

        fixpos = np.array((fix_trial['param1'], fix_trial['param2'])).swapaxes(0, 1)
        fixinfo['x'].extend(fixpos[:, 0])
        fixinfo['y'].extend(fixpos[:, 1])

        obj_dist_all = np.hypot(*np.rollaxis(fixpos[:, None, :] - objpos[imgID][None, :, :], -1))
        fixinfo['obj_dist'].extend(obj_dist_all.min(axis=1))
        argmin = obj_dist_all.argmin(axis=1)
        fixinfo['objID'].extend(objID[imgID][argmin])
        fixinfo['obj_pos_x'].extend(objpos[imgID][argmin, 0])
        fixinfo['obj_pos_y'].extend(objpos[imgID][argmin, 1])

        # pick up saccades during the free viewing period
        if pairing is None:
            sac_trial = eye_events[mask_sac & mask_fv]
        else:
            idx_fix = np.where(mask_fix & mask_fv)[0]
            mask_sac_paired = np.zeros_like(eye_events, bool)
            if pairing == "sacfix":
                mask_sac_paired[idx_fix - 1] = True
            elif pairing == "fixsac":
                mask_sac_paired[idx_fix + 1] = True
            else:
                raise ValueError("pairing must be either 'sacfix' or 'fixsac'")
            sac_trial = eye_events[mask_sac_paired & mask_fv]

        # store saccade parameters in the buffer
        sacinfo['trialID'].extend([trialID] * len(sac_trial))
        sacinfo['imgID'].extend([imgID] * len(sac_trial))
        sacinfo['bgID'].extend([bgID[imgID]] * len(sac_trial))
        sacinfo['on'].extend(sac_trial['on'] - clkcnt_img_on)
        sacinfo['off'].extend(sac_trial['off'] - clkcnt_img_on)
        sacinfo['dur'].extend(sac_trial['off'] - sac_trial['on'])
        sacinfo['velo'].extend(sac_trial['param1'])
        sacinfo['accl'].extend(sac_trial['param2'])

        sacpos_on = np.array((sac_trial['x_on'], sac_trial['y_on'])).swapaxes(0, 1)
        sacpos_off = np.array((sac_trial['x_off'], sac_trial['y_off'])).swapaxes(0, 1)
        sacinfo['x_on'].extend(sacpos_on[:, 0])
        sacinfo['y_on'].extend(sacpos_on[:, 1])
        sacinfo['x_off'].extend(sacpos_off[:, 0])
        sacinfo['y_off'].extend(sacpos_off[:, 1])
        sacinfo['amp'].extend(np.hypot(sacpos_off[:, 0]-sacpos_on[:, 0], sacpos_off[:, 1]-sacpos_on[:, 1]))

        obj_dist_on_all = np.hypot(*np.rollaxis(sacpos_on[:, None, :] - objpos[imgID][None, :, :], -1))
        sacinfo['obj_dist_on'].extend(obj_dist_on_all.min(axis=1))
        argmin_on = obj_dist_on_all.argmin(axis=1)
        sacinfo['objID_on'].extend(objID[imgID][argmin_on])
        sacinfo['obj_pos_x_on'].extend(objpos[imgID][argmin_on, 0])
        sacinfo['obj_pos_y_on'].extend(objpos[imgID][argmin_on, 1])

        obj_dist_off_all = np.hypot(*np.rollaxis(sacpos_off[:, None, :] - objpos[imgID][None, :, :], -1))
        sacinfo['obj_dist_off'].extend(obj_dist_off_all.min(axis=1))
        argmin_off = obj_dist_off_all.argmin(axis=1)
        sacinfo['objID_off'].extend(objID[imgID][argmin_off])
        sacinfo['obj_pos_x_off'].extend(objpos[imgID][argmin_off, 0])
        sacinfo['obj_pos_y_off'].extend(objpos[imgID][argmin_off, 1])

    return sacinfo, fixinfo

def get_sactype(sacinfo, threshold):
    def _sactype(ID_on, ID_off, dist_on, dist_off):
        if dist_on > threshold and dist_off > threshold:
            return 4
        elif dist_on > threshold:
            return 3
        elif dist_off > threshold:
            return 2
        elif ID_on != ID_off:
            return 1
        else:
            return 0
    return np.array([_sactype(*x) for x in zip(sacinfo['objID_on'], sacinfo['objID_off'], sacinfo['obj_dist_on'], sacinfo['obj_dist_off'])])


if __name__ == "__main__":
    stimdir = "C:/Users/ito/datasets/osaka/stimuli"
    stimsetnames = (
        # "fv_gray_20141110",
        # "fv_gray_20141111",
        # "fv_gray_20141117",
        "fv_scene_20141125",
    )
    for stimsetname in stimsetnames:
        imgIDs = get_imgID(stimdir, stimsetname)
        oris = [-1] * len(imgIDs)
        print '    "{0}": {{'.format(stimsetname)
        print '        "bgID": ('
        print '            {0}'.format(', '.join([str(x) for x in imgIDs]))
        print '        ),'
        print '        "orientation": ('
        print '            {0}'.format(', '.join([str(x) for x in oris]))
        print '        )'
        print '    },'

        # print stimsetname
        # print get_imgID(stimdir, stimsetname)
        # print get_objID(stimdir, stimsetname)
        # print

