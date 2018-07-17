import os
import re

import numpy as np
import scipy.io as spio
import h5py

def find_filenames(datadir, subject, session, rec, filetype, pc=None):
    if filetype not in ['imginfo', 'stimtiming', 'param', 'parameter', 'task', 'daq', 'lvd', 'odml', 'hdf5', 'RF']:
        raise ValueError("Filetype {0} is not supported.".format(filetype))

    if filetype in ['daq', 'lvd', 'hdf5', 'odml', 'RF']:
        searchdir = "{dir}/{sbj}/{sess}".format(dir=datadir, sbj=subject, sess=session)
        if filetype in ['RF',]:
            re_filename = re.compile("{0}{1}.*".format(filetype, session))
        elif filetype in ['lvd', 'hdf5'] and pc in ['pc1', 'pc2', 'pc3']:
            re_filename = re.compile('{session}.*_rec{rec}.*{pc}\.{filetype}$'.format(**locals()))
        else:
            re_filename = re.compile('{session}.*_rec{rec}.*\.{filetype}$'.format(**locals()))
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
    block = blockdata['g_block_num']
    trial = blockdata['TRIAL_NUM']

    num_trials = max(trial)
    success = []
    stimID = []
    task = []
    for i_trial in range(num_trials):
        trialID = i_trial + 1
        trialdata = blockdata[blockdata['TRIAL_NUM'] == trialID]
        success.append(trialdata[-1]['SF_FLG'])
        stimID.append(trialdata[0]['t_tgt_data'])
        task.append(trialdata[-1]['g_task_switch'])

    events = np.array(zip(evID, evtime, block, trial), dtype=[('evID', int), ('evtime', long), ('block', int), ('trial', int)])
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
            print "{}: .mat file for {}.png not found".format(stimsetdir, imgID)
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
            print "{}: .mat file for {0}.png not found".format(stimsetdir, imgID)

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

def get_eyeevent_info(eye_events, stiminfo, task_events, param, minlat=0, objdeg=None, objnum=None, pairing=None,
                      reject_eccentric_trials=None, fold=(1, 1), maxfixdur=20000, sampling_rate=1, use_trial_time=True):
    objID, objpos, objsize, bgID, objdeg_stim, objnum_stim = stiminfo

    fixinfo = {'trialID': [], 'imgID': [], 'bgID': [], 'on': [], 'off': [], 'dur': [], 'x': [], 'y': [], 'objID': [], 'obj_dist': [], 'obj_pos_x': [], 'obj_pos_y': [], 'order': []}
    sacinfo = {'trialID': [], 'imgID': [], 'bgID': [], 'on': [], 'off': [], 'dur': [], 'x_on': [], 'y_on': [], 'x_off': [], 'y_off': [], 'amp': [], 'angle': [], 'velo': [], 'accl': [], 'objID_on': [], 'objID_off': [], 'obj_dist_on': [], 'obj_dist_off': [], 'obj_pos_x_on': [], 'obj_pos_y_on': [], 'obj_pos_x_off': [], 'obj_pos_y_off': [], 'order': []}

    if pairing is not None:
        eye_events = extract_fixsac_pair(eye_events, pairing)

    mask_fix = eye_events['eventID'] == 200
    # mask_fix = mask_fix & ((eye_events['off'] - eye_events['on']) < maxfixlen)
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

        # reject trials without center object
        if reject_eccentric_trials is not None:
            if reject_eccentric_trials:
                if np.sqrt((objpos[imgID][0]**2).sum()) >= 1:
                    continue
            else:
                if np.any([np.sqrt((pos**2).sum()) < 1 for pos in objpos[imgID]]):
                    continue

        # skip trials according to fold
        if i_trial % fold[0] != fold[1] - 1:
            continue

        # pick up fixations during the free viewing period
        clkcnt_img_on = taskev_trial['evtime'][taskev_trial['evID'] == 311][0]
        clkcnt_img_off = taskev_trial['evtime'][taskev_trial['evID'] == 312][0]
        minlatlen = int(minlat * sampling_rate)
        if pairing == "sacfix":
            mask_fv = (clkcnt_img_on + minlatlen <= eye_events['on']) & (eye_events['on'] < clkcnt_img_off)
        elif pairing == "fixsac":
            mask_fv = (clkcnt_img_on <= eye_events['on']) & (eye_events['off'] < clkcnt_img_off)
        else:
            mask_fv = (clkcnt_img_on + minlatlen <= eye_events['on']) & (eye_events['off'] < clkcnt_img_off)
        fix_trial = eye_events[mask_fix & mask_fv]

        # skip trials with no fixations
        if len(fix_trial) == 0:
            continue

        # skip trials with fixations longer than 1 sec (indication of drowsiness)
        if maxfixdur is not None:
            maxfixlen = int(maxfixdur * sampling_rate)
            if (fix_trial['off'] - fix_trial['on']).max() > maxfixlen:
                continue

        # store fixation parameters in the buffer
        fixinfo['trialID'].extend([trialID] * len(fix_trial))
        fixinfo['imgID'].extend([imgID] * len(fix_trial))
        fixinfo['bgID'].extend([bgID[imgID]] * len(fix_trial))
        fixinfo['order'].extend(range(len(fix_trial)))
        if use_trial_time:
            fixinfo['on'].extend((fix_trial['on']-clkcnt_img_on) / sampling_rate)
            fixinfo['off'].extend((fix_trial['off']-clkcnt_img_on) / sampling_rate)
        else:
            fixinfo['on'].extend(fix_trial['on'])
            fixinfo['off'].extend(fix_trial['off'])
        fixinfo['dur'].extend((fix_trial['off']-fix_trial['on']) / sampling_rate)

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
        if pairing == "sacfix":
            idx_fix = np.where(mask_fix & mask_fv)[0]
            sac_trial = eye_events[idx_fix - 1]
        elif pairing == "fixsac":
            idx_fix = np.where(mask_fix & mask_fv)[0]
            sac_trial = eye_events[idx_fix + 1]
        else:
            sac_trial = eye_events[mask_sac & mask_fv]

        # store saccade parameters in the buffer
        sacinfo['trialID'].extend([trialID] * len(sac_trial))
        sacinfo['imgID'].extend([imgID] * len(sac_trial))
        sacinfo['bgID'].extend([bgID[imgID]] * len(sac_trial))
        sacinfo['order'].extend(range(len(sac_trial)))
        if use_trial_time:
            sacinfo['on'].extend((sac_trial['on']-clkcnt_img_on) / sampling_rate)
            sacinfo['off'].extend((sac_trial['off']-clkcnt_img_on) / sampling_rate)
        else:
            sacinfo['on'].extend(sac_trial['on'])
            sacinfo['off'].extend(sac_trial['off'])
        sacinfo['dur'].extend((sac_trial['off']-sac_trial['on']) / sampling_rate)
        sacinfo['velo'].extend(sac_trial['param1'])
        sacinfo['accl'].extend(sac_trial['param2'])

        sacpos_on = np.array((sac_trial['x_on'], sac_trial['y_on'])).swapaxes(0, 1)
        sacpos_off = np.array((sac_trial['x_off'], sac_trial['y_off'])).swapaxes(0, 1)
        sacinfo['x_on'].extend(sacpos_on[:, 0])
        sacinfo['y_on'].extend(sacpos_on[:, 1])
        sacinfo['x_off'].extend(sacpos_off[:, 0])
        sacinfo['y_off'].extend(sacpos_off[:, 1])
        x_diff = sacpos_off[:, 0]-sacpos_on[:, 0]
        y_diff = sacpos_off[:, 1]-sacpos_on[:, 1]
        sacinfo['amp'].extend(np.hypot(x_diff, y_diff))
        sacinfo['angle'].extend(np.arctan2(y_diff, x_diff))

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
            return 4  # background-to-background saccade
        elif dist_on > threshold:
            return 3  # background-to-object saccade
        elif dist_off > threshold:
            return 2  # object-to-background saccade
        elif ID_on != ID_off:
            return 1  # trans-object saccade
        else:
            return 0  # intra-object saccade
    return np.array([_sactype(*x) for x in zip(sacinfo['objID_on'], sacinfo['objID_off'], sacinfo['obj_dist_on'], sacinfo['obj_dist_off'])])

def get_intra_obj_order(sacinfo, fixinfo, sactype):
    # identify order of saccades within object
    intra_obj_order = -np.ones_like(sactype, int)
    intra_obj_order[(sactype == 2) | (sactype == 4)] = 0
    intra_obj_order[(sactype == 1) | (sactype == 3)] = 1
    for i in range(len(sactype)-1):
        if fixinfo['off'][i] != sacinfo['on'][i+1]:
            continue
        if intra_obj_order[i] >= 1 and sactype[i+1] == 0:
            intra_obj_order[i+1] = intra_obj_order[i] + 1
    return intra_obj_order

def get_intra_obj_order_rev(sacinfo, fixinfo, sactype):
    # identify order of saccades within object
    sactype_rev = sactype[::-1]
    intra_obj_order_rev = -np.ones_like(sactype_rev, int)
    intra_obj_order_rev[(sactype_rev == 3) | (sactype_rev == 4)] = 0
    intra_obj_order_rev[(sactype_rev == 1) | (sactype_rev == 2)] = 1
    sac_on = sacinfo['on'][-1:0:-1]
    fix_off = fixinfo['off'][-2::-1]
    for i in range(len(sactype_rev)-1):
        if fix_off[i] != sac_on[i]:
            continue
        if intra_obj_order_rev[i] >= 1 and sactype_rev[i+1] == 0:
            intra_obj_order_rev[i+1] = intra_obj_order_rev[i] + 1
    return intra_obj_order_rev[::-1]

def extract_fixsac_pair(eye_events, mode="fixsac"):
    idx_fix_all = np.where(eye_events['eventID'] == 200)[0]
    idx_sac_all = np.where(eye_events['eventID'] == 100)[0]

    if mode == "fixsac":
        idx_sac = np.array([x for x in idx_sac_all if x - 1 in idx_fix_all])
        idx_fix = idx_sac - 1
    elif mode == "sacfix":
        idx_sac = np.array([x for x in idx_sac_all if x + 1 in idx_fix_all])
        idx_fix = idx_sac + 1
    else:
        raise ValueError('mode must be either "fixsac" or "sacfix".')

    idx_all = np.hstack((idx_sac, idx_fix))
    idx_all.sort()
    return eye_events[idx_all]


def identify_artifact_saccades(eye_events, coeffs):
    # idntify artifact saccade based on amp-velo-accl relations
    amp = np.hypot(eye_events['x_off']-eye_events['x_on'], eye_events['y_off']-eye_events['y_on'])
    velo = np.array(eye_events['param1'])
    accl = np.array(eye_events['param2'])

    mask_sac = eye_events['eventID']==100
    mask_amp2velo = velo > coeffs['amp2velo'] * amp
    mask_amp2accl = accl > coeffs['amp2accl'] * amp
    mask_velo2accl = accl > coeffs['velo2accl'] * velo

    # return np.where(mask_sac & mask_amp2accl & mask_velo2accl)[0]
    return np.where(mask_sac & mask_amp2velo & mask_amp2accl & mask_velo2accl)[0]


def remove_artifact_saccades(eye_events, idx_artsac, fixdur_min=0.04, sampling_rate=20000):
    def truncate_fixations(fix_pre, fix_post):
        dur_pre = fix_pre['off'] - fix_pre['on']
        dur_post = fix_post['off'] - fix_post['on']
        fix_trunc = {'eventID': 200}
        for key in fix_pre:
            if key in ('param1', 'param2'):
                fix_trunc[key] = (dur_pre*fix_pre[key] + dur_post*fix_post[key]) / (dur_pre + dur_post)
            elif key in ('off', 'x_off', 'y_off'):
                fix_trunc[key] = fix_post[key]
            elif key in ('on', 'x_on', 'y_on'):
                fix_trunc[key] = fix_pre[key]
        return fix_trunc

    def reconstruct_fixation_from_inter_saccade_interval(sac_pre, sac_post):
        fix_recon = {
            'eventID': 200,
            'on': sac_pre['off'],
            'off': sac_post['on'],
            'x_on': sac_pre['x_off'],
            'y_on': sac_pre['y_off'],
            'x_off': sac_post['x_on'],
            'y_off': sac_post['y_on'],
            'param1': (sac_pre['x_off'] + sac_post['x_on']) / 2,
            'param2': (sac_pre['y_off'] + sac_post['y_on']) / 2,
        }
        return fix_recon

    # convert the input array to a dict of lists, so that deletion of elements in the middle of sequence can be done
    # easily with list.pop()
    events = {key: eye_events[key].tolist() for key in eye_events.dtype.names}

    # as the fixation-truncation algorithm doesn't work for an artifact saccade at the very end of the eye event array,
    # this needs to be removed before applying the algorithm.
    while idx_artsac[-1] == len(events['on']) - 1:
        i = idx_artsac[-1]
        for key in events:
            events[key].pop(i)
        if events['eventID'][i-1] == 200:
            # Since any fixation must be preceded and followed by proper saccades, if the removed artifact saccade is
            # following a fixation, we need to remove this fixation too.
            for key in events:
                events[key].pop(i-1)
        idx_artsac = idx_artsac[:-1]

    # Scan through artifact saccades and truncate the fixations splitted by the artifact saccade.
    for i in idx_artsac[::-1]:
        # The scan is backwards from the last artifact saccade, so that the modification of the event list (removal of
        # the scanned artifact saccade and truncation of the surrounding fixations) doesn't change the positions of the
        # rest of artifact saccades in the list.

        eventID_pre = events['eventID'][i-1]
        eventID_post = events['eventID'][i+1]
        if (eventID_pre == 200) and (eventID_post == 200):
            # Case where the both preceding and following events are fixation:
            #   just truncate these fixations
            fix_pre = {key: events[key][i-1] for key in events}
            fix_post = {key: events[key][i+1] for key in events}
            fix_trunc = truncate_fixations(fix_pre, fix_post)
            for key in events:
                events[key].pop(i+1)
                events[key].pop(i)
                events[key][i-1] = fix_trunc[key]
        elif eventID_post == 200:
            # Case where only the following event is a fixation:
            #   check if the gap to the preceding saccade is shorter than fixdur_min. If it is, it is likely to be cut
            #   out from the following fixation by the current artifact saccade, so truncate the gap to the following
            #   fixation.
            gap = events['on'][i] - events['off'][i-1]
            if gap < fixdur_min * sampling_rate:
                sac_pre = {key: events[key][i-1] for key in events}
                sac_post = {key: events[key][i] for key in events}
                fix_pre = reconstruct_fixation_from_inter_saccade_interval(sac_pre, sac_post)
                fix_post = {key: events[key][i+1] for key in events}
                fix_trunc = truncate_fixations(fix_pre, fix_post)
                for key in events:
                    events[key][i+1] = fix_trunc[key]
                    events[key].pop(i)
            else:
                for key in events:
                    events[key].pop(i+1)
                    events[key].pop(i)
        elif eventID_pre == 200:
            # Case where only the preceding event is a fixation:
            #   check if the gap to the following saccade is shorter than fixdur_min. If it is, truncate it to the
            #   preceding fixation.
            gap = events['on'][i+1] - events['off'][i]
            if gap < fixdur_min * sampling_rate:
                sac_pre = {key: events[key][i] for key in events}
                sac_post = {key: events[key][i+1] for key in events}
                fix_pre = {key: events[key][i-1] for key in events}
                fix_post = reconstruct_fixation_from_inter_saccade_interval(sac_pre, sac_post)
                fix_trunc = truncate_fixations(fix_pre, fix_post)
                for key in events:
                    events[key].pop(i)
                    events[key][i-1] = fix_trunc[key]
            else:
                for key in events:
                    events[key].pop(i)
                    events[key].pop(i-1)

        else:
            # Case where the artifact saccade is not followed and not preceded by a fixation:
            #   just remove the artifact saccade
            for key in events:
                events[key].pop(i)

    if events['eventID'][0] == 200:
        # This case can happen when the very first event in the input eye event array was an artifact saccade. Since
        # any fixation must be preceded and followed by proper saccades, we remove this fixation which is the very
        # first event in the result of the artifact saccade removal.
        for key in events:
            events[key].pop(0)

    eye_events_cleaned = np.empty_like(eye_events)
    eye_events_cleaned = eye_events_cleaned[:len(events['on'])]
    for key in events:
        eye_events_cleaned[key] = events[key]

    return eye_events_cleaned



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

