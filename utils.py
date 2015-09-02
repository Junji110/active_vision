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

