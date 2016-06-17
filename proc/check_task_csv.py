import os.path
import re

import numpy as np

# import parameters
from parameters.check_task_csv import *


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


def check_task_csv(fn_task, blk):
    convfunc = lambda x: long(x)
    converters = {'INTERVAL': convfunc, 'TIMING_CLOCK': convfunc, 'GL_TIMER_VAL': convfunc}
    taskdata = np.genfromtxt(fn_task, skip_header=1, delimiter=',', names=True, dtype=None, converters=converters)
    blockdata = taskdata[taskdata['g_block_num'] == blk]
    trial = blockdata['TRIAL_NUM']

    print "\nCheck {}".format(fn_task)

    num_trials = max(trial)
    for i_trial in range(num_trials):
        trialID = i_trial + 1
        trialdata = blockdata[blockdata['TRIAL_NUM'] == trialID]
        if trialdata[0]['t_tgt_data'] != trialdata[-1]['t_tgt_data']:
            print "Block {}, trial {}: stimulus ID mismatch".format(blk, trialID), trialdata['t_tgt_data']
        if 12 in trialdata['log_task_ctrl']:
            print "Block {}, trial {}: event ID 12 occurred".format(blk, trialID), trialdata['log_task_ctrl']


#   =============================
#            MAIN ROUTINE
#   =============================
for species, sbj, sess, rec, blk, stimsetname, tasktype in datasets:
    if species != selected_species:
        continue
    if sbj not in selected_subjects:
        continue

    # set species specific variables
    if species == "Human":
        fn_task = "{dir}/{task}/{sbj}/{sbj}{rec}00{blk}{sym}_task.csv".format(
            dir=rawdir[species], task=tasktype, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
    else:
        fn_task = find_filenames(rawdir[species], sbj, sess, rec, 'task')[0]

    # load task events and parameters from task file
    check_task_csv(fn_task, blk)
