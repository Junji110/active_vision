import sys
import os.path

import numpy as np

sys.path.append("/users/ito/toolbox/cili")
from cili.util import load_eyelink_dataset

from prep.parameters.asc2task import *

fields = ("TIMING_CLOCK", "g_task_switch", "g_rec_no", "g_block_num", "TRIAL_NUM", "log_task_ctrl", "t_tgt_data", "SF_FLG")

for sbj, sess, rec, blk, tasktype, taskID in dataset_info:
    fn_data = "{dir}/{task}/{sbj}/{sbj}{rec}00{blk}{sym}.asc".format(
        dir=datadir, task=tasktype, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
    samps, events = load_eyelink_dataset(fn_data)
    msgs = events.dframes['MSG']


    # pick indices of trial starts and ends
    idxs_trial_on = np.where(msgs.label.values == "TRIALID")[0]
    idxs_trial_off = np.where(msgs.label.values == "TRIAL_END")[0]
    assert(len(idxs_trial_on) == len(idxs_trial_off))


    # generate csv
    task_lines = []
    task_lines.append('task')
    task_lines.append(",".join(fields))

    for i_on, i_off in zip(idxs_trial_on, idxs_trial_off):
        trialID = msgs.content.values[i_on]

        # identify stimulus ID
        for i in range(i_on, i_off+1):
            if msgs.label.values[i] == "SYNCTIME":
                v_msg = msgs.content.values[i+1].split()
                sf_flg = 1
                stimID = os.path.splitext(v_msg[2])[0]
                break
        else:
            sf_flg = 0
            stimID = ""

        for i in range(i_on, i_off+1):
            msg_label = msgs.label.values[i]
            if msg_label == "FIXONNN":
                output = map(str, [msgs.index.values[i], taskID, rec, blk, trialID, 301, stimID, sf_flg])
            elif msg_label == "FIXOFFF":
                output = map(str, [msgs.index.values[i], taskID, rec, blk, trialID, 310, stimID, sf_flg])
            elif msg_label == "SYNCTIME":
                output = map(str, [msgs.index.values[i], taskID, rec, blk, trialID, 311, stimID, sf_flg])
            elif msg_label == "ENDTIME":
                output = map(str, [msgs.index.values[i], taskID, rec, blk, trialID, 312, stimID, sf_flg])
            else:
                continue
            task_lines.append(",".join(output))

    fn_task = "{dir}/{task}/{sbj}/{sbj}{rec}00{blk}{sym}_task.csv".format(
        dir=datadir, task=tasktype, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
    with open(fn_task, "w") as f:
        f.write("\n".join(task_lines))
