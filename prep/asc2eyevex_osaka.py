import sys
import os.path

import numpy as np

sys.path.append("/users/ito/toolbox/cili")
from cili.util import load_eyelink_dataset

from parameters.asc2eyevex import *

fields = ("TIMING_CLOCK", "g_task_switch", "g_rec_no", "g_block_num", "TRIAL_NUM", "log_task_ctrl", "t_tgt_data", "SF_FLG")

for sbj, sess, rec, blk, tasktype, taskID in dataset_info:
    fn_data = "{dir}/{task}/{sbj}/{sbj}{rec}00{blk}{sym}.asc".format(
        dir=datadir, task=tasktype, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
    samps, events = load_eyelink_dataset(fn_data)
    saccades = events.dframes['ESACC']
    fixations = events.dframes['EFIX']
    blinks = events.dframes['EBLINK']

    # retrieve screen size from gaze coordinates in MSG events
    for i_msg, msg in events.dframes['MSG'].iterrows():
        if msg.label == "GAZE_COORDS":
            gaze_coords = map(float, msg.content.split())
            break
    screen_size = (gaze_coords[2] - gaze_coords[0], gaze_coords[3] - gaze_coords[1])

    # NOTE! this may be a bug of cili:
    # `esacc.last_onset` is NOT the onset of the current event, but the offset.
    # The onset is stored as the index of `esacc`. The same holds for `efix`
    # and `eblink`.
    blinkdata = []
    for i_blink, blink in blinks.iterrows():
        idx_on = i_blink
        idx_off = i_blink + blink.duration
        blinkdata.append((idx_on, idx_off))
    blinkdata = np.array(blinkdata, dtype=[('on', long), ('off', long)])

    sacdata = []
    for i_sac, sac in saccades.iterrows():
        idx_on = i_sac
        idx_off = i_sac + sac.duration
        # reject saccades that contain a blink
        if np.any((idx_on <= blinkdata['on']) & (blinkdata['off'] <= idx_off)):
            continue
        # EYELINK seems to erroneously register an saccade at the beginning of
        # recording. Such saccades have almost zero amplitude and duration of
        # one sample. They are rejected here.
        if sac.x_start - sac.x_end < 10 and sac.y_start - sac.y_end < 10 and sac.duration <= 4:
            continue
        sacdata.append((idx_on, idx_off, sac.x_start, sac.y_start, sac.x_end, sac.y_end, sac.peak_velocity, 0.0))
    sacdata = np.array(sacdata, dtype=[('on', long), ('off', long), ('x_on', float), ('y_on', float), ('x_off', float), ('y_off', float), ('param1', float), ('param2', float)])

    fixdata = []
    for i_fix, fix in fixations.iterrows():
        idx_on = i_fix
        idx_off = i_fix + fix.duration
        fixdata.append((idx_on, idx_off, fix.x_pos, fix.y_pos, fix.x_pos, fix.y_pos, fix.x_pos, fix.y_pos))
    fixdata = np.array(fixdata, dtype=[('on', long), ('off', long), ('x_on', float), ('y_on', float), ('x_off', float), ('y_off', float), ('param1', float), ('param2', float)])

    # change the coordinate from (left, top, right, bottom) =
    # (0, 0, screen_size_x, screen_size_y) (i.e., top-left origin) to
    # (-screen_size_x/2, screen_size_y/2, screen_size_x/2, -screen_size_y/2)
    # (i.e., center origin), and convert units from pixel to degree
    pxl2deg_x = lambda x: (x - screen_size[0] / 2) / pxlperdeg
    pxl2deg_y = lambda y: (-y + screen_size[1] / 2) / pxlperdeg
    sacdata['x_on'] = pxl2deg_x(sacdata['x_on'])
    sacdata['y_on'] = pxl2deg_y(sacdata['y_on'])
    sacdata['x_off'] = pxl2deg_x(sacdata['x_off'])
    sacdata['y_off'] = pxl2deg_y(sacdata['y_off'])
    fixdata['x_on'] = pxl2deg_x(fixdata['x_on'])
    fixdata['y_on'] = pxl2deg_y(fixdata['y_on'])
    fixdata['x_off'] = pxl2deg_x(fixdata['x_off'])
    fixdata['y_off'] = pxl2deg_y(fixdata['y_off'])
    fixdata['param1'] = pxl2deg_x(fixdata['param1'])
    fixdata['param2'] = pxl2deg_y(fixdata['param2'])

    eye_event_data = np.append(sacdata, fixdata)
    eye_event_ID = np.array([100] * len(sacdata) + [200] * len(fixdata))
    idx_sort = eye_event_data['on'].argsort()
    eye_event_data = eye_event_data[idx_sort]
    eye_event_ID = eye_event_ID[idx_sort]

    # generate csv
    eyevex_lines = []
    fields = ["eventID",]
    fields.extend(eye_event_data.dtype.names)
    eyevex_lines.append("\t".join(fields))

    for evID, evdata in zip(eye_event_ID, eye_event_data):
        output = [evID,]
        output.extend(evdata)
        eyevex_lines.append("\t".join(map(str, output)))

    fn_eyevex = "{dir}/eyeevents/{sbj}{rec}00{blk}{sym}_eyeevent.dat".format(
        dir=prepdir, sbj=sbj, rec=rec, blk=blk, sym=tasktype[0])
    with open(fn_eyevex, "w") as f:
        f.write("\n".join(eyevex_lines))
