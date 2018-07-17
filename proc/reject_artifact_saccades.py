import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append("/users/ito/toolbox")
import active_vision.utils as avutils


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

        # Monitor eye events in a specific time range (for debugging)
        # if 367.3*20000 < eye_events[i]['on'] < 367.4*20000:
        #     print("eye event at i-1: {}, {}-{}".format(events['eventID'][i-1], events['on'][i-1]/20000., events['off'][i-1]/20000.))
        #     print("eye event at i: {}, {}-{}".format(events['eventID'][i], events['on'][i]/20000., events['off'][i]/20000.))
        #     print("eye event at i+1: {}, {}-{}".format(events['eventID'][i+1], events['on'][i+1]/20000., events['off'][i+1]/20000.))
        #     print("")

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
    monkey = "HIME"
    # monkey = "SATSUKI"
    # datadir = "/users/ito/datasets/osaka/PROCESSED/{}/eyeevents_ordered".format(monkey)
    datadir = "/users/ito/datasets/osaka/PREPROCESSED/{}/eyeevents".format(monkey)

    # coeffs = {'amp2accl': 25000., 'velo2accl': 250.}
    coeffs = {'amp2velo': 110., 'amp2accl': 25000., 'velo2accl': 200.}

    for fn in sorted(os.listdir(datadir)):
        eye_events = avutils.load_eyevex(datadir+"/"+fn)

        # artifact saccade removal
        idx_artsac = identify_artifact_saccades(eye_events, coeffs)
        print("{}, {} artifact saccades".format(fn, len(idx_artsac)))
        print("{} events, first artsac at {}, last artsac at {}".format(len(eye_events), idx_artsac[0], idx_artsac[-1]))
        print("")
        eye_events_cleaned = remove_artifact_saccades(eye_events, idx_artsac)

        # print(eye_events[:30])
        # print("")
        # print(eye_events_cleaned[:30])
        # print("")

        mask_sac = eye_events['eventID']==100
        mask_artsac = np.zeros_like(mask_sac)
        mask_artsac[idx_artsac] = True
        mask_realsac = mask_sac & np.invert(mask_artsac)
        amps = np.hypot(eye_events['x_off']-eye_events['x_on'], eye_events['y_off']-eye_events['y_on'])

        # visualize the amp-velo-accl relations of the original data in a 3D plot
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(111, projection='3d')
        # ax1.set_title(fn)
        # ax1.set_xlabel("Saccade amplitude (deg)")
        # ax1.set_ylabel("Peak saccade velocity (deg/s)")
        # ax1.set_zlabel("Peak saccade acceleration (deg/s2)")
        # ax1.scatter(amps[mask_realsac], eye_events['param1'][mask_realsac], eye_events['param2'][mask_realsac], s=1, color='b')
        # ax1.scatter(amps[mask_artsac], eye_events['param1'][mask_artsac], eye_events['param2'][mask_artsac], s=1, color='r')
        # fig1.tight_layout()
        # plt.show()

        # visualize the amp-velo-accl relations in 2D plots
        fig2 = plt.figure()

        ax2_1 = fig2.add_subplot(221)
        ax2_1.set_xlabel("Saccade amplitude (deg)")
        ax2_1.set_ylabel("Sac. velo. (deg/s)")
        # ax2_1.plot(amps[mask_sac], eye_events['param1'][mask_sac], 'k,')
        ax2_1.plot(amps[mask_realsac], eye_events['param1'][mask_realsac], 'b,')
        ax2_1.plot(amps[mask_artsac], eye_events['param1'][mask_artsac], 'r,')
        x = np.linspace(0, 30, 1000)
        ax2_1.plot(x, coeffs['amp2velo'] * x, ls='--', color='gray')
        ax2_1.set_xscale('log')
        ax2_1.set_yscale('log')
        ax2_1.set_xlim(amps[mask_sac].min(), amps[mask_sac].max())
        ax2_1.set_ylim(eye_events['param1'][mask_sac].min(), eye_events['param1'][mask_sac].max())
        ax2_1.grid()

        ax2_2 = fig2.add_subplot(223)
        ax2_2.set_xlabel("Saccade amplitude (deg)")
        ax2_2.set_ylabel("Sac. accel. (deg/s2)")
        # ax2_2.plot(amps[mask_sac], eye_events['param2'][mask_sac], 'k,')
        ax2_2.plot(amps[mask_realsac], eye_events['param2'][mask_realsac], 'b,')
        ax2_2.plot(amps[mask_artsac], eye_events['param2'][mask_artsac], 'r,')
        x = np.linspace(0, 30, 1000)
        ax2_2.plot(x, coeffs['amp2accl'] * x, ls='--', color='gray')
        ax2_2.set_xscale('log')
        ax2_2.set_yscale('log')
        ax2_2.set_xlim(amps[mask_sac].min(), amps[mask_sac].max())
        ax2_2.set_ylim(eye_events['param2'][mask_sac].min(), eye_events['param2'][mask_sac].max())
        ax2_2.grid()

        ax2_3 = fig2.add_subplot(222)
        ax2_3.set_xlabel("Sac. velo. (deg/s)")
        ax2_3.set_ylabel("Sac. accel. (deg/s2)")
        # ax2_3.plot(eye_events['param1'][mask_sac], eye_events['param2'][mask_sac], 'k,')
        ax2_3.plot(eye_events['param1'][mask_realsac], eye_events['param2'][mask_realsac], 'b,')
        ax2_3.plot(eye_events['param1'][mask_artsac], eye_events['param2'][mask_artsac], 'r,')
        x = np.linspace(0, 1500, 1000)
        ax2_3.plot(x, coeffs['velo2accl'] * x, ls='--', color='gray')
        ax2_3.set_xscale('log')
        ax2_3.set_yscale('log')
        ax2_3.set_xlim(eye_events['param1'][mask_sac].min(), eye_events['param1'][mask_sac].max())
        ax2_3.set_ylim(eye_events['param2'][mask_sac].min(), eye_events['param2'][mask_sac].max())
        ax2_3.grid()

        fig2.tight_layout()
        plt.show()

        # visualize the distribution of gaps (non-fixation inter saccade intervals) around artifact saccades
        # cases = {'no_gap': 0, 'gap_pre': 0, 'gap_post': 0, 'isolated': 0}
        # gaps = {'pre': [], 'post': []}
        # for i in idx_artsac:
        #     eventID_pre = eye_events['eventID'][i - 1]
        #     eventID_post = eye_events['eventID'][i + 1]
        #     if (eventID_pre == 200) and (eventID_post == 200):
        #         cases['no_gap'] += 1
        #     elif eventID_pre == 200:
        #         cases['gap_post'] += 1
        #         gaps['post'].append((eye_events['on'][i+1] - eye_events['off'][i])/20000.)
        #     elif eventID_post == 200:
        #         cases['gap_pre'] += 1
        #         gaps['pre'].append((eye_events['on'][i] - eye_events['off'][i-1])/20000.)
        #     else:
        #         cases['isolated'] += 1
        #         gaps['post'].append((eye_events['on'][i+1] - eye_events['off'][i])/20000.)
        #         gaps['pre'].append((eye_events['on'][i] - eye_events['off'][i-1])/20000.)
        # print(cases)
        # plt.hist(gaps['pre'], bins=100, range=(0, 1), alpha=0.5, label="gap_pre")
        # plt.hist(gaps['post'], bins=100, range=(0, 1), alpha=0.5, label="gap_post")
        # plt.legend()
        # plt.show()

        # visualize the result as time series of saccade and fixation periods as well as fixation positions
        # plt.subplot(311)
        # for i in range(1000):
        #     plt.plot((eye_events_cleaned[i]['on']/20000., eye_events_cleaned[i]['off']/20000.),
        #              (eye_events_cleaned[i]['eventID'], eye_events_cleaned[i]['eventID']), color='r', lw=5)
        # for i in range(2000):
        #     if i in idx_artsac:
        #         color='magenta'
        #     else:
        #         color='black'
        #     plt.plot((eye_events[i]['on']/20000., eye_events[i]['off']/20000.),
        #              (eye_events[i]['eventID'], eye_events[i]['eventID']), color=color)
        # plt.grid()
        #
        # plt.subplot(312, sharex=plt.gca())
        # for i in range(1000):
        #     if eye_events_cleaned['eventID'][i] == 100:
        #         continue
        #     plt.plot((eye_events_cleaned[i]['on']/20000., eye_events_cleaned[i]['off']/20000.),
        #              (eye_events_cleaned[i]['x_on'], eye_events_cleaned[i]['x_off']), color='r', lw=5)
        #     plt.plot((eye_events_cleaned[i]['on']+eye_events_cleaned[i]['off'])/2/20000.,
        #              eye_events_cleaned[i]['param1'], 'rx')
        # for i in range(2000):
        #     if eye_events['eventID'][i] == 100:
        #         continue
        #     plt.plot((eye_events[i]['on']/20000., eye_events[i]['off']/20000.),
        #              (eye_events[i]['x_on'], eye_events[i]['x_off']), color='k')
        #     plt.plot((eye_events[i]['on']+eye_events[i]['off'])/2/20000.,
        #              eye_events[i]['param1'], 'k.')
        # plt.grid()
        #
        # plt.subplot(313, sharex=plt.gca())
        # for i in range(1000):
        #     if eye_events_cleaned['eventID'][i] == 100:
        #         continue
        #     plt.plot((eye_events_cleaned[i]['on']/20000., eye_events_cleaned[i]['off']/20000.),
        #              (eye_events_cleaned[i]['y_on'], eye_events_cleaned[i]['y_off']), color='r', lw=5)
        #     plt.plot((eye_events_cleaned[i]['on']+eye_events_cleaned[i]['off'])/2/20000.,
        #              eye_events_cleaned[i]['param2'], 'rx')
        # for i in range(2000):
        #     if eye_events['eventID'][i] == 100:
        #         continue
        #     plt.plot((eye_events[i]['on']/20000., eye_events[i]['off']/20000.),
        #              (eye_events[i]['y_on'], eye_events[i]['y_off']), color='k')
        #     plt.plot((eye_events[i]['on']+eye_events[i]['off'])/2/20000.,
        #              eye_events[i]['param2'], 'k.')
        # plt.grid()
        #
        # plt.show()

        # visualize the result as fixation duration histograms
        # dur = np.array(eye_events['off'] - eye_events['on']) / 20000.
        # mask_fix = eye_events['eventID']==200
        # print("{} fixations before artifact removal".format(mask_fix.sum()))
        # plt.subplot(121)
        # plt.title("Before artifact removal")
        # plt.hist(dur[mask_fix], bins=200, range=[0, 2], alpha=0.5)
        # dur = np.array(eye_events_cleaned['off'] - eye_events_cleaned['on']) / 20000.
        # mask_fix = eye_events_cleaned['eventID']==200
        # print("{} fixations after artifact removal".format(mask_fix.sum()))
        # plt.subplot(122, sharey=plt.gca())
        # plt.title("after artifact removal")
        # plt.hist(dur[mask_fix], bins=200, range=[0, 2], alpha=0.5)
        # plt.show()

