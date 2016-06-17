import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt

from fileio import lvdread
from analysis import butterworth_filter
import utils as avutils


def extract_threshold_crossings(data, threshold):
    mask = data < threshold
    if mask.sum() == 0:
        return np.array([]), np.array([])
    mask_diff = np.diff(mask.astype(np.int))
    idxs_on = np.where(mask_diff == 1)[0]
    idxs_off = np.where(mask_diff == -1)[0]
    if idxs_off[0] < idxs_on[0]:
        idxs_off = idxs_off[1:]
    min_len = np.min((len(idxs_on), len(idxs_off)))
    idxs_on = idxs_on[:min_len]
    idxs_off = idxs_off[:min_len] + 1
    return idxs_on, idxs_off

def extract_threshold_crossing_peaks(data, threshold):
    idxs_on, idxs_off = extract_threshold_crossings(data, threshold)
    idxs_peak = []
    for idx_on, idx_off in zip(idxs_on, idxs_off):
        idxs_peak.append(data[idx_on:idx_off].argmin() + idx_on)
    return np.array(idxs_peak, np.int)

def get_binned_event_rates(idxs_event, idxs_bin):
    rates = []
    for idx_ini, idx_fin in zip(*idxs_bin):
        idxs_event_in_bin = idxs_event[(idx_ini <= idxs_event) & (idxs_event < idx_fin)]
        rates.append(len(idxs_event_in_bin) / bin_width)
    return np.array(rates)

def get_binned_event_magnitude_means(data, idxs_event, idxs_bin):
    means = []
    for idx_ini, idx_fin in zip(*idxs_bin):
        idxs_event_in_bin = idxs_event[(idx_ini <= idxs_event) & (idxs_event < idx_fin)]
        means.append(data[idxs_event_in_bin].mean())
    return np.array(means)

def detect_event_magnitude_change(data, idxs_event, idxs_bin):
    pvals = []
    for idx_ini, idx_fin in zip(*idxs_bin):
        idx_mid = (idx_fin + idx_ini) / 2
        idxs_event_in_bin1 = idxs_event[(idx_ini <= idxs_event) & (idxs_event < idx_mid)]
        idxs_event_in_bin2 = idxs_event[(idx_mid <= idxs_event) & (idxs_event < idx_fin)]
        if len(idxs_event_in_bin1) * len(idxs_event_in_bin2) == 0:
            pvals.append(np.nan)
        else:
            D, pval = spstats.ks_2samp(data[idxs_event_in_bin1], data[idxs_event_in_bin2])
            pvals.append(pval)
    return np.array(pvals)

def detect_event_interval_change(idxs_event, idxs_bin):
    intervals = np.diff(idxs_event)
    idxs_interval = (idxs_event[:-1] + idxs_event[1:]) / 2
    pvals = []
    for idx_ini, idx_fin in zip(*idxs_bin):
        idx_mid = (idx_fin + idx_ini) / 2
        intervals_in_bin1 = intervals[(idx_ini <= idxs_interval) & (idxs_interval < idx_mid)]
        intervals_in_bin2 = intervals[(idx_mid <= idxs_interval) & (idxs_interval < idx_fin)]
        if len(intervals_in_bin1) * len(intervals_in_bin2) == 0:
            pvals.append(np.nan)
        else:
            D, pval = spstats.ks_2samp(intervals_in_bin1, intervals_in_bin2)
            pvals.append(pval)
    return np.array(pvals)


if __name__ == "__main__":
    # file information
    datasetdir = "z:"
    # datasetdir = "/users/junji/desktop/ito/datasets/osaka"
    rawdir = "{}/RAWDATA".format(datasetdir)
    prepdir = "{}/PREPROCESSED".format(datasetdir)
    savedir = "."

    # analysis parameters
    hpfreq = 500.0
    lpfreq = None
    threshold_factor = -4.5
    bin_width = 50  # bin width in sec
    bin_step = 1  # bin step in sec
    significance_level = 0.01

    # execution parameters
    # savefig = True
    savefig = False

    use_median_based_std = True
    # use_median_based_std = False

    # session information
    datasets = [
        ["HIME", "20140908", 4, 5, "pc1"],
        ["HIME", "20140908", 4, 5, "pc2"],
        ["SATSUKI", "20150811", 6, 2, "pc1"],
        ["SATSUKI", "20150811", 6, 2, "pc2"],
    ]


    for sbj, sess, rec, blk, pc in datasets:
        # --- set filenames
        fn_task_all = avutils.find_filenames(rawdir, sbj, sess, rec, 'task')
        fn_task = fn_task_all[0]
        fn_data_all = avutils.find_filenames(rawdir, sbj, sess, rec, 'lvd')
        fn_data = [x for x in fn_data_all if pc in x][0]
        fn_fig = "{savedir}/{sbj}/MUA_summary/{sess}_rec{rec}_blk{blk}.png".format(**locals())

        # load experimental event markers from the task file
        task_events, task_params = avutils.load_task(fn_task, blk)
        idx_blk_on = task_events['evtime'][0]
        idx_blk_off = task_events['evtime'][-1]
        idxs_trial_on = task_events['evtime'][task_events['evID'] == 311] - idx_blk_on
        idxs_trial_off = task_events['evtime'][task_events['evID'] == 312] - idx_blk_on
        num_trial = min((len(idxs_trial_on), len(idxs_trial_off)))

        # load experimental parameters from the raw data file
        reader = lvdread.LVDReader(fn_data)
        lvdparam = reader.get_param()
        lvdheader = reader.get_header()
        fs = lvdparam['sampling_rate']
        channels = lvdheader['AIUsedChannelName'][:-1]

        # define bins for time resolved analysis
        bin_length = np.int(bin_width * fs)
        bin_shift = np.int(bin_step * fs)
        idxs_bin_ini = np.arange(0, idx_blk_off - idx_blk_on - bin_length, bin_shift).astype(np.int)
        idxs_bin_fin = idxs_bin_ini + bin_length
        times_bin = (idxs_bin_fin + idxs_bin_ini) / 2. / fs

        # define bins for change detection
        bin_length = np.int(bin_width * 2 * fs)
        bin_shift = np.int(bin_step * fs)
        idxs_cdbin_ini = np.arange(0, idx_blk_off - idx_blk_on - bin_length, bin_shift).astype(np.int)
        idxs_cdbin_fin = idxs_cdbin_ini + bin_length
        times_cdbin = (idxs_cdbin_fin + idxs_cdbin_ini) / 2. / fs

        # setup figure axes
        fig = plt.figure(figsize=(30, 17))
        fig_title = "{sbj}_{sess}_rec{rec}_blk{blk}_{pc} (bin width {bin_width:.1f} s, bin step {bin_step:.1f} s, alpha = {significance_level:.3f})".format(**locals())
        fig.suptitle(fig_title)
        # fig.subplots_adjust(left=0.04, right=0.98, bottom=0.04, top=0.99, hspace=0.3)
        # gs = gridspec.GridSpec(10, 1)
        # ax_ch = fig.add_subplot(gs[0:9, :])
        # ax_alert = fig.add_subplot(gs[9:10, :], sharex=ax_ch)
        ax_alert = plt.axes([0.03, 0.04, 0.95, 0.05])
        ax_ch = plt.axes([0.03, 0.11, 0.95, 0.86])

        alert_level_peak_height = np.zeros(len(times_cdbin))
        alert_level_peak_interval = np.zeros(len(times_cdbin))
        print "\n{0}".format(fig_title)
        for i_ch, channel in enumerate(channels):
            # if i_ch != 1: continue  # for a quick run

            # load single channel raw data
            rawdata = reader.get_data(channel=channel, samplerange=(idx_blk_on, idx_blk_off))

            print "{0} data loading done.".format(channel)

            # filter the raw data and set the threshold for spike detection
            filtdata = butterworth_filter(rawdata[0], fs, hpfreq, lpfreq)
            if use_median_based_std:
                filtdata_std = np.median(np.abs(filtdata) / 0.6745)  # c.f. Quiroga et al. 2004
            else:
                filtdata_std = filtdata.std()
            threshold = threshold_factor * filtdata_std

            # extract spike times
            idxs_peak = extract_threshold_crossing_peaks(filtdata, threshold)

            # compute time-resolved measures
            peak_rates = get_binned_event_rates(idxs_peak, (idxs_bin_ini, idxs_bin_fin))
            peak_height_means = get_binned_event_magnitude_means(filtdata, idxs_peak, (idxs_bin_ini, idxs_bin_fin))

            # change point detection
            pvals_peak_height_change = detect_event_magnitude_change(filtdata, idxs_peak, (idxs_cdbin_ini, idxs_cdbin_fin))
            alert_level_peak_height[pvals_peak_height_change < significance_level] += 1
            sm_peak_height_change = np.log10((1 - pvals_peak_height_change) / pvals_peak_height_change)

            pvals_peak_interval_change = detect_event_interval_change(idxs_peak, (idxs_cdbin_ini, idxs_cdbin_fin))
            alert_level_peak_interval[pvals_peak_interval_change < significance_level] += 1
            sm_peak_interval_change = np.log10((1 - pvals_peak_interval_change) / pvals_peak_interval_change)

            print "{0} data processing done.".format(channel)

            # plot single channel data
            ax_ch.plot(idxs_peak/fs, (filtdata[idxs_peak] - threshold) / (filtdata.min() - threshold) - i_ch, 'k,')
            ax_ch.plot(times_bin, (peak_height_means - threshold) / (filtdata.min() - threshold) - i_ch, c='g', alpha=0.5)
            ax_ch.plot(times_bin, peak_rates / peak_rates.max() - i_ch, c='r', alpha=0.5)
            # ax_ch.plot(times_cdbin, sm_peak_height_change / 5 - i_ch, c='g', alpha=0.5)
            # ax_ch.plot(times_cdbin, sm_peak_interval_change / 5 - i_ch, c='r', alpha=0.5)

            # draw change detected periods
            idxs_peak_height_change_ini, idxs_peak_height_change_fin = extract_threshold_crossings(pvals_peak_height_change, significance_level)
            for idx_ini, idx_fin in zip(idxs_peak_height_change_ini, idxs_peak_height_change_fin):
                t_ini = times_cdbin[idx_ini]
                t_fin = times_cdbin[idx_fin]
                y_min = 1.0/len(channels)*(len(channels)-i_ch-1)
                y_max = 1.0/len(channels)*(len(channels)-i_ch)
                ax_ch.axvspan(t_ini, t_fin, y_min, y_max, color='g', alpha=0.1)

            idxs_peak_interval_change_ini, idxs_peak_interval_change_fin = extract_threshold_crossings(pvals_peak_interval_change, significance_level)
            for idx_ini, idx_fin in zip(idxs_peak_interval_change_ini, idxs_peak_interval_change_fin):
                t_ini = times_cdbin[idx_ini]
                t_fin = times_cdbin[idx_fin]
                y_min = 1.0/len(channels)*(len(channels)-i_ch-1)
                y_max = 1.0/len(channels)*(len(channels)-i_ch)
                ax_ch.axvspan(t_ini, t_fin, y_min, y_max, color='r', alpha=0.1)

            print "{0} data plot done.".format(channel)

        # plot the alert level
        ax_alert.plot(times_cdbin, alert_level_peak_height, c='g')
        ax_alert.plot(times_cdbin, alert_level_peak_interval, c='r')
        ax_alert.plot(times_cdbin, alert_level_peak_interval + alert_level_peak_height, c='k')

        # draw trial markers
        for i_trial in range(num_trial):
            idx_ini = idxs_trial_on[i_trial]
            idx_fin = idxs_trial_off[i_trial]
            ax_alert.axvspan(idx_ini/fs, idx_fin/fs, color='k', alpha=0.1)

        # put axes labels
        ax_ch.set_yticks(np.arange(-23, 1, 1))
        ax_ch.set_yticklabels(lvdheader['AIUsedChannelName'][23::-1])
        ax_ch.set_ylim(-23, 1)
        ax_ch.set_ylabel("Channel")
        ax_ch.grid(c='gray', ls='-')
        ax_alert.set_xlabel("Time (s)")
        ax_alert.set_ylabel("Alert level")
        ax_alert.grid(c='gray', ls='-')

        if savefig:
            plt.savefig(fn_fig)
            print "Figure saved as {}".format(fn_fig)
            plt.clf()
        else:
            plt.show()
