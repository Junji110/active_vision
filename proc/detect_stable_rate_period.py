import numpy as np
import scipy
import sklearn.cluster
import scipy.spatial.distance
import scipy.stats as spstats
import matplotlib.pyplot as plt
from matplotlib import gridspec

import utils


def gap(data, refs=None, nrefs=10, ks=range(1, 11)):
    """
    # gap.py
    # (c) 2013 Mikael Vejdemo-Johansson
    # BSD License
    #
    # SciPy function to compute the gap statistic for evaluating k-means clustering.
    # Gap statistic defined in
    # Tibshirani, Walther, Hastie:
    #  Estimating the number of clusters in a data set via the gap statistic
    #  J. R. Statist. Soc. B (2001) 63, Part 2, pp 411-423

    Compute the Gap statistic for an nxm dataset in data.

    Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
    or state the number k of reference distributions in nrefs for automatic generation with a
    uniformed distribution within the bounding box of data.

    Give the list of k-values for which you want to compute the statistic in ks.
    """
    dst = scipy.spatial.distance.euclidean
    shape = data.shape
    if refs == None:
        tops = data.max(axis=0)
        bots = data.min(axis=0)
        dists = scipy.matrix(scipy.diag(tops-bots))

        rands = scipy.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i]*dists + bots
    else:
        rands = refs

    gaps = scipy.zeros((len(ks),))
    for (i, k) in enumerate(ks):
        kmc, kml, _, = sklearn.cluster.k_means(data, k, n_init=1)
        disp = sum([dst(data[m, :], kmc[kml[m], :]) for m in range(shape[0])])

        refdisps = scipy.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            kmc, kml, _, = sklearn.cluster.k_means(rands[:, :, j], k, n_init=1)
            refdisps[j] = sum([dst(rands[m, :, j], kmc[kml[m], :]) for m in range(shape[0])])
        gaps[i] = scipy.mean(scipy.log(refdisps)) - scipy.log(disp)
    return gaps


def segment_successive_occurrences(data, value):
    # detect the edges of successive occurrences of `value` in `data`
    data_bin = (data == value).astype(int)
    idx_inc = np.where(np.diff(data_bin) == 1)[0]
    idx_dec = np.where(np.diff(data_bin) == -1)[0]

    # special cases where the edges are too few and so the following boundary treatment doesn't work
    if len(idx_dec) == 0 and len(idx_inc) == 0:
        return None
    elif len(idx_dec) == 1 and len(idx_inc) == 0:
        idx_inc = np.array([-1,])
    elif idx_dec.size == 0 and idx_inc.size == 1:
        idx_dec = np.array([len(data) - 1,])

    # treatment of the boundaries in order to assure that idx_inc and idx_dec are of the same size and idx_inc[i] always
    # precedes idx_dec[i] for any i
    if idx_dec[0] < idx_inc[0]:
        idx_inc = np.hstack(([-1,], idx_inc))
    if idx_dec[-1] < idx_inc[-1]:
        idx_dec = np.hstack((idx_dec, [len(data) - 1,]))
    return np.array(zip(idx_inc + 1, idx_dec + 1))


if __name__ == "__main__":
    # file information
    # datasetdir = "z:"
    # datasetdir = "/users/junji/desktop/ito/datasets/osaka"
    datasetdir = "/home/ito/datasets/osaka"
    rawdir = "{}/RAWDATA".format(datasetdir)
    prepdir = "{}/PREPROCESSED".format(datasetdir)
    savedir = "."

    # analysis parameters
    sampling_rate = 20000.0
    bin_size = 15  # bin width in number of trials
    bin_step = 1  # bin step in number of trials
    num_bin_hist = 40
    timebin_size = 50.0  # time-based bin width in seconds
    timebin_step = 5.0  # time-based bin step in seconds
    fdr_q = 0.05
    rpv_threshold = 1.0  # refractory period voilation threshold in ms

    # plot parameters
    colors_task = ["white", "blue", "yellow", "green"]

    # execution parameters
    # estimate_nums_clst = True
    estimate_nums_clst = False

    # savefig = True
    savefig = False

    # session information
    datasets = [
        # ["SATSUKI", "20151027", 5, 2, "V1", ""],
        # ["SATSUKI", "20151027", 5, 2, "V1", "Demerged"],
        # ["SATSUKI", "20151027", 5, 2, "IT", ""],
        # ["SATSUKI", "20151027", 5, 2, "IT", "Demerged"],
        # ["SATSUKI", "20151110", 7, 2, "V1", ""],
        # ["SATSUKI", "20151110", 7, 2, "V1", "Demerged"],
        # ["SATSUKI", "20151110", 7, 2, "IT", ""],
        ["SATSUKI", "20151110", 7, 2, "IT", "Demerged"],
    ]

    for dataset in datasets:
        sbj, sess, rec, blk, site, cluster_type = dataset
        fn_spikes = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
        print "\n{sbj}:{fn_spikes} ({cluster_type})".format(**locals())

        # set filenames
        # fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        fn_class = "{dir}/tmp/new/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        fn_task = utils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]

        # load task events
        print "\tLoading task data file..."
        task_events, task_param = utils.load_task(fn_task)
        blks = np.unique(task_events['block'])
        ts_blk_on = []
        ts_blk_off = []
        tasks_blk = []
        for b in blks:
            blk_events, blk_param = utils.load_task(fn_task, b)
            task_blk = blk_param['task'][0]
            tasks_blk.append(task_blk)
            ts_blk_on.append(blk_events["evtime"][0] / sampling_rate)
            ts_blk_off.append(blk_events["evtime"][-2] / sampling_rate)
            if b == blk:
                evID_img_on = task_blk*100 + 11
                evID_img_off = task_blk*100 + 12
                ts_img_on = []
                ts_img_off = []
                for i_trial in range(task_param["num_trials"]):
                    # reject failure trials
                    if task_param['success'][i_trial] <= 0:
                        continue

                    trialID = i_trial + 1
                    trial_events = blk_events[blk_events['trial'] == trialID]

                    # reject trials with missing image-onset or offset events
                    if (trial_events['evID'] != evID_img_on).all() or (trial_events['evID'] != evID_img_off).all():
                        continue

                    ts_img_on.append(trial_events['evtime'][trial_events['evID'] == evID_img_on][0] / sampling_rate)
                    ts_img_off.append(trial_events['evtime'][trial_events['evID'] == evID_img_off][0] / sampling_rate)
                ts_img_on = np.array(ts_img_on)
                ts_img_off = np.array(ts_img_off)
                assert(len(ts_img_on) == len(ts_img_off))
                num_trial = len(ts_img_on)
        recdur = ts_blk_off[-1]
        print "\t...done.\n"

        # load data
        print "\tLoading spike data file..."
        dataset = np.genfromtxt(fn_class, skip_header=2, dtype=None, names=True)
        dataset["event_time"] += ts_blk_on[blk]
        if cluster_type == "Demerged":
            num_ch = len(dataset.dtype.names) - 4
        else:
            num_ch = len(dataset.dtype.names) - 3
        print "\t...done.\n"

        mask_blk = (ts_blk_on[blk] < dataset["event_time"]) & (dataset["event_time"] < ts_blk_off[blk])
        for unit_type in np.unique(dataset["type"]):
            mask_type = (dataset["type"] == unit_type)
            unit_subtypes = np.unique(dataset["subtype"][mask_type]) if cluster_type == "Demerged" else [0, ]
            for unit_subtype in unit_subtypes:
                if cluster_type is "Demerged":
                    mask_subtype = dataset["subtype"] == unit_subtype
                    mask_unit = mask_type & mask_subtype
                    unitID = "{}({})".format(unit_type, unit_subtype)
                else:
                    mask_unit = mask_type
                    unitID = "{}".format(unit_type)
                num_spike = mask_unit.sum()
                # if subtype != 0:
                #     continue
                # mask_subtype = (dataset["subtype"] == subtype)
                # mask_unit = mask_blk & mask_type & mask_subtype
                # num_spike = mask_unit.sum()
                if num_spike < num_trial:
                    print "\tUnit {} has too few spikes in block {} ({} spikes).\n".format(unitID, blk, num_spike)
                    continue

                print "\tProcessing unit {} ({} spikes)...".format(unitID, num_spike)

                spike_times = dataset["event_time"][mask_unit]
                trial_firing_rates = []
                trial_times = []
                for t_ini, t_fin in zip(ts_img_on, ts_img_off):
                    mask_trial = (t_ini <= spike_times) & (spike_times < t_fin)
                    trial_firing_rates.append(mask_trial.sum() / (t_fin-t_ini))
                    trial_times.append((t_ini + t_fin) / 2)
                trial_firing_rates = np.array(trial_firing_rates)
                trial_times = np.array(trial_times)
                if np.all(trial_firing_rates == 0):
                    print "\tUnit {} is not active in any trial.\n".format(unitID)
                    continue

                idx_unit_trial_ini = np.where(trial_firing_rates > 0)[0][0]
                idx_unit_trial_fin = np.where(trial_firing_rates > 0)[0][-1]
                num_active_trial = idx_unit_trial_fin - idx_unit_trial_ini + 1
                if num_active_trial <= 2*bin_size:
                    print "\tUnit {} is active in too few trials ({} trials).\n".format(unitID, num_active_trial)
                    continue

                # combine covariance values of multiple channels into one array
                spike_covs = np.empty((num_ch, num_spike))
                for i_ch in range(num_ch):
                    ch_label = "ch{}".format(i_ch)
                    spike_covs[i_ch] = dataset[ch_label][mask_unit]

                # define unit channel as the one with the maximum mean covariance
                unit_ch = spike_covs.mean(1).argmax()

                # compute trial resolved measures
                bin_edges = np.arange(idx_unit_trial_ini, idx_unit_trial_fin-2*bin_size+1, bin_step)
                num_bin = bin_edges.size
                bin_times_pval = np.empty(num_bin)
                fr_pvals = np.zeros(num_bin)
                for i, idx_ini in enumerate(bin_edges):
                    fr1 = trial_firing_rates[idx_ini:idx_ini+bin_size]
                    fr2 = trial_firing_rates[idx_ini+bin_size:idx_ini+2*bin_size]
                    _, fr_pvals[i] = spstats.ks_2samp(fr1, fr2)
                    bin_times_pval[i] = (trial_times[idx_ini+bin_size-1] + trial_times[idx_ini+bin_size]) / 2

                # define the p-value threshold for FDR with Benjamini-Hochberg method
                pval_thresholds = np.linspace(fdr_q / len(fr_pvals), fdr_q, len(fr_pvals))
                idxs_rejected_pval = np.where(np.sort(fr_pvals) < pval_thresholds)[0]
                pval_threshold = 0 if len(idxs_rejected_pval) == 0 else pval_thresholds[idxs_rejected_pval.max()]
                unstable_segment_edges = segment_successive_occurrences(fr_pvals < pval_threshold, True)
                stable_segment_edges = segment_successive_occurrences(fr_pvals < pval_threshold, False)

                # identify segments of stable (or unstable) rate and assign segment IDs
                # segment ID 0: when no unstable segments are identified, ID 0 is assigned to the whole episode
                # segment ID 1, 2, 3, ...: stable segments with the longest duration, the 2nd longest, and so on
                # segment ID -1, -2, -3, ...: unstable segments with the longest duration, the 2nd longest, and so on
                seg_edges = {}
                segIDs = np.empty_like(fr_pvals)
                if len(idxs_rejected_pval) == 0:
                    segIDs[:] = 0
                    seg_edges[0] = [0, len(fr_pvals)]
                elif len(idxs_rejected_pval) == len(fr_pvals):
                    segIDs[:] = -1
                    seg_edges[-1] = [0, len(fr_pvals)]
                else:
                    segID = -1
                    for i_seg in np.argsort([x[1]-x[0] for x in unstable_segment_edges])[::-1]:
                        seg_edge = unstable_segment_edges[i_seg]
                        seg_edges[segID] = seg_edge
                        segIDs[seg_edge[0]:seg_edge[1]] = segID
                        segID -= 1
                    segID = 1
                    for i_seg in np.argsort([x[1]-x[0] for x in stable_segment_edges])[::-1]:
                        seg_edge = stable_segment_edges[i_seg]
                        seg_edges[segID] = seg_edge
                        segIDs[seg_edge[0]:seg_edge[1]] = segID
                        segID += 1
                seg_time_ranges = {}
                for segID, seg_edge in seg_edges.items():
                    t_ini = spike_times[0] if seg_edge[0] == 0 else bin_times_pval[seg_edge[0]]
                    t_fin = spike_times[-1] if seg_edge[1] == len(fr_pvals) else bin_times_pval[seg_edge[1]-1]
                    seg_time_ranges[segID] = [t_ini, t_fin]

                # compute time resolved measures
                timebin_edges = np.arange(spike_times[0], spike_times[-1]-timebin_size+timebin_step/2, timebin_step)
                num_timebin = timebin_edges.size
                timebin_times = timebin_edges + timebin_size/2
                firing_rates = np.zeros(timebin_times.size)
                cov_means = np.zeros((num_ch, num_timebin))
                for i, t_ini in enumerate(timebin_edges):
                    idxs_spikes_in_bin = (t_ini <= spike_times) & (spike_times < t_ini+timebin_size)
                    firing_rates[i] = (idxs_spikes_in_bin).sum() / timebin_size
                    cov_means[:, i] = spike_covs[:, idxs_spikes_in_bin].mean(1)

                print "\t...done.\n"

                # make plots
                plt.figure(figsize=(10, 8))
                plt.subplots_adjust(left=0.08, right=0.96)
                title = "{} unit {} (Ch {}, {} spikes)".format(fn_spikes, unitID, unit_ch, num_spike)
                title += "\nbin size: {} trials, bin step: {} trials, FDR-q: {}, RPV threshold: {} ms".format(bin_size, bin_step, fdr_q, rpv_threshold)
                plt.suptitle(title)
                gs = gridspec.GridSpec(5, 2, width_ratios=[4, 1])

                plt.subplot(gs[0])
                plt.xlabel("Time (s)")
                plt.ylabel("Channel")
                X, Y = np.meshgrid(
                    timebin_times,
                    np.linspace(-0.5, num_ch-0.5, num_ch+1)
                )
                vmax = np.abs(cov_means).max()
                plt.pcolormesh(X, Y, cov_means, vmax=vmax, vmin=-vmax, cmap="bwr")
                plt.xlim(0, recdur)
                plt.ylim(23, 0)
                plt.grid(color="gray")
                # plt.colorbar().set_label("Waveform-template covariance")

                plt.subplot(gs[2])
                plt.xlabel("Time (s)")
                plt.ylabel("Spike size (cov)")
                plt.plot(spike_times, spike_covs[unit_ch], ",", color="black")
                for i_blk, b in enumerate(blks):
                    if b == 0:
                        continue
                    plt.axvspan(ts_blk_on[i_blk], ts_blk_off[i_blk], color=colors_task[tasks_blk[i_blk]], alpha=0.1, linewidth=0)
                for t_ini, t_fin in zip(ts_img_on, ts_img_off):
                # for i in range(idx_unit_trial_ini, idx_unit_trial_fin+1):
                #     t_ini = ts_img_on[i]
                #     t_fin = ts_img_off[i]
                    plt.axvspan(t_ini, t_fin, color=colors_task[tasks_blk[blk]], alpha=0.1, linewidth=0)
                plt.xlim(0, recdur)
                plt.ylim(0, 200)
                plt.grid(color="gray")
                plt.subplot(gs[3])
                plt.xlabel("Count")
                plt.ylabel("Spike size (cov)")
                plt.hist(spike_covs[unit_ch], bins=200, range=[0, 200], orientation="horizontal", linewidth=0, color="black")
                plt.grid(color="gray")

                ax1 = plt.subplot(gs[4])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(trial_times[idx_unit_trial_ini:idx_unit_trial_fin+1], trial_firing_rates[idx_unit_trial_ini:idx_unit_trial_fin+1], color="black", marker="+")
                plt.xlim(0, recdur)
                plt.ylim(ymin=0)
                plt.grid(color="gray")
                ax2 = ax1.twinx()
                plt.ylabel("Surprise")
                plt.plot(bin_times_pval, np.log10((1.0 - fr_pvals) / fr_pvals), color="magenta")
                plt.axhline(y=0, color="magenta", linestyle=":")
                plt.xlim(0, recdur)
                plt.ylim(-10, 10)

                ax1 = plt.subplot(gs[6])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(timebin_times, firing_rates, color="black")
                # for segID, seg_edge in seg_edges.items():
                for segID, [t_ini, t_fin] in seg_time_ranges.items():
                    seg_color = "blue" if segID >= 0 else "red"
                    plt.axvspan(t_ini, t_fin, color=seg_color, alpha=0.1, linewidth=0)
                plt.xlim(0, recdur)
                plt.ylim(ymin=0)
                plt.grid(color="gray")
                # ax2 = ax1.twinx()
                # plt.plot(bin_times_pval[segIDs >= 0], segIDs[segIDs >= 0], 'bo')
                # plt.plot(bin_times_pval[segIDs < 0], segIDs[segIDs < 0], 'ro')
                # plt.xlabel("Time (s)")
                # plt.ylabel("Segment ID")
                # plt.xlim(t_blk_on, t_blk_off)

                plt.subplot(gs[8])
                plt.xlabel("Time (s)")
                plt.ylabel("ISI (ms)")
                isis = np.diff(spike_times) * 1000
                plt.plot(spike_times[:-1], isis, ",", color="black")
                plt.axhline(rpv_threshold, color="red")
                for i_blk, b in enumerate(blks):
                    if b == 0:
                        continue
                    plt.axvspan(ts_blk_on[i_blk], ts_blk_off[i_blk], color=colors_task[tasks_blk[i_blk]], alpha=0.1, linewidth=0)
                for t_ini, t_fin in zip(ts_img_on, ts_img_off):
                # for i in range(idx_unit_trial_ini, idx_unit_trial_fin+1):
                #     t_ini = ts_img_on[i]
                #     t_fin = ts_img_off[i]
                    plt.axvspan(t_ini, t_fin, color=colors_task[tasks_blk[blk]], alpha=0.1, linewidth=0)
                plt.yscale('log')
                plt.xlim(0, recdur)
                plt.ylim(0.1, 1000)
                plt.grid(color="gray")
                plt.subplot(gs[9])
                plt.xlabel("Count")
                plt.ylabel("ISI (log10(ms))")
                plt.hist(np.log10(isis), bins=200, range=[np.log10(0.1), np.log10(1000)], orientation="horizontal", linewidth=0, color="black")
                plt.axhline(np.log10(rpv_threshold), color="red")
                plt.ylim(np.log10(0.1), np.log10(1000))
                plt.grid(color="gray")

                if savefig:
                    fn_fig = "{}/{}_unit{}.png".format(savedir, fn_spikes, unitID)
                    plt.savefig(fn_fig)
                    print "\tFigure saved as {}\n".format(fn_fig)
                    plt.close("all")
                else:
                    plt.show()
