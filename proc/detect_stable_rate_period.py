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
    datasetdir = "/users/junji/desktop/ito/datasets/osaka"
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

    # plot parameters
    colors_task = ["white", "blue", "yellow", "green"]

    # execution parameters
    # cluster_type = ""
    cluster_type = "Demerged"

    # estimate_nums_clst = True
    estimate_nums_clst = False

    savefig = True
    # savefig = False

    # session information
    datasets = [
        # ["HIME", "20140908", 4, "pc1", "09081309V1hp2"],
        ["HIME", "20140908", 4, 11, "pc1", "09081319V1hp2"],
        ["HIME", "20140908", 4, 11, "pc2", "09081319IThp2"],
        ["SATSUKI", "20150811", 6, 2, "pc1", "08111157rec6V1hp2"],
        ["SATSUKI", "20150811", 6, 2, "pc2", "08111157rec6IThp2"],
        ]

    for dataset in datasets:
        sbj, sess, rec, blk, pc, fn_spikes = dataset
        print "\n{sbj}_{sess}_rec{rec}_{pc} ({fn_spikes})".format(**locals())

        # set filenames
        fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        fn_task = utils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]

        # load task events
        print "\tLoading task data file..."
        task_events, task_param = utils.load_task(fn_task, blk)
        task_blk = task_param['task'][0]
        evID_img_on = task_blk*100 + 11
        evID_img_off = task_blk*100 + 12
        t_blk_on = task_events["evtime"][0] / sampling_rate
        t_blk_off = task_events["evtime"][-2] / sampling_rate
        ts_img_on = []
        ts_img_off = []
        for i_trial in range(task_param["num_trials"]):
            # reject failure trials
            if task_param['success'][i_trial] <= 0:
                continue

            trialID = i_trial + 1
            trial_events = task_events[task_events['trial'] == trialID]

            # reject trials with missing image-onset or offset events
            if (trial_events['evID'] != evID_img_on).all() or (trial_events['evID'] != evID_img_off).all():
                continue

            ts_img_on.append(trial_events['evtime'][trial_events['evID'] == evID_img_on][0] / sampling_rate)
            ts_img_off.append(trial_events['evtime'][trial_events['evID'] == evID_img_off][0] / sampling_rate)
        ts_img_on = np.array(ts_img_on)
        ts_img_off = np.array(ts_img_off)
        assert(len(ts_img_on) == len(ts_img_off))
        num_trial = len(ts_img_on)
        print "\t...done.\n"

        # load data
        print "\tLoading spike data file..."
        dataset = np.genfromtxt(fn_class, skip_header=2, dtype=None, names=True)
        num_ch = len(dataset.dtype.names) - 4
        recdur = dataset["event_time"][-1]
        print "\t...done.\n"

        mask_blk = (t_blk_on < dataset["event_time"]) & (dataset["event_time"] < t_blk_off)
        for unit_type in np.unique(dataset["type"]):
            mask_type = (dataset["type"] == unit_type)
            if cluster_type is "Demerged":
                subtypes = np.unique(dataset["subtype"][mask_type])
            else:
                subtypes = [0,]
            for subtype in subtypes:
                # if subtype != 0:
                #     continue
                mask_subtype = (dataset["subtype"] == subtype)
                mask_unit = mask_blk & mask_type & mask_subtype
                num_spike = mask_unit.sum()
                if num_spike < num_trial:
                    print "\tUnit {}({}) has too few spikes in block {} ({} spikes).\n".format(unit_type, subtype, blk, num_spike)
                    continue

                print "\tProcessing unit {}({}) ({} spikes)...".format(unit_type, subtype, num_spike)

                spike_times = dataset["event_time"][mask_unit]
                mask_unit_trial = [((t_ini <= spike_times) & (spike_times < t_fin)).sum() > 0 for t_ini, t_fin in zip(ts_img_on, ts_img_off)]
                idx_unit_trial_ini = np.where(mask_unit_trial)[0][0]
                idx_unit_trial_fin = np.where(mask_unit_trial)[0][-1]
                num_active_trial = idx_unit_trial_fin - idx_unit_trial_ini + 1
                if num_active_trial <= 2*bin_size:
                    print "\tUnit {}({}) is active in too few trials ({} trials).\n".format(unit_type, subtype, num_active_trial)
                    continue

                # combine covariance values of multiple channels into one array
                spike_covs = np.empty((num_ch, num_spike))
                for i_ch in range(num_ch):
                    ch_label = "ch{}".format(i_ch)
                    spike_covs[i_ch] = dataset[ch_label][mask_unit]

                # define unit channel as the one with the maximum mean covariance
                unit_ch = spike_covs.mean(1).argmax()

                trial_firing_rates = []
                trial_times = []
                for t_ini, t_fin in zip(ts_img_on, ts_img_off):
                    mask_trial = (t_ini <= spike_times) & (spike_times < t_fin)
                    trial_firing_rates.append(mask_trial.sum() / (t_fin-t_ini))
                    trial_times.append((t_ini + t_fin) / 2)

                # compute trial resolved measures
                bin_edges = np.arange(idx_unit_trial_ini, idx_unit_trial_fin-2*bin_size+1, bin_step)
                # bin_edges = np.arange(0, num_trial - 2 * bin_size, bin_step)
                num_bin = bin_edges.size
                bin_times_pval = np.empty(num_bin)
                fr_pvals = np.zeros(num_bin)
                for i, idx_ini in enumerate(bin_edges):
                    fr1 = trial_firing_rates[idx_ini:idx_ini+bin_size]
                    fr2 = trial_firing_rates[idx_ini+bin_size:idx_ini+2*bin_size]
                    _, fr_pvals[i] = spstats.ks_2samp(fr1, fr2)
                    bin_times_pval[i] = (trial_times[idx_ini+bin_size-1] + trial_times[idx_ini+bin_size]) / 2

                # compute time resolved measures
                timebin_edges = np.arange(0, spike_times[-1], timebin_step)
                timebin_times = timebin_edges + timebin_size/2
                firing_rates = np.zeros(timebin_times.size)
                for i, t_ini in enumerate(timebin_edges[:-1]):
                    firing_rates[i] = ((t_ini <= spike_times) & (spike_times < t_ini+timebin_size)).sum() / timebin_size

                # FDR with Benjamini-Hochberg method
                pval_thresholds = np.linspace(fdr_q / len(fr_pvals), fdr_q, len(fr_pvals))
                idxs_rejected_pval = np.where(np.sort(fr_pvals) < pval_thresholds)[0]
                if len(idxs_rejected_pval) > 1:
                    pval_threshold = pval_thresholds[idxs_rejected_pval.max()]
                    rejected_segment_edges = segment_successive_occurrences(fr_pvals < pval_threshold, 1)
                else:
                    rejected_segment_edges = None


                print "\t...done.\n"

                # make plots
                plt.figure(figsize=(10, 8))
                plt.subplots_adjust(left=0.08, right=0.96)
                title = "{} unit {}({}) (Ch {}, {} spikes)".format(fn_spikes, unit_type, subtype, unit_ch, num_spike)
                title += "\nbin size: {} trials, bin step: {} trials, FDR-q: {}".format(bin_size, bin_step, fdr_q)
                plt.suptitle(title)
                gs = gridspec.GridSpec(5, 2, width_ratios=[4, 1])

                # plt.subplot(gs[0])
                # plt.xlabel("Time (s)")
                # plt.ylabel("Channel")
                # X, Y = np.meshgrid(
                #     bin_times,
                #     np.linspace(-0.5, num_ch-0.5, num_ch+1)
                # )
                # vmax = np.abs(cov_means).max()
                # plt.pcolormesh(X, Y, cov_means, vmax=vmax, vmin=-vmax, cmap="bwr")
                # plt.xlim(0, recdur)
                # plt.ylim(23, 0)
                # plt.grid(color="gray")
                # # plt.colorbar().set_label("Waveform-template covariance")

                plt.subplot(gs[2])
                plt.xlabel("Time (s)")
                plt.ylabel("Spike size (cov)")
                plt.plot(spike_times, spike_covs[unit_ch], ",", color="black")
                # for t_ini, t_fin in zip(ts_img_on, ts_img_off):
                for i in range(idx_unit_trial_ini, idx_unit_trial_fin+1):
                    t_ini = ts_img_on[i]
                    t_fin = ts_img_off[i]
                    plt.axvspan(t_ini, t_fin, color=colors_task[task_blk], alpha=0.1, linewidth=0)
                plt.xlim(t_blk_on, t_blk_off)
                plt.ylim(0, 200)
                plt.grid(color="gray")
                plt.subplot(gs[3])
                plt.xlabel("Count")
                plt.ylabel("Spike size (cov)")
                plt.hist(spike_covs[unit_ch], bins=200, range=[0, 200], orientation="horizontal", linewidth=0, color="black")
                plt.grid(color="gray")

                # ax1 = plt.subplot(gs[4])
                # plt.xlabel("Time (s)")
                # plt.ylabel("Spike size (cov)")
                # X, Y = np.meshgrid(
                #     bin_times,
                #     bin_edges_hist
                # )
                # plt.pcolormesh(X, Y, spike_size_hist, cmap="rainbow")
                # plt.xlim(0, recdur)
                # plt.ylim(0, 200)
                # plt.grid(color="gray")
                # # plt.colorbar().set_label("Count")
                # if estimate_nums_clst:
                #     ax2 = ax1.twinx()
                #     plt.ylabel("# of clusters")
                #     plt.plot(bin_times, nums_clst, color="magenta")
                #     plt.xlim(0, recdur)
                #     plt.ylim(0, 3)

                ax1 = plt.subplot(gs[6])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(trial_times[idx_unit_trial_ini:idx_unit_trial_fin+1], trial_firing_rates[idx_unit_trial_ini:idx_unit_trial_fin+1], color="black", marker="+")
                plt.xlim(t_blk_on, t_blk_off)
                plt.ylim(ymin=0)
                plt.grid(color="gray")
                ax2 = ax1.twinx()
                plt.ylabel("Surprise")
                plt.plot(bin_times_pval, np.log10((1.0 - fr_pvals) / fr_pvals), color="magenta")
                plt.axhline(y=0, color="magenta", linestyle=":")
                plt.xlim(t_blk_on, t_blk_off)
                plt.ylim(-10, 10)

                ax1 = plt.subplot(gs[8])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(timebin_times, firing_rates, color="black")
                if rejected_segment_edges is not None:
                    for idx_ini, idx_fin in rejected_segment_edges:
                        t_ini = spike_times[0] if idx_ini == 0 else bin_times_pval[idx_ini]
                        t_fin = spike_times[-1] if idx_ini == len(bin_times_pval)+1 else bin_times_pval[idx_fin-1]
                        plt.axvspan(t_ini, t_fin, color="red", alpha=0.2, linewidth=0)
                plt.xlim(t_blk_on, t_blk_off)
                plt.ylim(ymin=0)
                plt.grid(color="gray")

                if savefig:
                    fn_fig = "{}/{}_unit{}({}).png".format(savedir, fn_spikes, unit_type, subtype)
                    plt.savefig(fn_fig)
                    print "\tFigure saved as {}\n".format(fn_fig)
                    plt.close("all")
                else:
                    plt.show()
