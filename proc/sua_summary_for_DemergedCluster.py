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
    bin_size = 500  # bin width in number of spikes
    bin_step = 50  # bin step in number of spikes
    num_bin_hist = 40
    timebin_size = 50.0  # time-based bin width in seconds
    timebin_step = 5.0  # time-based bin step in seconds

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
        ["HIME", "20140908", 4, "pc1", "09081319V1hp2"],
        ["HIME", "20140908", 4, "pc2", "09081319IThp2"],
        ["SATSUKI", "20150811", 6, "pc1", "08111157rec6V1hp2"],
        ["SATSUKI", "20150811", 6, "pc2", "08111157rec6IThp2"],
        ]

    for dataset in datasets:
        sbj, sess, rec, pc, fn_spikes = dataset
        print "\n{sbj}_{sess}_rec{rec}_{pc} ({fn_spikes})".format(**locals())

        # set filenames
        fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        fn_task = utils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]

        # load task events
        print "\tLoading task data file..."
        task_events, task_param = utils.load_task(fn_task)
        blks = np.unique(task_events['block'])
        ts_blk_on = []
        ts_blk_off = []
        tasks_blk = []
        for blk in blks:
            blk_events, blk_param = utils.load_task(fn_task, blk)
            tasks_blk.append(blk_param['task'][0])
            ts_blk_on.append(blk_events["evtime"][0] / sampling_rate)
            ts_blk_off.append(blk_events["evtime"][-2] / sampling_rate)
        print "\t...done.\n"

        # load data
        print "\tLoading spike data file..."
        dataset = np.genfromtxt(fn_class, skip_header=2, dtype=None, names=True)
        num_ch = len(dataset.dtype.names) - 4
        recdur = dataset["event_time"][-1]
        print "\t...done.\n"

        types = np.unique(dataset["type"])
        for type in types:
            mask_type = dataset["type"] == type
            if cluster_type is "Demerged":
                subtypes = np.unique(dataset["subtype"][mask_type])
            else:
                subtypes = [0,]
            for subtype in subtypes:
                # if subtype != 0:
                #     continue
                mask_subtype = dataset["subtype"] == subtype
                mask_unit = mask_type & mask_subtype
                num_spike = mask_unit.sum()
                if num_spike <= 2*bin_size:
                    print "\tUnit {} has too few spikes ({} spikes in {} sec).\n".format(type, num_spike, recdur)
                    continue

                print "\tProcessing unit {}({}) ({} spikes)...".format(type, subtype, num_spike)

                spike_times = dataset["event_time"][mask_unit]

                # combine covariance values of multiple channels into one array
                spike_covs = np.empty((num_ch, num_spike))
                for i_ch in range(num_ch):
                    ch_label = "ch{}".format(i_ch)
                    spike_covs[i_ch] = dataset[ch_label][mask_unit]

                # define unit channel as the one with the maximum mean covariance
                unit_ch = spike_covs.mean(1).argmax()

                # compute spike count resolved measures
                bin_edges = np.arange(0, num_spike - 2 * bin_size, bin_step)
                num_bin = bin_edges.size
                bin_times = np.empty(num_bin)
                bin_times_pval = np.empty(num_bin)
                cov_means = np.zeros((num_ch, num_bin))
                cov_stds = np.zeros((num_ch, num_bin))
                cov_pvals = np.zeros(num_bin)
                isis = np.zeros(num_bin)
                isi_stds = np.zeros(num_bin)
                isi_pvals = np.zeros(num_bin)
                spike_size_hist = np.zeros((num_bin_hist, num_bin))
                nums_clst = np.zeros(num_bin)
                for i, idx_ini in enumerate(bin_edges):
                    idxs_spikes_in_bin1 = np.arange(idx_ini, idx_ini + bin_size)
                    idxs_spikes_in_bin2 = np.arange(idx_ini + bin_size, idx_ini + 2 * bin_size)

                    # bin_times[i] = (spike_times[idxs_spikes_in_bin1[0]] + spike_times[idxs_spikes_in_bin1[-1]]) / 2
                    bin_times[i] = np.median(spike_times[idxs_spikes_in_bin1])
                    bin_times_pval[i] = (spike_times[idxs_spikes_in_bin1[-1]] + spike_times[idxs_spikes_in_bin2[0]]) / 2

                    # inter-spike interval
                    isi1 = np.diff(spike_times[idxs_spikes_in_bin1])
                    isi2 = np.diff(spike_times[idxs_spikes_in_bin2])
                    _, isi_pvals[i] = spstats.ks_2samp(isi1, isi2)
                    isis[i] = isi1.mean()
                    isi_stds[i] = isi1.std()


                    # spike sizes across channels
                    covs = spike_covs[:, idxs_spikes_in_bin1]
                    cov_means[:, i] = covs.mean(1)

                    # spike size histogram (at the unit channel)
                    hist, bin_edges_hist = np.histogram(covs[unit_ch], range=[0, 200], bins=num_bin_hist)
                    spike_size_hist[:, i] = hist
                    if estimate_nums_clst:
                        gaps = gap(covs[unit_ch][:, np.newaxis], nrefs=1, ks=[1, 2])
                        nums_clst[i] = gaps.argmax() + 1
                        print "\t{} in {}".format(i, len(bin_edges))

                # compute time resolved measures
                timebin_edges = np.arange(0, spike_times[-1], timebin_step)
                timebin_times = timebin_edges + timebin_size/2
                firing_rates = np.zeros(timebin_times.size)
                for i, t_ini in enumerate(timebin_edges[:-1]):
                    firing_rates[i] = ((t_ini <= spike_times) & (spike_times < t_ini+timebin_size)).sum() / timebin_size

                # FDR with Benjamini-Hochberg method
                qval = 0.01
                pval_thresholds = np.linspace(qval / len(isi_pvals), qval, len(isi_pvals))
                idxs_rejected_pval = np.where(np.sort(isi_pvals) < pval_thresholds)[0]
                if len(idxs_rejected_pval) > 1:
                    pval_threshold = pval_thresholds[idxs_rejected_pval.max()]
                    rejected_segment_edges = segment_successive_occurrences(isi_pvals < pval_threshold, 1)
                else:
                    rejected_segment_edges = None

                # plt.plot(timebin_times, firing_rates, color="black")
                # for idx_ini, idx_fin in rejected_segment_edges:
                #     plt.axvspan(bin_times_pval[idx_ini], bin_times_pval[idx_fin-1], color="red", alpha=0.2, linewidth=0)
                # # plt.plot(np.sort(isi_pvals))
                # # plt.plot(pval_thresholds)
                # # plt.axhline(y=pval_threshold)
                # plt.show()

                print "\t...done.\n"

                # make plots
                plt.figure(figsize=(10, 8))
                plt.subplots_adjust(left=0.08, right=0.96)
                title = "{} unit {}({}) (Ch {}, {} spikes)".format(fn_spikes, type, subtype, unit_ch, num_spike)
                title += "\nbin size: {} spks, bin step: {} spks, FDR q: {}".format(bin_size, bin_step, qval)
                plt.suptitle(title)
                gs = gridspec.GridSpec(5, 2, width_ratios=[4, 1])

                plt.subplot(gs[0])
                plt.xlabel("Time (s)")
                plt.ylabel("Channel")
                X, Y = np.meshgrid(
                    bin_times,
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
                for i_blk, blk in enumerate(blks):
                    if blk == 0:
                        continue
                    plt.axvspan(ts_blk_on[i_blk], ts_blk_off[i_blk], color=colors_task[tasks_blk[i_blk]], alpha=0.1)
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
                plt.ylabel("Spike size (cov)")
                X, Y = np.meshgrid(
                    bin_times,
                    bin_edges_hist
                )
                plt.pcolormesh(X, Y, spike_size_hist, cmap="rainbow")
                plt.xlim(0, recdur)
                plt.ylim(0, 200)
                plt.grid(color="gray")
                # plt.colorbar().set_label("Count")
                if estimate_nums_clst:
                    ax2 = ax1.twinx()
                    plt.ylabel("# of clusters")
                    plt.plot(bin_times, nums_clst, color="magenta")
                    plt.xlim(0, recdur)
                    plt.ylim(0, 3)

                # ax1 = plt.subplot(gs[6])
                # plt.xlabel("Time (s)")
                # plt.ylabel("Spike size (cov)")
                # plt.plot(bin_times, covs[unit_ch], color="black")
                # # plt.fill_between(bin_times, covs[unit_ch]-2*cov_stds[unit_ch], covs[unit_ch]+2*cov_stds[unit_ch], color="black", linewidth=0, alpha=0.2)
                # plt.ylim(ymin=0)
                # plt.grid(color="gray")
                # ax2 = ax1.twinx()
                # plt.ylabel("Surprise")
                # plt.plot(bin_times_pval, np.log10((1.0 - cov_pvals) / cov_pvals), color="magenta")
                # plt.axhline(y=0, color="magenta", linestyle=":")
                # plt.xlim(0, recdur)
                # plt.ylim(-10, 10)

                ax1 = plt.subplot(gs[6])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(bin_times, 1.0/isis, color="black")
                plt.ylim(ymin=0)
                plt.grid(color="gray")
                ax2 = ax1.twinx()
                plt.ylabel("Surprise")
                plt.plot(bin_times_pval, np.log10((1.0 - isi_pvals) / isi_pvals), color="magenta")
                plt.axhline(y=0, color="magenta", linestyle=":")
                plt.xlim(0, recdur)
                plt.ylim(-10, 10)

                ax1 = plt.subplot(gs[8])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(timebin_times, firing_rates, color="black")
                if rejected_segment_edges is not None:
                    for idx_ini, idx_fin in rejected_segment_edges:
                        t_ini = 0 if idx_ini == 0 else bin_times_pval[idx_ini]
                        t_fin = recdur if idx_ini == len(bin_times_pval)+1 else bin_times_pval[idx_fin-1]
                        plt.axvspan(t_ini, t_fin, color="red", alpha=0.2, linewidth=0)
                plt.xlim(0, recdur)
                plt.ylim(ymin=0)
                plt.grid(color="gray")

                if savefig:
                    fn_fig = "{}/{}_unit{}({}).png".format(savedir, fn_spikes, type, subtype)
                    plt.savefig(fn_fig)
                    print "\tFigure saved as {}\n".format(fn_fig)
                    plt.close("all")
                else:
                    plt.show()
