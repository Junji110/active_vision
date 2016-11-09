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


if __name__ == "__main__":
    # file information
    datasetdir = "/home/ito/datasets/osaka"
    rawdir = "{}/RAWDATA".format(datasetdir)
    prepdir = "{}/PREPROCESSED".format(datasetdir)
    savedir = "./scratch"

    # analysis parameters
    sampling_rate = 20000.0
    bin_size = 200  # bin width in number of spikes
    bin_step = 20  # bin step in number of spikes
    num_bin_hist = 40
    rpv_threshold = 1.0  # refractory period voilation threshold in ms

    # plot parameters
    colors_task = ["white", "blue", "yellow", "green"]

    # execution parameters

    # detect_change_point = True
    detect_change_point = False

    estimate_nums_clst = True
    # estimate_nums_clst = False

    # savefig = True
    savefig = False

    # session information
    datasets = [
        # format: [sbj, sess, rec, blk, site, cluster_type]
        # ["HIME", "20140908", 4, 0, "09081319V1hp2", ""],  # for the 'long' sorting data, set "blk" to "0" and set "site" to the .class_Cluster file name
        # ["HIME", "20140908", 4, 0, "09081319IThp2", ""],
        # ["SATSUKI", "20150811", 6, 0, "08111157rec6V1hp2", ""],
        # ["SATSUKI", "20150811", 6, 0, "08111157rec6IThp2", ""],
        # ["HIME", "20140908", 4, 11, "V1", ""],
        # ["HIME", "20140908", 4, 11, "V1", "Demerged"],
        # ["HIME", "20140908", 4, 11, "IT", ""],
        # ["HIME", "20140908", 4, 11, "IT", "Demerged"],
        # ["SATSUKI", "20151027", 5, 2, "V1", ""],
        # ["SATSUKI", "20151027", 5, 2, "V1", "Demerged"],
        # ["SATSUKI", "20151027", 5, 2, "IT", ""],
        # ["SATSUKI", "20151027", 5, 2, "IT", "Demerged"],
        # ["SATSUKI", "20151110", 7, 2, "V1", ""],
        # ["SATSUKI", "20151110", 7, 2, "V1", "Demerged"],
        ["SATSUKI", "20151110", 7, 2, "IT", ""],
        # ["SATSUKI", "20151110", 7, 2, "IT", "Demerged"],
    ]


    for dataset in datasets:
        sbj, sess, rec, blk, site, cluster_type = dataset

        # set filenames
        fn_task = utils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]
        if blk == 0:
            fn_spikes = site
            fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        else:
            fn_spikes = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
            fn_class = "{dir}/tmp/new/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        print "\n{sbj}:{fn_spikes} ({cluster_type})".format(**locals())

        # load task events
        print "\tLoading task data file..."
        task_events, task_param = utils.load_task(fn_task)
        blks = np.unique(task_events['block'])
        ts_blk_on = []
        ts_blk_off = []
        tasks_blk = []
        for b in blks:
            blk_events, blk_param = utils.load_task(fn_task, b)
            tasks_blk.append(blk_param['task'][0])
            ts_blk_on.append(blk_events["evtime"][0] / sampling_rate)
            ts_blk_off.append(blk_events["evtime"][-2] / sampling_rate)
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

        types = np.unique(dataset["type"])
        for i_unit, type in enumerate(types):
            mask_type = dataset["type"] == type
            subtypes = np.unique(dataset["subtype"][mask_type]) if cluster_type == "Demerged" else [0,]
            for subtype in subtypes:
                if cluster_type is "Demerged":
                    mask_subtype = dataset["subtype"] == subtype
                    mask_unit = mask_type & mask_subtype
                    unitID = "{}({})".format(type, subtype)
                else:
                    mask_unit = mask_type
                    unitID = "{}".format(type)
                num_spike = mask_unit.sum()
                if num_spike <= bin_size:
                    print "\tUnit {}  has too few spikes ({} spikes in {} sec).\n".format(unitID, num_spike, recdur)
                    continue

                print "\tProcessing unit {} ({} spikes)...".format(unitID, num_spike)

                spike_times = dataset["event_time"][mask_unit]

                # combine covariance values of multiple channels into one array
                spike_covs = np.empty((num_ch, num_spike))
                for i_ch in range(num_ch):
                    ch_label = "ch{}".format(i_ch)
                    spike_covs[i_ch] = dataset[ch_label][mask_unit]

                # define unit channel as the one with the maximum mean covariance
                unit_ch = spike_covs.mean(1).argmax()

                # compute time resolved measures
                bin_edges = np.arange(0, num_spike - bin_size, bin_step)
                num_bin = bin_edges.size
                bin_times = np.empty(num_bin)
                cov_means = np.zeros((num_ch, num_bin))
                isi_means = np.zeros(num_bin)
                spike_size_hist = np.zeros((num_bin_hist, num_bin))
                if estimate_nums_clst:
                    nums_clst = np.zeros(num_bin)
                if detect_change_point:
                    bin_times_pval = np.empty(num_bin)
                    cov_pvals = np.zeros(num_bin)
                    isi_pvals = np.zeros(num_bin)
                for i, idx_ini in enumerate(bin_edges):
                    idxs_spikes_in_bin = np.arange(idx_ini, idx_ini + bin_size)
                    bin_times[i] = spike_times[idxs_spikes_in_bin][[0, -1]].mean()

                    # inter-spike interval
                    isi = np.diff(spike_times[idxs_spikes_in_bin])
                    isi_means[i] = isi.mean()

                    # covariance as spike size
                    cov = spike_covs[:, idxs_spikes_in_bin]
                    cov_means[:, i] = cov.mean(1)

                    if detect_change_point:
                        if idx_ini + 2*bin_size < num_spike:
                            idxs_spikes_in_bin2 = np.arange(idx_ini + bin_size, idx_ini + 2*bin_size)
                            bin_times_pval[i] = (spike_times[idxs_spikes_in_bin[0]] + spike_times[idxs_spikes_in_bin2[-1]]) / 2
                            isi2 = np.diff(spike_times[idxs_spikes_in_bin2])
                            _, isi_pvals[i] = spstats.ks_2samp(isi, isi2)
                            cov2 = spike_covs[:, idxs_spikes_in_bin2]
                            _, cov_pvals[i] = spstats.ks_2samp(cov[unit_ch], cov2[unit_ch])

                    # spike size histogram
                    hist, bin_edges_hist = np.histogram(cov[unit_ch], range=[0, 200], bins=num_bin_hist)
                    spike_size_hist[:, i] = hist

                    if estimate_nums_clst:
                        gaps = gap(cov[unit_ch][:, np.newaxis], nrefs=1, ks=[1, 2])
                        nums_clst[i] = gaps.argmax() + 1

                    if i % 100 == 99:
                        print "\t...{} of {} bins processed...".format(i+1, len(bin_edges))


                print "\t...done.\n"

                # make plots
                plt.figure(figsize=(10, 8))
                plt.subplots_adjust(left=0.08, right=0.96)
                title = "{} unit {} (Ch {}, {} spikes)".format(fn_spikes, unitID, unit_ch, num_spike)
                title += "\nbin size: {} spks, bin step: {} spks, RPV threshold: {} ms".format(bin_size, bin_step, rpv_threshold)
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
                for i_blk, b in enumerate(blks):
                    if b == 0:
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
                # plt.plot(bin_times, cov_means[unit_ch], color="black")
                # plt.xlim(0, recdur)
                # plt.ylim(ymin=0)
                # plt.grid(color="gray")
                # if detect_change_point:
                #     ax2 = ax1.twinx()
                #     plt.ylabel("Surprise")
                #     plt.plot(bin_times_pval, np.log10((1.0 - cov_pvals) / cov_pvals), color="magenta")
                #     plt.axhline(y=0, color="magenta", linestyle=":")
                #     plt.xlim(0, recdur)
                #     plt.ylim(-10, 10)

                ax1 = plt.subplot(gs[6])
                plt.xlabel("Time (s)")
                plt.ylabel("Firing rate (1/s)")
                plt.plot(bin_times, 1.0 / isi_means, color="black")
                plt.xlim(0, recdur)
                plt.ylim(ymin=0)
                plt.grid(color="gray")
                if detect_change_point:
                    ax2 = ax1.twinx()
                    plt.ylabel("Surprise")
                    plt.plot(bin_times_pval, np.log10((1.0 - isi_pvals) / isi_pvals), color="magenta")
                    plt.axhline(y=0, color="magenta", linestyle=":")
                    plt.xlim(0, recdur)
                    plt.ylim(-10, 10)

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
