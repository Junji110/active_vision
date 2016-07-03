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
    return zip(idx_inc + 1, idx_dec + 1)


def replace_short_segments(data, value, repl_len, repl_to):
    data_local = np.array(data)
    seg_edges = segment_successive_occurrences(data_local, value)
    if seg_edges is None:
        return data_local
    for idx_ini, idx_fin in seg_edges:
        if idx_fin - idx_ini <= repl_len:
            data_local[idx_ini:idx_fin] = repl_to
    return data_local

    # # detect the edges of successive occurrences of `value` in `data`
    # data_replaced = np.array(data)
    # data_bin = (data_replaced == value).astype(int)
    # idx_inc = np.where(np.diff(data_bin) == 1)[0]
    # idx_dec = np.where(np.diff(data_bin) == -1)[0]
    #
    # # special cases where the edges are too few and so the following boundary treatment doesn't work
    # if len(idx_dec) <= 1 and len(idx_inc) <= 1:
    #     if len(idx_dec) == 1 and len(idx_inc) == 0:
    #         data_replaced[:idx_dec[0]+1] = repl_to if idx_dec[0]+1 <= repl_len else value
    #     elif idx_dec.size == 0 and idx_inc.size == 1:
    #         data_replaced[idx_inc[-1]+1:] = repl_to if len(data_bin)-(idx_inc[-1]+1) <= repl_len else value
    #     elif idx_dec.size == 1 and idx_inc.size == 1 and idx_dec[0] < idx_inc[0]:
    #         data_replaced[:idx_dec[0]+1] = repl_to if idx_dec[0]+1 <= repl_len else value
    #         data_replaced[idx_inc[-1]+1:] = repl_to if len(data_bin)-(idx_inc[-1]+1) <= repl_len else value
    #     return data_replaced
    #
    # # treatment of the boundaries in order to assure that idx_inc and idx_dec are of the same size and idx_inc[i] always
    # # precedes idx_dec[i] for any i
    # if idx_dec[0] < idx_inc[0]:
    #     data_replaced[:idx_dec[0]+1] = repl_to if idx_dec[0]+1 <= repl_len else value
    #     idx_dec = idx_dec[1:]
    # if idx_dec[-1] < idx_inc[-1]:
    #     data_replaced[idx_inc[-1]+1:] = repl_to if len(data_bin)-(idx_inc[-1]+1) <= repl_len else value
    #     idx_inc = idx_inc[:-1]
    #
    # for idx_ini, idx_fin in zip(idx_inc, idx_dec):
    #     if idx_fin - idx_ini <= repl_len:
    #         data_replaced[idx_ini+1:idx_fin+1] = repl_to
    # return data_replaced


if __name__ == "__main__":
    # file information
    # datasetdir = "z:"
    datasetdir = "/users/junji/desktop/ito/datasets/osaka"
    rawdir = "{}/RAWDATA".format(datasetdir)
    prepdir = "{}/PREPROCESSED".format(datasetdir)
    savedir = "."

    # analysis parameters
    sampling_rate = 20000.0
    bin_size = 500  # bin size in number of spikes
    bin_step = 50  # bin step in number of spikes

    # session information
    datasets = [
         ["HIME", "20140908", 4, "pc1", "09081309V1hp2"],
        # ["HIME", "20140908", 4, "pc1", "09081319V1hp2"],
        # ["HIME", "20140908", 4, "pc2", "09081319IThp2"],
        # ["SATSUKI", "20150811", 6, "pc1", "08111157rec6V1hp2"],
        # ["SATSUKI", "20150811", 6, "pc2", "08111157rec6IThp2"],
        ]

    for dataset in datasets:
        sbj, sess, rec, pc, fn_spikes = dataset
        print "\n{sbj}_{sess}_rec{rec}_{pc} ({fn_spikes})".format(**locals())

        # set filenames
        fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
        fn_class = "./{fn}.class_DemergedCluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
        fn_task = utils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]

        # load data
        print "\tLoading spike data file..."
        dataset = np.genfromtxt(fn_class, skip_header=2, dtype=None, names=True)
        num_ch = len(dataset.dtype.names) - 3
        recdur = dataset["event_time"][-1]
        print "\t...done.\n"

        unitIDs = np.unique(dataset["type"])
        new_unitID = unitIDs.max() + 1
        for i_unit, unitID in enumerate(unitIDs):
            mask_unit = dataset["type"] == unitID
            num_spike = mask_unit.sum()
            if num_spike < bin_size:
                print "\tUnit {} has too few spikes ({} spikes in {} sec).\n".format(unitID, num_spike, recdur)
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

            # time resolved estimation of cluster number
            bin_edges = np.arange(0, num_spike - bin_size, bin_step)
            num_bin = bin_edges.size
            bin_times = np.empty(num_bin)
            nums_clst = np.zeros(num_bin)
            for i, idx_ini in enumerate(bin_edges):
                idxs_spikes_in_bin = np.arange(idx_ini, idx_ini + bin_size)

                bin_times[i] = (spike_times[idxs_spikes_in_bin[0]] + spike_times[idxs_spikes_in_bin[-1]]) / 2

                cov = spike_covs[:, idxs_spikes_in_bin]
                gaps = gap(cov[unit_ch][:, np.newaxis], nrefs=1, ks=[1, 2])
                nums_clst[i] = gaps.argmax() + 1

                print "{} in {}".format(i, len(bin_edges))

            print "\t...done.\n"

            # smooth nums_clst so that a sudden shift of single unit's spike size within a bin is not detected as
            # multiple units
            # first, flip short segments of 1's to 2's (= concatenate neighboring segments of 2's as much as possible)
            nums_clst = replace_short_segments(nums_clst, 1, bin_size/bin_step, 2)
            # then, flip isolated short segments of 2's to 1's
            nums_clst = replace_short_segments(nums_clst, 2, bin_size/bin_step, 1)

            # nums_clst[:len(nums_clst)/4] = 2
            # nums_clst[len(nums_clst)/2:len(nums_clst)/4*3] = 2

            # re-assign new unit IDs if 2 clusters are found in any bin
            if np.any(nums_clst == 2):
                unitIDs_tmp = np.ones_like(dataset["type"][mask_unit])
                # assign a negative unit ID to the spikes in segments of 2's
                seg_edges = segment_successive_occurrences(nums_clst, 2)
                if seg_edges is not None:
                    for idx_ini, idx_fin in seg_edges:
                        idx_ini_spike = 0 if idx_ini == 0 else bin_edges[idx_ini]
                        idx_fin_spike = len(mask_unit) if idx_fin == len(bin_edges) else bin_edges[idx_fin-1] + bin_size
                        unitIDs_tmp[idx_ini_spike:idx_fin_spike] = -new_unitID
                        new_unitID += 1
                # assign a positive unit ID to the spikes in segments of 1's
                seg_edges = segment_successive_occurrences(nums_clst, 1)
                if seg_edges is not None:
                    for idx_ini, idx_fin in seg_edges:
                        idx_ini_spike = 0 if idx_ini == 0 else bin_edges[idx_ini] + bin_size/2
                        idx_fin_spike = len(mask_unit) if idx_fin == len(bin_edges) else bin_edges[idx_fin-1] + bin_size/2
                        unitIDs_tmp[idx_ini_spike:idx_fin_spike] = new_unitID
                        new_unitID += 1
                dataset["type"][mask_unit] = unitIDs_tmp

                # plt.plot(dataset["type"][mask_unit])
                # plt.plot(bin_edges+bin_size/2, nums_clst * new_unitID)
                # plt.grid()
                # plt.show()

            # make plots
            plt.figure(figsize=(10, 8))
            plt.subplots_adjust(left=0.08, right=0.96)
            title = "{} unit {} (Ch {}, {} spikes)".format(fn_spikes, unitID, unit_ch, num_spike)
            title += "\nbin size: {} spks, bin step: {} spks".format(bin_size, bin_step)
            plt.suptitle(title)

            ax1 = plt.subplot(111)
            plt.xlabel("Time (s)")
            plt.ylabel("Spike size (cov)")
            plt.plot(spike_times, spike_covs[unit_ch], ",", color="black")
            plt.xlim(0, recdur)
            plt.ylim(0, 200)
            plt.grid(color="gray")
            ax2 = ax1.twinx()
            plt.ylabel("# of clusters")
            plt.plot(bin_times, nums_clst, color="magenta")
            plt.xlim(0, recdur)
            plt.ylim(0, 3)

            plt.show()

        # with open("{}.class_DemergedCluster".format(fn_spikes), "w") as f:
        #     f.write("OriginalFile={}\n".format(fn_class))
        #     f.write("bin_size[spikes]={bin_size}\tbin_step[spikes]={bin_step}\n".format(**locals()))
        #     f.write("\t".join(dataset.dtype.names) + "\n")
        #     for data in dataset:
        #         f.write("\t".join([str(x) for x in data]) + "\n")

