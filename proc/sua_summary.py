import numpy as np
import scipy.stats as spstats
import matplotlib.pyplot as plt
from matplotlib import gridspec

import utils


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
    bin_width = 50  # bin width in sec
    bin_step = 1  # bin step in sec
    isi_max_ref = 0.003  # maximum ISI of spikes in refractory period

    # plot parameters
    colors_task = ["white", "blue", "yellow", "green"]

    # execution parameters
    savefig = True
    # savefig = False

    # session information
    datasets = [
        # ["HIME", "20140908", 4, "pc1", "09081319V1hp2"],
        # ["HIME", "20140908", 4, "pc2", "09081319IThp2"],
        # ["SATSUKI", "20150811", 6, "pc1", "08111157rec6V1hp2"],
        # ["SATSUKI", "20150811", 6, "pc2", "08111157rec6IThp2"],
        ["SATSUKI", "20151027", 5, 2, "IT"],
        ["SATSUKI", "20151027", 5, 2, "V1"],
        ["SATSUKI", "20151110", 7, 2, "IT"],
        ["SATSUKI", "20151110", 7, 2, "V1"],
        ]

    for dataset in datasets:
        sbj, sess, rec, blk, site = dataset
        fn_spikes = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
        print "\n{sbj}:{fn_spikes}".format(**locals())

        # fn_class_in = "{dir}/tmp/new/{fn}.class_Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
        # set filenames
        # fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
        fn_class = "{dir}/tmp/new/{fn}.class_Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
        # fn_seltypes = "{dir}/{sbj}/spikes/24ch/{fn}.types_SelectedCluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
        fn_seltypes = "{dir}/tmp/new/{fn}.types_SelectedCluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes)
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

        # get the list of selected units
        dataset = np.genfromtxt(fn_seltypes, dtype=None, names=True)
        selunitID = np.unique(dataset["type"])

        # load data
        print "\tLoading spike data file..."
        dataset = np.genfromtxt(fn_class, skip_header=2, dtype=None, names=True)
        num_ch = len(dataset.dtype.names) - 3
        recdur = dataset["event_time"][-1]
        print "\t...done.\n"

        unitID = np.unique(dataset["type"])
        for i_unit, uid in enumerate(unitID):
            selected = True if uid in selunitID else False

            mask_unit = dataset["type"] == uid
            num_spike = mask_unit.sum()
            if num_spike / (recdur/bin_width) < 2:
                print "\tUnit {} has too few spikes ({} spikes in {} sec).\n".format(uid, num_spike, recdur)
                continue

            print "\tProcessing unit {} ({} spikes)...".format(uid, num_spike)

            spike_times = dataset["event_time"][mask_unit]

            # combine covariance values of multiple channels into one array
            spike_covs = np.empty((num_ch, num_spike))
            for i_ch in range(num_ch):
                ch_label = "ch{}".format(i_ch)
                spike_covs[i_ch] = dataset[ch_label][mask_unit]

            # define unit channel as the one with the maximum mean covariance
            unit_ch = spike_covs.mean(1).argmax()

            # compute time resolved measures
            bin_edges = np.arange(0, spike_times[-1], bin_step)
            bin_times = bin_edges + bin_width/2
            num_bin = bin_times.size
            covs = np.zeros((num_ch, num_bin))
            cov_stds = np.zeros((num_ch, num_bin))
            cov_pvals = np.zeros(num_bin)
            isis = np.zeros(num_bin)
            isi_stds = np.zeros(num_bin)
            isi_pvals = np.zeros(num_bin)
            isi_reffrac = np.zeros(num_bin)
            for i, t_ini in enumerate(bin_edges[:-1]):
                mask_spikes_in_bin1 = (t_ini <= spike_times) & (spike_times < t_ini+bin_width)
                mask_spikes_in_bin2 = (t_ini+bin_width <= spike_times) & (spike_times < t_ini+2*bin_width)
                if mask_spikes_in_bin1.sum() < 2 or mask_spikes_in_bin2.sum() < 2 :
                    continue

                # inter-spike interval
                isi1 = np.diff(spike_times[mask_spikes_in_bin1])
                isi2 = np.diff(spike_times[mask_spikes_in_bin2])
                _, isi_pvals[i] = spstats.ks_2samp(isi1, isi2)
                isis[i] = isi1.mean()
                isi_stds[i] = isi1.std()
                isi_reffrac[i] = (isi1 < isi_max_ref).sum() / np.float(isi1.size)

                # covariance as spike size
                cov1 = spike_covs[:, mask_spikes_in_bin1]
                cov2 = spike_covs[:, mask_spikes_in_bin2]
                _, cov_pvals[i] = spstats.ks_2samp(cov1[unit_ch], cov2[unit_ch])
                covs[:, i] = cov1.mean(1)
                cov_stds[:, i] = cov1.std(1)

            print "\t...done.\n"

            # make plots
            plt.figure(figsize=(10,8))
            plt.subplots_adjust(left=0.08, right=0.96)
            title = "{} unit {} (Ch {}, {} spikes)".format(fn_spikes, uid, unit_ch, num_spike)
            if selected:
                title +="*"
            title += "\nbin width: {} s, bin step: {} s".format(bin_width, bin_step)
            plt.suptitle(title)
            gs = gridspec.GridSpec(5, 2, width_ratios=[4, 1])

            # plt.subplot(511)
            plt.subplot(gs[0])
            plt.xlabel("Time (s)")
            plt.ylabel("Channel")
            X, Y = np.meshgrid(
                np.linspace(-bin_step, bin_edges[-1]+bin_step/2, num_bin+1)+bin_width/2,
                np.linspace(-0.5, num_ch-0.5, num_ch+1)
            )
            vmax = np.abs(covs).max()
            plt.pcolormesh(X, Y, covs, vmax=vmax, vmin=-vmax, cmap="bwr")
            plt.xlim(0, recdur)
            plt.ylim(23, 0)
            plt.grid(color="gray")
            # plt.colorbar().set_label("Waveform-template covariance")

            # plt.subplot(512)
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

            # plt.subplot(513)
            plt.subplot(gs[4])
            plt.xlabel("Time (s)")
            plt.ylabel("Fraction of refractory ISIs")
            plt.plot(bin_times, isi_reffrac, color="black")
            plt.xlim(0, recdur)
            plt.ylim(0, 0.3)
            plt.grid(color="gray")

            # ax1 = plt.subplot(514)
            ax1 = plt.subplot(gs[6])
            plt.xlabel("Time (s)")
            plt.ylabel("Spike size (cov)")
            plt.plot(bin_times, covs[unit_ch], color="black")
            # plt.fill_between(bin_times, covs[unit_ch]-2*cov_stds[unit_ch], covs[unit_ch]+2*cov_stds[unit_ch], color="black", linewidth=0, alpha=0.2)
            plt.ylim(ymin=0)
            plt.grid(color="gray")
            ax2 = ax1.twinx()
            plt.ylabel("Surprise")
            plt.plot(bin_times+bin_width/2, np.log10((1.0-cov_pvals)/cov_pvals), color="magenta")
            plt.axhline(y=0, color="magenta", linestyle=":")
            plt.xlim(0, recdur)
            plt.ylim(-10, 10)

            # ax1 = plt.subplot(515)
            ax1 = plt.subplot(gs[8])
            plt.xlabel("Time (s)")
            plt.ylabel("Firing rate (1/s)")
            plt.plot(bin_times, 1.0/isis, color="black")
            # plt.ylabel("Inter-spike interval")
            # plt.plot(bin_times, isis, color="black")
            # plt.fill_between(bin_times, isis-2*isi_stds, isis+2*isi_stds, color="black", linewidth=0, alpha=0.2)
            plt.ylim(ymin=0)
            plt.grid(color="gray")
            ax2 = ax1.twinx()
            plt.ylabel("Surprise")
            plt.plot(bin_times+bin_width/2, np.log10((1.0-isi_pvals)/isi_pvals), color="magenta")
            plt.axhline(y=0, color="magenta", linestyle=":")
            plt.xlim(0, recdur)
            plt.ylim(-10, 10)

            if savefig:
                fn_fig = "{}/{}_unit{}.png".format(savedir, fn_spikes, uid)
                plt.savefig(fn_fig)
                print "\tFigure saved as {}\n".format(fn_fig)
                plt.close("all")
            else:
                plt.show()
