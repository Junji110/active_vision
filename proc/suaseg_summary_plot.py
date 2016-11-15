import numpy as np
import scipy
import sklearn.cluster
import scipy.spatial.distance
import scipy.stats as spstats
import matplotlib.pyplot as plt
from matplotlib import gridspec

from odml.tools.xmlparser import XMLWriter, XMLReader

from suaseg import find_filenames, load_task, identify_trial_time_ranges


if __name__ == "__main__":
    # file information
    odmldir = "."
    savedir = "."

    # plot parameters
    bin_size = 5.0
    bin_step = 2.5

    # savefig = True
    savefig = False

    from suaseg_conf import *

    for sbj, sess, rec, blk, site in datasets:
        dataset_name = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
        odml_name = "{}_rec{}_blk{}_{}_h_SUA".format(sess, rec, blk, site)
        filename_odml = "{}/{}.odml".format(odmldir, odml_name)
        metadata = XMLReader().fromFile(filename_odml)

        rpv_threshold = metadata["Dataset"]["SpikeData"].properties["RPVThreshold"].value.data * 1000

        # load spike data and extract necessary information
        filename_class = metadata["Dataset"]["SpikeData"].properties["File"].value.data
        print filename_class
        print "\tLoading spike data..."
        spike_data = np.genfromtxt(filename_class, skip_header=2, dtype=None, names=True)
        print "\t...done.\n"
        spike_times = spike_data['event_time']
        spike_covs = np.array([spike_data["ch{}".format(i_ch)] for i_ch in range(params["NumChannels"])])
        spike_types = spike_data['type']

        # load task event data and extract necessary information
        filename_task = find_filenames(taskdir, sbj, sess, rec, 'task')[0]
        print "\tLoading task event data..."
        task_events, task_params = load_task(filename_task, blk)
        print "\t...done.\n"
        # time stamps in the task data file are relative to the beginning of the recording. Here the onset time of the
        # block is subtracted from the time stamps so that they are relative to the beginning of the block.
        task_events["evtime"] -= task_events["evtime"][0]
        block_dur = task_events["evtime"][-1] / params["SamplingRate"]
        trial_time_ranges = identify_trial_time_ranges(task_events, task_params, params["SamplingRate"])

        unitIDs = [x.data for x in metadata["Dataset"]["SpikeData"].properties["UnitIDs"].values]
        for unitID in unitIDs:
            mask_unit = (spike_types == unitID)
            unit_label = "Unit{}".format(unitID)
            sect_unit = metadata["Dataset"]["SpikeData"][unit_label]
            unit_ch = sect_unit.properties["Channel"].value.data
            num_spike = sect_unit.properties["NumSpikes"].value.data
            num_trial = sect_unit.properties["NumTrials"].value.data
            num_period = sect_unit.properties["NumPeriods"].value.data

            # compute time resolved firing rate
            spike_times_unit = spike_times[mask_unit]
            spike_covs_unit = spike_covs[:, mask_unit]
            bin_times = np.arange(spike_times_unit[0] + bin_size / 2, spike_times_unit[-1], bin_step)
            firing_rates = np.empty(bin_times.size)
            for i, t_ini in enumerate(bin_times):
                idxs_spikes_in_bin = (t_ini-bin_size/2 <= spike_times_unit) & (spike_times_unit < t_ini+bin_size/2)
                firing_rates[i] = (idxs_spikes_in_bin).sum() / bin_size

            # organize axes
            fig = plt.figure(figsize=(10, 8))
            fig.subplots_adjust(left=0.08, right=0.96)
            title = "{}:{}".format(sbj, dataset_name)
            title += "\nUnit {} (Ch {}, {} spikes, {} trials)".format(unitID, unit_ch, num_spike, num_trial)
            fig.suptitle(title)
            gs = gridspec.GridSpec(5, 2, width_ratios=[4, 1])
            ax_covs = fig.add_subplot(gs[2])
            ax_covs.set_xlabel("Time (s)")
            ax_covs.set_ylabel("Spike size (cov)")
            ax_covs_hist = fig.add_subplot(gs[3])
            ax_covs_hist.set_xlabel("Count")
            ax_covs_hist.set_ylabel("Spike size (cov)")
            ax_periods = fig.add_subplot(gs[4])
            ax_periods.set_xlabel("Time (s)")
            ax_periods.set_ylabel("Firing rate (1/s)")
            ax_periods2 = ax_periods.twinx()
            ax_periods2.set_xlabel("Time (s)")
            ax_periods2.set_ylabel("Unimodality")
            ax_segs = fig.add_subplot(gs[6])
            ax_segs.set_xlabel("Time (s)")
            ax_segs.set_ylabel("Firing rate (1/s)")
            ax_segs2 = ax_segs.twinx()
            ax_segs2.set_xlabel("Time (s)")
            ax_segs2.set_ylabel("RPV")
            ax_isis = fig.add_subplot(gs[8])
            ax_isis.set_xlabel("Time (s)")
            ax_isis.set_ylabel("ISI (ms)")
            ax_isis_hist = fig.add_subplot(gs[9])
            ax_isis_hist.set_xlabel("Count")
            ax_isis_hist.set_ylabel("ISI (log10(ms))")

            # make plots
            # plt.subplot(gs[0])
            # plt.xlabel("Time (s)")
            # plt.ylabel("Channel")
            # X, Y = np.meshgrid(
            #     timebin_times,
            #     np.linspace(-0.5, num_ch-0.5, num_ch+1)
            # )
            # vmax = np.abs(cov_means).max()
            # plt.pcolormesh(X, Y, cov_means, vmax=vmax, vmin=-vmax, cmap="bwr")
            # plt.xlim(0, recdur)
            # plt.ylim(23, 0)
            # plt.grid(color="gray")
            # # plt.colorbar().set_label("Waveform-template covariance")

            ax_covs.plot(spike_times_unit, spike_covs_unit[unit_ch], "k,")
            for t_ini, t_fin in trial_time_ranges:
                ax_covs.axvspan(t_ini, t_fin, color='green', alpha=0.1, linewidth=0)
            ax_covs.set_xlim(0, block_dur)
            ax_covs.set_ylim(0, 200)
            ax_covs.grid(color="gray")
            ax_covs_hist.hist(spike_covs[unit_ch, mask_unit], bins=200, range=[0, 200], orientation="horizontal", linewidth=0, color="black")
            ax_covs_hist.grid(color="gray")

            ax_periods.plot(bin_times, firing_rates, 'k-')
            for t_ini, t_fin in trial_time_ranges:
                ax_periods.axvspan(t_ini, t_fin, color='green', alpha=0.1, linewidth=0)
            ax_periods.set_xlim(0, block_dur)
            ax_periods.set_ylim(ymin=0)
            ax_periods.grid(color="gray")
            ax_periods2.axhline(y=1, color="blue", linestyle=":")

            ax_segs.plot(bin_times, firing_rates, 'k-')
            for t_ini, t_fin in trial_time_ranges:
                ax_segs.axvspan(t_ini, t_fin, color='green', alpha=0.1, linewidth=0)
            ax_segs.set_xlim(0, block_dur)
            ax_segs.set_ylim(ymin=0)
            ax_segs.grid(color="gray")
            ax_segs2.axhline(y=0.01, color="blue", linestyle=":")

            for t_ini, t_fin in trial_time_ranges:
                ax_isis.axvspan(t_ini, t_fin, color='green', alpha=0.1, linewidth=0)
            isis = np.diff(spike_times_unit) * 1000
            ax_isis.plot(spike_times_unit[:-1], isis, "k,")
            ax_isis.axhline(rpv_threshold, color="red", linestyle=':')
            ax_isis.set_yscale('log')
            ax_isis.set_xlim(0, block_dur)
            ax_isis.set_ylim(0.1, 1000)
            ax_isis.grid(color="gray")
            ax_isis_hist.hist(np.log10(isis), bins=50, range=[np.log10(0.1), np.log10(1000)], orientation="horizontal", linewidth=0, color="black")
            ax_isis_hist.axhline(np.log10(rpv_threshold), color="red", linestyle=':')
            ax_isis_hist.set_ylim(np.log10(0.1), np.log10(1000))
            ax_isis_hist.grid(color="gray")

            periodIDs = [] if num_period ==0 else [x.data for x in sect_unit.properties["PeriodIDs"].values]
            for periodID in periodIDs:
                period_label = "Period{}".format(periodID)
                sect_period = metadata["Dataset"]["SpikeData"][unit_label][period_label]
                t_ini = sect_period.properties["Start"].value.data
                t_fin = sect_period.properties["End"].value.data
                unimodality = sect_period.properties["MeanUnimodality"].value.data
                num_seg = sect_period.properties["NumSegments"].value.data
                mask_period = mask_unit & (t_ini <= spike_times) & (spike_times <= t_fin)

                ax_periods2.plot([t_ini, t_fin], [unimodality, unimodality], 'b-')
                plotcolor = 'red' if periodID >= 0 else 'blue'
                ax_periods2.axvspan(t_ini, t_fin, color=plotcolor, alpha=0.1)
                ax_periods2.set_xlim(0, block_dur)
                ax_periods2.set_ylim(0, 2)

                segIDs = [] if num_seg == 0 else [x.data for x in sect_period.properties["SegmentIDs"].values]
                for segID in segIDs:
                    seg_label = "Segment{}".format(segID)
                    sect_seg = metadata["Dataset"]["SpikeData"][unit_label][period_label][seg_label]
                    t_ini = sect_seg.properties["Start"].value.data
                    t_fin = sect_seg.properties["End"].value.data
                    rpv = sect_seg.properties["RPV"].value.data
                    mask_seg = mask_period & (t_ini <= spike_times) & (spike_times <= t_fin)

                    ax_segs2.plot([t_ini, t_fin], [rpv, rpv], 'b-')
                    plotcolor = 'red' if segID >= 0 else 'blue'
                    ax_segs2.axvspan(t_ini, t_fin, color=plotcolor, alpha=0.1)
                    ax_segs2.set_xlim(0, block_dur)
                    ax_segs2.set_ylim(0, 0.1)

            plt.show()



        #
        #         # make plots
        #         plt.subplot(gs[0])
        #         plt.xlabel("Time (s)")
        #         plt.ylabel("Channel")
        #         X, Y = np.meshgrid(
        #             timebin_times,
        #             np.linspace(-0.5, num_ch-0.5, num_ch+1)
        #         )
        #         vmax = np.abs(cov_means).max()
        #         plt.pcolormesh(X, Y, cov_means, vmax=vmax, vmin=-vmax, cmap="bwr")
        #         plt.xlim(0, recdur)
        #         plt.ylim(23, 0)
        #         plt.grid(color="gray")
        #         # plt.colorbar().set_label("Waveform-template covariance")
        #
        #         plt.subplot(gs[2])
        #         plt.xlabel("Time (s)")
        #         plt.ylabel("Spike size (cov)")
        #         plt.plot(spike_times, spike_covs[unit_ch], ",", color="black")
        #         for i_blk, b in enumerate(blks):
        #             if b == 0:
        #                 continue
        #             plt.axvspan(ts_blk_on[i_blk], ts_blk_off[i_blk], color=colors_task[tasks_blk[i_blk]], alpha=0.1, linewidth=0)
        #         for t_ini, t_fin in zip(ts_img_on, ts_img_off):
        #         # for i in range(idx_unit_trial_ini, idx_unit_trial_fin+1):
        #         #     t_ini = ts_img_on[i]
        #         #     t_fin = ts_img_off[i]
        #             plt.axvspan(t_ini, t_fin, color=colors_task[tasks_blk[blk]], alpha=0.1, linewidth=0)
        #         plt.xlim(0, recdur)
        #         plt.ylim(0, 200)
        #         plt.grid(color="gray")
        #         plt.subplot(gs[3])
        #         plt.xlabel("Count")
        #         plt.ylabel("Spike size (cov)")
        #         plt.hist(spike_covs[unit_ch], bins=200, range=[0, 200], orientation="horizontal", linewidth=0, color="black")
        #         plt.grid(color="gray")
        #
        #         ax1 = plt.subplot(gs[4])
        #         plt.xlabel("Time (s)")
        #         plt.ylabel("Firing rate (1/s)")
        #         plt.plot(trial_times[idx_unit_trial_ini:idx_unit_trial_fin+1], trial_firing_rates[idx_unit_trial_ini:idx_unit_trial_fin+1], color="black", marker="+")
        #         plt.xlim(0, recdur)
        #         plt.ylim(ymin=0)
        #         plt.grid(color="gray")
        #         ax2 = ax1.twinx()
        #         plt.ylabel("Surprise")
        #         plt.plot(bin_times_pval, np.log10((1.0 - fr_pvals) / fr_pvals), color="magenta")
        #         plt.axhline(y=0, color="magenta", linestyle=":")
        #         plt.xlim(0, recdur)
        #         plt.ylim(-10, 10)
        #
        #         ax1 = plt.subplot(gs[6])
        #         plt.xlabel("Time (s)")
        #         plt.ylabel("Firing rate (1/s)")
        #         plt.plot(timebin_times, firing_rates, color="black")
        #         # for segID, seg_edge in seg_edges.items():
        #         for segID, [t_ini, t_fin] in seg_time_ranges.items():
        #             seg_color = "blue" if segID >= 0 else "red"
        #             plt.axvspan(t_ini, t_fin, color=seg_color, alpha=0.1, linewidth=0)
        #         plt.xlim(0, recdur)
        #         plt.ylim(ymin=0)
        #         plt.grid(color="gray")
        #         # ax2 = ax1.twinx()
        #         # plt.plot(bin_times_pval[segIDs >= 0], segIDs[segIDs >= 0], 'bo')
        #         # plt.plot(bin_times_pval[segIDs < 0], segIDs[segIDs < 0], 'ro')
        #         # plt.xlabel("Time (s)")
        #         # plt.ylabel("Segment ID")
        #         # plt.xlim(t_blk_on, t_blk_off)
        #
        #         plt.subplot(gs[8])
        #         plt.xlabel("Time (s)")
        #         plt.ylabel("ISI (ms)")
        #         isis = np.diff(spike_times) * 1000
        #         plt.plot(spike_times[:-1], isis, ",", color="black")
        #         plt.axhline(rpv_threshold, color="red")
        #         for i_blk, b in enumerate(blks):
        #             if b == 0:
        #                 continue
        #             plt.axvspan(ts_blk_on[i_blk], ts_blk_off[i_blk], color=colors_task[tasks_blk[i_blk]], alpha=0.1, linewidth=0)
        #         for t_ini, t_fin in zip(ts_img_on, ts_img_off):
        #         # for i in range(idx_unit_trial_ini, idx_unit_trial_fin+1):
        #         #     t_ini = ts_img_on[i]
        #         #     t_fin = ts_img_off[i]
        #             plt.axvspan(t_ini, t_fin, color=colors_task[tasks_blk[blk]], alpha=0.1, linewidth=0)
        #         plt.yscale('log')
        #         plt.xlim(0, recdur)
        #         plt.ylim(0.1, 1000)
        #         plt.grid(color="gray")
        #         plt.subplot(gs[9])
        #         plt.xlabel("Count")
        #         plt.ylabel("ISI (log10(ms))")
        #         plt.hist(np.log10(isis), bins=200, range=[np.log10(0.1), np.log10(1000)], orientation="horizontal", linewidth=0, color="black")
        #         plt.axhline(np.log10(rpv_threshold), color="red")
        #         plt.ylim(np.log10(0.1), np.log10(1000))
        #         plt.grid(color="gray")
        #
        #         if savefig:
        #             fn_fig = "{}/{}_unit{}.png".format(savedir, fn_spikes, unitID)
        #             plt.savefig(fn_fig)
        #             print "\tFigure saved as {}\n".format(fn_fig)
        #             plt.close("all")
        #         else:
        #             plt.show()
