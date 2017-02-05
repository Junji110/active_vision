import datetime

import numpy as np
import scipy
import sklearn.cluster
import scipy.spatial.distance
import scipy.stats as spstats

import odml
import utils


class odMLFactory(object):
    def __init__(self, section_info={}, default_props={}, filename='', strict=True):
        self.sect_info = section_info
        self.def_props = default_props
        self.strict = strict
        if filename:
            self._sections = self.__get_sections_from_file(filename)
        else:
            self._sections = {}
            for sectname in self.__get_top_section_names():
                self._sections[sectname] = self.__gen_section(sectname)

    def __get_sections_from_file(self, filename):
        # load odML from file
        with open(filename, 'r') as fd_odML:
            metadata = odml.tools.xmlparser.XMLReader().fromFile(fd_odML)
        sections = {}
        for sect in metadata.sections:
            sections[sect.name] = sect
        return sections

    def __get_top_section_names(self):
        topsectnames = []
        for key in self.sect_info:
            if '/' not in key:
                topsectnames.append(key)
        return topsectnames

    def __add_property(self, sect, prop, strict=True):
        if sect.contains(odml.Property(prop['name'], None)):
            sect.remove(sect.properties[prop['name']])
        elif strict is True:
            raise ValueError("Property '{0}' does not exist in section '{1}'.".format(prop['name'], sect.name))
        sect.append(odml.Property(**prop))

    def __gen_section(self, name, parent=''):
        longname = parent + name
        sect = odml.Section(name=name, type=self.sect_info[longname]['type'])

        # add properties
        if longname in self.def_props:
            for prop in self.def_props[longname]:
                self.__add_property(sect, prop, strict=False)

        # add subsections
        if 'subsections' in self.sect_info[longname]:
            for subsectname in self.sect_info[longname]['subsections']:
                sect.append(self.__gen_section(subsectname, longname+'/'))

        return sect

    def __get_section_from_longname(self, sectname):
        def get_subsect(sect, names):
            if len(names) == 0:
                return sect
            else:
                return get_subsect(sect.sections[names[0]], names[1:])

        names = sectname.split('/')
        if names[0] not in self._sections:
            return None
        else:
            return get_subsect(self._sections[names[0]], names[1:])

    def put_values(self, properties):
        for sectname, sectprops in properties.items():
            sect = self.__get_section_from_longname(sectname)
            if sect is None:
                raise ValueError("Invalid section name '{0}'".format(sectname))
            else:
                for prop in sectprops:
                    self.__add_property(sect, prop, self.strict)

    def get_odml(self, author, version=None):
        metadata = odml.Document(author, datetime.date.today(), version)
        for sect in self._sections.values():
            metadata.append(sect)
        return metadata

    def save_odml(self, filename, author, version=None):
        metadata = self.get_odml(author, version)
        odml.tools.xmlparser.XMLWriter(metadata).write_file(filename)


def print_metadata(metadata):
    def print_section(sect, ntab=0, tabstr='    '):
        tabs = tabstr * ntab
        print("{0}{1} (type: {2})".format(tabs, sect.name, sect.type))
        tabs = tabstr * (ntab + 1)
        for prop in sect.properties:
            if isinstance(prop.value, list):
                data = [str(x.data) for x in prop.value]
                unit = "" if prop.value[0].unit is None else prop.value[0].unit
                print("{0}{1}: [{2}] {3} (dtype: {4})".format(tabs, prop.name, ', '.join(data), unit, prop.value[0].dtype))
            else:
                unit = "" if prop.value.unit is None else prop.value.unit
                print("{0}{1}: {2} {3} (dtype: {4})".format(tabs, prop.name, prop.value.data, unit, prop.value.dtype))
        print

        for subsect in sect.sections:
            print_section(subsect, ntab+1)

    print("Version {0}, Created by {1} on {2}".format(metadata.version, metadata.author, metadata.date))
    print
    for sect in metadata.sections:
        print_section(sect)
        print


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


def compute_properties(spike_times, spike_covs, rpv_threshold):
    num_spike = spike_times.size
    start_time = spike_times[0]
    end_time = spike_times[-1]
    isis = np.diff(spike_times)
    rpv = (isis < rpv_threshold).sum() / np.float(isis.size)
    unit_ch = spike_covs.mean(1).argmax()
    return unit_ch, num_spike, start_time, end_time, rpv


if __name__ == "__main__":
    # file information
    datasetdir = "/home/ito/datasets/osaka"
    rawdir = "{}/RAWDATA".format(datasetdir)
    prepdir = "{}/PREPROCESSED".format(datasetdir)
    savedir = "."

    # analysis parameters
    sampling_rate = 20000.0
    bin_size = 15  # bin width in number of trials
    bin_step = 1  # bin step in number of trials
    fdr_q = 0.05
    rpv_threshold = 0.001  # refractory period voilation threshold in sec

    # session information
    datasets = [
        # format: [sbj, sess, rec, blk, site, cluster_type]
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
        dataset_name = "{}_rec{}_blk{}_{}_h".format(sess, rec, blk, site)
        print "\n{sbj}:{dataset_name} ({cluster_type})".format(**locals())

        # set filenames
        # fn_class = "{dir}/{sbj}/spikes/24ch_unselected/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=fn_spikes, typ=cluster_type)
        fn_class = "{dir}/tmp/new/{fn}.class_{typ}Cluster".format(dir=prepdir, sbj=sbj, fn=dataset_name, typ=cluster_type)
        fn_task = utils.find_filenames(rawdir, sbj, sess, rec, 'task')[0]

        # load task events
        print "\tLoading task data file..."
        task_events, task_param = utils.load_task(fn_task)
        print "\t...done.\n"
        blks = np.unique(task_events['block'])
        ts_blk_on = []
        ts_blk_off = []
        tasks_blk = []
        for b in blks:
            blk_events, blk_param = utils.load_task(fn_task, b)
            blk_events['evtime'] = blk_events['evtime'] / sampling_rate
            task_blk = blk_param['task'][0]
            tasks_blk.append(task_blk)
            ts_blk_on.append(blk_events["evtime"][0])
            ts_blk_off.append(blk_events["evtime"][-2])
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

                    ts_img_on.append(trial_events['evtime'][trial_events['evID'] == evID_img_on][0])
                    ts_img_off.append(trial_events['evtime'][trial_events['evID'] == evID_img_off][0])
                ts_img_on = np.array(ts_img_on)
                ts_img_off = np.array(ts_img_off)
                num_trial = len(ts_img_on)
        recdur = ts_blk_off[-1]

        section_info = {
            "Dataset": {"name": "Dataset", "type": "dataset", "subsections": ["SpikeData",]},
            "Dataset/SpikeData": {"name": "SpikeData", "type": "dataset/neural_data", "subsections": ["Block{}".format(blk)]},
        }
        props = {"Dataset/SpikeData": [
            {"name": "NumBlocks", "value": 1, "unit": "", "dtype": "int"},
            {"name": "BlockIDs", "value": [blk,], "unit": "", "dtype": "int"},
        ]
        }


        # record odML information
        sectname_blk = "Dataset/SpikeData/Block{}".format(blk)
        section_info[sectname_blk] = {
            "name": "Block{}".format(blk), "type": "collection/experimental_block", "subsections": []}
        props[sectname_blk] = [
            {"name": "File", "value": fn_class, "unit": "", "dtype": "string"},
            {"name": "SamplingRate", "value": sampling_rate, "unit": "Hz", "dtype": "float"},
            {"name": "FDR-q", "value": fdr_q, "unit": "", "dtype": "float"},
            {"name": "TrialBinSize", "value": bin_size, "unit": "", "dtype": "int"},
            {"name": "TrialBinStep", "value": bin_step, "unit": "", "dtype": "int"},
            {"name": "RPVThreshold", "value": rpv_threshold, "unit": "", "dtype": "float"},
        ]

        # load data
        print "\tLoading spike data file..."
        dataset = np.genfromtxt(fn_class, skip_header=2, dtype=None, names=True)
        print "\t...done.\n"

        dataset["event_time"] += ts_blk_on[blk]
        if cluster_type == "Demerged":
            num_ch = len(dataset.dtype.names) - 4
        else:
            num_ch = len(dataset.dtype.names) - 3

        unitIDs = np.unique(dataset["type"])
        unitIDs_processed = []
        num_unit_processed = 0
        for unitID in unitIDs:
            mask_unit = (dataset["type"] == unitID)

            periodIDs = np.unique(dataset["subtype"][mask_unit]) if cluster_type == "Demerged" else [0, ]
            periodIDs_processed = []
            num_period_processed = 0
            for periodID in periodIDs:
                if cluster_type is "Demerged":
                    mask_period = mask_unit & (dataset["subtype"] == periodID)
                    unit_name = "{}({})".format(unitID, periodID)
                else:
                    mask_period = mask_unit
                    unit_name = "{}".format(unitID)
                num_spike = mask_period.sum()
                if num_spike < num_trial:
                    print "\tUnit {} has too few spikes in block {} ({} spikes).\n".format(unit_name, blk, num_spike)
                    continue

                print "\tProcessing unit {} ({} spikes)...".format(unit_name, num_spike)

                spike_times = dataset["event_time"][mask_period]

                # compute trial resolved measures
                trial_firing_rates = np.empty(num_trial)
                trial_times = np.empty(num_trial)
                for i_trial, (t_ini, t_fin) in enumerate(zip(ts_img_on, ts_img_off)):
                    mask_trial = (t_ini <= spike_times) & (spike_times < t_fin)
                    trial_firing_rates[i_trial] = mask_trial.sum() / (t_fin-t_ini)
                    trial_times[i_trial] = (t_ini + t_fin) / 2
                if np.all(trial_firing_rates == 0):
                    print "\tUnit {} is not active in any trial.\n".format(unit_name)
                    continue

                idx_unit_trial_ini, idx_unit_trial_fin = np.where(trial_firing_rates > 0)[0][[0, -1]]
                # idx_unit_trial_fin = np.where(trial_firing_rates > 0)[0][-1]
                num_active_trial = idx_unit_trial_fin - idx_unit_trial_ini + 1
                if num_active_trial <= 2*bin_size:
                    print "\tUnit {} is active in too few trials ({} trials).\n".format(unit_name, num_active_trial)
                    continue

                bin_edges = np.arange(idx_unit_trial_ini, idx_unit_trial_fin - 2*bin_size + 1, bin_step)
                num_bin = bin_edges.size
                bin_times_pval = np.empty(num_bin)
                fr_pvals = np.zeros(num_bin)
                for i, idx_ini in enumerate(bin_edges):
                    fr_pre = trial_firing_rates[idx_ini:idx_ini + bin_size]
                    fr_post = trial_firing_rates[idx_ini + bin_size:idx_ini + 2*bin_size]
                    _, fr_pvals[i] = spstats.ks_2samp(fr_pre, fr_post)
                    bin_times_pval[i] = (trial_times[idx_ini+bin_size-1] + trial_times[idx_ini+bin_size]) / 2

                # define the p-value threshold for FDR with Benjamini-Hochberg method
                pval_thresholds = np.linspace(fdr_q / len(fr_pvals), fdr_q, len(fr_pvals))
                idxs_rejected_pval = np.where(np.sort(fr_pvals) < pval_thresholds)[0]
                pval_threshold = 0 if len(idxs_rejected_pval) == 0 else pval_thresholds[idxs_rejected_pval.max()]

                # identify segments of stable and unstable rate, and assign segment IDs
                # segment ID 0: when no unstable segments are identified, ID 0 is assigned to the whole episode
                # segment ID 1, 2, 3, ...: stable segments with the longest duration, the 2nd longest, and so on
                # segment ID -1, -2, -3, ...: unstable segments with the longest duration, the 2nd longest, and so on
                unstable_segment_edges = segment_successive_occurrences(fr_pvals < pval_threshold, True)
                stable_segment_edges = segment_successive_occurrences(fr_pvals < pval_threshold, False)
                seg_edges = {}
                if len(idxs_rejected_pval) == 0:
                    seg_edges[0] = [0, len(fr_pvals)]
                elif len(idxs_rejected_pval) == len(fr_pvals):
                    seg_edges[-1] = [0, len(fr_pvals)]
                else:
                    # assign negative segment IDs to unstable rate segments
                    segID = -1
                    for i_seg in np.argsort([x[1]-x[0] for x in unstable_segment_edges])[::-1]:
                        seg_edge = unstable_segment_edges[i_seg]
                        seg_edges[segID] = seg_edge
                        segID -= 1
                    # assign positive segment IDs to unstable rate segments
                    segID = 1
                    for i_seg in np.argsort([x[1]-x[0] for x in stable_segment_edges])[::-1]:
                        seg_edge = stable_segment_edges[i_seg]
                        seg_edges[segID] = seg_edge
                        segID += 1
                seg_time_ranges = {}
                for segID, seg_edge in seg_edges.items():
                    t_ini = spike_times[0] if seg_edge[0] == 0 else bin_times_pval[seg_edge[0]]
                    t_fin = spike_times[-1] if seg_edge[1] == len(fr_pvals) else bin_times_pval[seg_edge[1]-1]
                    seg_time_ranges[segID] = [t_ini, t_fin]

                # do segment-wise calculations here
                segIDs = sorted(seg_time_ranges.keys())
                num_seg = len(segIDs)
                segIDs_processed = []
                num_seg_processed = 0
                for segID, (t_ini, t_fin) in seg_time_ranges.items():
                    mask_t_ini = (t_ini <= dataset["event_time"])
                    mask_t_fin = (dataset["event_time"] <= t_fin)
                    mask_seg = mask_period & mask_t_ini & mask_t_fin
                    if mask_seg.sum() < num_trial:
                        continue

                    # compute segment-wise measures
                    spike_times = dataset["event_time"][mask_seg]
                    spike_covs = np.array([dataset["ch{}".format(i_ch)][mask_seg] for i_ch in range(num_ch)])
                    unit_ch, num_spike, start_time, end_time, rpv = compute_properties(spike_times, spike_covs, rpv_threshold)

                    # record odML information
                    sectname_seg = "Dataset/SpikeData/Block{}/Unit{}/Period{}/Segment{}".format(blk, unitID, periodID, segID)
                    section_info[sectname_seg] = {"name": "Segment{}".format(segID), "type": "dataset/neural_data"}
                    props[sectname_seg] = [
                        {"name": "Channel", "value": unit_ch, "unit": "", "dtype": "int"},
                        {"name": "NumSpikes", "value": num_spike, "unit": "", "dtype": "int"},
                        {"name": "Start", "value": t_ini, "unit": "s", "dtype": "float"},
                        {"name": "End", "value": t_fin, "unit": "s", "dtype": "float"},
                        {"name": "RPV", "value": rpv, "unit": "", "dtype": "float"},
                    ]

                    segIDs_processed.append(segID)
                    num_seg_processed += 1

                if num_seg_processed > 0:
                    # compute period-wise measures
                    spike_times = dataset["event_time"][mask_period]
                    spike_covs = np.array([dataset["ch{}".format(i_ch)][mask_unit] for i_ch in range(num_ch)])
                    unit_ch, num_spike, start_time, end_time, rpv = compute_properties(spike_times, spike_covs, rpv_threshold)

                    # record odML information
                    sectname_period = "Dataset/SpikeData/Block{}/Unit{}/Period{}".format(blk, unitID, periodID)
                    subsections = ["Segment{}".format(sid) for sid in segIDs]
                    section_info[sectname_period] = {
                        "name": "Period{}".format(periodID), "type": "dataset/neural_data", "subsections": subsections}
                    props[sectname_period] = [
                        {"name": "Channel", "value": unit_ch, "unit": "", "dtype": "int"},
                        {"name": "NumSpikes", "value": num_spike, "unit": "", "dtype": "int"},
                        {"name": "Start", "value": start_time, "unit": "s", "dtype": "float"},
                        {"name": "End", "value": end_time, "unit": "s", "dtype": "float"},
                        {"name": "RPV", "value": rpv, "unit": "", "dtype": "float"},
                        {"name": "SegmentIDs", "value": segIDs, "unit": "", "dtype": "int"},
                        {"name": "NumSegments", "value": num_seg, "unit": "", "dtype": "int"},
                    ]

                    periodIDs_processed.append(periodID)
                    num_period_processed += 1

                print "\t...done.\n"

            if num_period_processed > 0:
                # compute unit-wise measures
                spike_times = dataset["event_time"][mask_unit]
                spike_covs = np.array([dataset["ch{}".format(i_ch)][mask_unit] for i_ch in range(num_ch)])
                unit_ch, num_spike, start_time, end_time, rpv = compute_properties(spike_times, spike_covs, rpv_threshold)

                # record odML information
                sectname_unit = "Dataset/SpikeData/Block{}/Unit{}".format(blk, unitID)
                subsections = ["Period{}".format(pid) for pid in periodIDs_processed]
                section_info[sectname_unit] = {
                    "name": "Unit{}".format(unitID), "type": "dataset/neural_data", "subsections": subsections}
                props[sectname_unit] = [
                    {"name": "Channel", "value": unit_ch, "unit": "", "dtype": "int"},
                    {"name": "NumSpikes", "value": num_spike, "unit": "", "dtype": "int"},
                    {"name": "Start", "value": start_time, "unit": "s", "dtype": "float"},
                    {"name": "End", "value": end_time, "unit": "s", "dtype": "float"},
                    {"name": "RPV", "value": rpv, "unit": "", "dtype": "float"},
                    {"name": "PeriodIDs", "value": periodIDs_processed, "unit": "", "dtype": "int"},
                    {"name": "NumPeriods", "value": num_period_processed, "unit": "", "dtype": "int"},
                ]

                unitIDs_processed.append(unitID)
                num_unit_processed += 1

        if num_unit_processed > 0:
            subsections = ["Unit{}".format(uid) for uid in unitIDs_processed]
            section_info[sectname_blk]["subsections"] = subsections
            props[sectname_blk].extend([
                {"name": "UnitIDs", "value": unitIDs_processed, "unit": "", "dtype": "int"},
                {"name": "NumUnits", "value": num_unit_processed, "unit": "", "dtype": "int"},
            ])

        for prop in props.values():
            for prop_item in prop:
                del prop_item["unit"]
                del prop_item["dtype"]
        # for key in props:
        #     del props[key]["unit"]
        #     del props[key]["dtype"]

        print section_info.keys()
        odml_factory = odMLFactory(section_info, strict=False)
        odml_factory.put_values(props)
        author = "Junji Ito"
        version = 0.1
        fn_odml = "{}/{}_SUA.odml".format(savedir, dataset_name)
        odml_factory.save_odml(fn_odml, author, version)
        print("SUA metadata saved in {0}".format(fn_odml))

        # print out the odML structure for a check
        print_metadata(odml_factory.get_odml(author, version))
