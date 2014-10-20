import sys
from argparse import ArgumentParser

import numpy as np
import h5py

# the next line is not necessary if you have the appended path in your PYTHONPATH
sys.path.append("C:/Users/ito/toolbox")


def load_types(typesfile):
    '''
    Load unit type IDs and channels with non-zero spike amplitudes from .types_SelectedCluster file

        Arguments
        ----------
        typesfile : string
                   Filename of .types_SelectedCluster file to be loaded

        Returns
        -------
        unit_types:      list of strings
                            List of unit type IDs
        unit_channels:   list of lists of integers
                            Lists of channel IDs where non-zero spike amplitudes are recorded. Separate
                            lists are provided for individual units, and the order of the units is
                            identical to unit_types, i.e., unit_channels[0] is a list of channels with
                            non-zero spike amplitude for unit_types[0], unit_channels[1] for
                            unit_types[1], and so forth
    '''
    dataset = np.genfromtxt(typesfile, dtype=None, names=True)

    # store dtype names of the columns for spike amplitudes
    spkamp_names = filter(lambda x: 'Center' in x, dataset.dtype.names)

    # detect and store unit type IDs and channels with non-zero spike amplitudes
    unit_types = []
    unit_channels = []
    for data in dataset:
        unit_types.append(data['type'])
        spkamp = np.array([data[x] for x in spkamp_names])
        unit_channels.append(spkamp.argsort()[:-len(np.nonzero(spkamp)[0])-1:-1].astype('int32').tolist())

    return unit_types, unit_channels

def load_class(classfile, unit_types=None):
    '''
    Load spikes from .class_SelectedCluster file

        Arguments
        ----------
        classfile: string
            Filename of .types_SelectedCluster file to be loaded
        unit_types : list of integers (default: None)
            List of unit type IDs. When this is given, the function returns only spikes of the units
            listed in it. When not given, returns spikes of all the units in the data file

        Returns
        -------
        unit_spikes: list of numpy arrays
            Numpy arrays containing spike times of individial units
        unit_types: list of integers
            List of unit type IDs. When unit_types has been given as an arguments, the same list is
            returned. When not given, the function identifies all the units in the data file and
            returns their unit type IDs.
    '''
    dataset = np.genfromtxt(classfile, skip_header=2, dtype=None, names=True)

    # detect and store unit type IDs if not provided
    if unit_types is None:
        unit_types = np.unique(dataset['type']).tolist()

    # make lists of spikes for individual units
    unit_spikes = []
    for utype in unit_types:
        unit_spikes.append(dataset['event_time'][dataset['type'] == utype])

    return unit_spikes, unit_types


if __name__ == "__main__":
    # load parameters from configuration files
    from active_vision.conf.conf_files import projectdir, prepdir

    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--filename")
    parser.add_argument("--classdir", default=prepdir)
    parser.add_argument("--hdf5dir", default=projectdir)
    parser.add_argument("--sbj", default="HIME")
    arg = parser.parse_args()

    sbj = arg.sbj

    # set filenames
    fn_class = "{dir}/{sbj}/spikes/24ch/{fn}.class_SelectedCluster".format(dir=prepdir, sbj=sbj, fn=arg.filename)
    fn_types = "{dir}/{sbj}/spikes/24ch/{fn}.types_SelectedCluster".format(dir=prepdir, sbj=sbj, fn=arg.filename)

    # load a list of unit IDs (i.e. "types") and lists of channels where units have finite spike amplitudes
    print("Loading unit IDs...")
    unit_types, unit_channels = load_types(fn_types)
    print("Done.")

    # load spike times of each unit
    print("Loading spike times...")
    unit_spikes, _ = load_class(fn_class, unit_types)
    print("Done.")

    # save results
    fn_hdf5 = "{dir}/data/{fn}.hdf5".format(dir=arg.hdf5dir, fn=arg.filename)
    with h5py.File(fn_hdf5, 'w') as f:
        f.create_dataset("unitIDs", data=unit_types)
        for i_type, type in enumerate(unit_types):
            f.create_dataset("unit{0}_spikes".format(type), data=np.array(unit_spikes[i_type]))
            f.create_dataset("unit{0}_channels".format(type), data=unit_channels[i_type])
