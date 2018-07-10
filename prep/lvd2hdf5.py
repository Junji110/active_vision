import sys
import argparse

import h5py
import numpy as np

sys.path.append("/users/ito/toolbox")
import active_vision.fileio.lvdread as lvdread
import active_vision.utils as avutils

from parameters.lvd2hdf5 import *


if __name__ == "__main__":
    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvddir", default=datadir_default)
    parser.add_argument("--hdf5dir", default=savedir_default)
    parser.add_argument("--dsfactor", type=int, default=dsfactor_default)
    parser.add_argument("--chunk_size", type=int, default=chunk_size_default)
    arg = parser.parse_args()
    dsfactor = arg.dsfactor
    chunk_size = arg.chunk_size
    datadir = arg.lvddir
    savedir = arg.hdf5dir

    for sbj, sess, rec, pc in datasets:
        dataset_name = "{sbj}:{sess}_rec{rec}_{pc}".format(**locals())
        print "\n{dataset_name} being processed...".format(**locals())

        # --- find filenames
        fn_lvd = avutils.find_filenames(arg.lvddir, sbj, sess, rec, 'lvd', pc)[0]

        # initialize LVD Reader
        reader = lvdread.LVDReader(fn_lvd)

        # load parameters
        header = reader.get_header()
        param = reader.get_param()
        data_length = param['data_length']
        num_chunk = data_length / chunk_size
        chunk_size_ds = chunk_size / dsfactor

        fn_hdf5 = "{savedir}/{sess}_rec{rec}_{pc}.hdf5".format(**locals())
        with h5py.File(fn_hdf5, 'w') as f:
            grp = f.create_group("/param")
            for key in param:
                grp.create_dataset(key, data=param[key])
            grp.create_dataset("downsample_factor", data=dsfactor)

            grp = f.create_group("/header")
            for key in header:
                grp.create_dataset(key, data=header[key])

            grp = f.create_group("/data")
            for chname in header['AIUsedChannelName']:
                print "\t...{chname} being loaded...".format(**locals())
                chdata = np.empty(data_length / dsfactor)
                for i_chunk in range(num_chunk):
                    data_chunk = reader.get_data(channel=chname, samplerange=(i_chunk * chunk_size, (i_chunk+1) * chunk_size))
                    chdata[i_chunk * chunk_size_ds : (i_chunk + 1) * chunk_size_ds] = data_chunk[0][::dsfactor]
                    print "\t\t...data chunk {0} of {1} loaded...".format(i_chunk+1, num_chunk)
                data_chunk = reader.get_data(channel=chname, samplerange=(num_chunk * chunk_size, data_length))
                chdata[num_chunk * chunk_size_ds:] = data_chunk[0][::dsfactor]

                grp.create_dataset(chname, data=chdata)
                print "\t...done.".format(**locals())

        print '...all done. LVD data file "{fn_lvd}" converted to "{fn_hdf5}"'.format(**locals())
