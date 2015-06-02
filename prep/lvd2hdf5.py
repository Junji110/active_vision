import sys
import argparse

import h5py
import numpy as np

sys.path.append("/users/ito/toolbox")
import active_vision.fileio.lvdread as lvdread
import active_vision.utils as avutils


if __name__ == "__main__":
    # --- load parameters from configuration files
    from active_vision.conf.conf_files import projectdir, rawdir
    
    # parse command line options
    parser = argparse.ArgumentParser()
    parser.add_argument("--lvddir", default=rawdir)
    parser.add_argument("--hdf5dir", default=projectdir)
    parser.add_argument("--sbj", default="HIME")
    parser.add_argument("--data", nargs=3, default=(20140528, 1, 'pc1'))
    parser.add_argument("--dsfactor", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=6000000)
    arg = parser.parse_args()
    
    sbj = arg.sbj
    sess, rec, pc = arg.data
    dsfactor = arg.dsfactor
    chunk_size = arg.chunk_size

    # --- find filenames
    fn_lvd = avutils.find_filenames(arg.lvddir, sbj, sess, rec, 'lvd')
    if len(fn_lvd) == 0:
        raise IOError("LVD data file not found in {0}".format(arg.lvddir))

    for fn in fn_lvd:
        if pc in fn:
            fn_pc = fn
            break
    else:
        raise IOError("{0} data file not found.".format(pc))

    # initialize LVD Reader
    reader = lvdread.LVDReader(fn_pc)

    # load parameters
    header = reader.get_header()
    param = reader.get_param()
    data_length = param['data_length']
    num_chunk = data_length / chunk_size
    chunk_len = chunk_size / dsfactor

    fn_hdf5 = "{0}/{1}_rec{2}_{3}.hdf5".format(arg.hdf5dir, sess, rec, pc)
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
            print "Processing channel {0}...".format(chname)
            chdata = np.empty(data_length / dsfactor)
            for i_chunk in range(num_chunk):
                data_chunk = reader.get_data(channel=chname, samplerange=(i_chunk * chunk_size, (i_chunk + 1) * chunk_size))
                chdata[i_chunk * chunk_len:(i_chunk + 1) * chunk_len] = data_chunk[0][::10]
                print "\tdata chunk {0} of {1} loaded.".format(i_chunk, num_chunk)
            data_chunk = reader.get_data(channel=chname, samplerange=(num_chunk * chunk_size, data_length))
            chdata[num_chunk * chunk_len:] = data_chunk[0][::10]

            grp.create_dataset(chname, data=chdata)
            print "Channel {0} done.".format(chname)

    print 'LVD data file "{0}" converted to "{1}"'.format(fn_pc, fn_hdf5)