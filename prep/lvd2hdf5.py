import sys
import argparse

import h5py

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
    arg = parser.parse_args()
    
    sbj = arg.sbj
    sess, rec, pc = arg.data
    dsfactor = arg.dsfactor

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
            data = reader.get_data(channel=chname)
            grp.create_dataset(chname, data=data[0, ::dsfactor])
            print "Channel {0} done.".format(chname)

    print 'LVD data file "{0}" converted to "{1}"'.format(fn_pc, fn_hdf5)