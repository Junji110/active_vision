from argparse import ArgumentParser

import h5py

# import active_vision
import active_vision.fileio.lvdread as lvdread
import active_vision.utils as avutils


if __name__ == "__main__":
    # --- load parameters from configuration files
    from active_vision.conf.conf_files import projectdir, rawdir
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--lvddir", default=rawdir)
    parser.add_argument("--hdf5dir", default=projectdir)
    parser.add_argument("--sbj", default="HIME")
    parser.add_argument("--data", nargs=2, default=(20140528, 1))
    parser.add_argument("--dsfactor", type=int, default=10)
    arg = parser.parse_args()
    
    sbj = arg.sbj
    sess, rec = arg.data
    dsfactor = arg.dsfactor

    # --- find filenames
    fn_lvd = ['', '', '']
    fn_lvd_found = avutils.find_filenames(arg.lvddir, sbj, sess, rec, 'lvd')
    for i_pc in range(3):
        fn = [x for x in fn_lvd_found if 'pc{0}'.format(i_pc+1) in x]
        if len(fn) > 0:
            fn_lvd[i_pc] = fn[0]
    
    for i_pc, fn_pc in enumerate(fn_lvd):
        if fn_pc is '':
            print "PC{0} data file not found.".format(i_pc + 1)
            continue
        
        # initialize LVD Reader
        reader = lvdread.LVDReader(fn_pc)
        
        # load parameters
        header = reader.get_header()
        param = reader.get_param()
        
        fn_hdf5 = "{0}/{1}_rec{2}_pc{3}.hdf5".format(arg.hdf5dir, sess, rec, i_pc+1)
        with h5py.File(fn_hdf5, 'w') as f:
            grp = f.create_group("/param")
            for key in param:
                grp.create_dataset(key, data=param[key])
            grp.create_dataset("donwsample_factor", data=dsfactor)

            grp = f.create_group("/header")
            for key in header:
                grp.create_dataset(key, data=header[key])

            grp = f.create_group("/data")
            for chname in header['AIUsedChannelName']:
                print "Processing channel {0}...".format(chname)
                data = reader.get_data(channel=chname)
                grp.create_dataset(chname, data=data[0, ::dsfactor])
                print "Channel {0} done.".format(chname)

        print 'PC{0} data file "{1}" converted to "{2}"'.format(i_pc + 1, fn_pc, fn_hdf5)