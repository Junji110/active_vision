from argparse import ArgumentParser

import numpy as np
import h5py

# import active_vision
import active_vision.fileio.lvdread as lvdread
import active_vision.utils as avutils

def load_data_in_chunks(reader, dsfactor=10, chunksize=5000000, verbose=False):
    param = reader.get_param()
    data_length = param['data_length']
    if chunksize >= data_length:
        chunksize = (data_length / dsfactor) * dsfactor
    num_chunk = data_length / chunksize

    data = np.empty((num_ch, data_length / dsfactor))
    for i_chunk in range(num_chunk):
        idx_ini = i_chunk * chunksize
        idx_fin = (i_chunk + 1) * chunksize
        data_orig = reader.get_data(samplerange=(idx_ini, idx_fin))
        data[:, idx_ini/dsfactor:idx_fin/dsfactor] = data_orig[:, ::dsfactor]
        if verbose:
            print "Chunk {0} of {1} done.".format(i_chunk+1, num_chunk)
    idx_ini = num_chunk * chunksize
    idx_fin = data_length
    data_orig = reader.get_data(samplerange=(idx_ini, idx_fin))
    data[:, idx_ini/dsfactor:idx_fin/dsfactor] = data_orig[:, ::dsfactor]
    
    return data

if __name__ == "__main__":
    # --- load parameters from configuration files
    from active_vision.conf.conf_files import projectdir, rawdir
    
    # parse command line options
    parser = ArgumentParser()
    parser.add_argument("--lvddir", default=rawdir)
    parser.add_argument("--hdf5dir", default=projectdir)
    parser.add_argument("--sbj", default="HIME")
    parser.add_argument("--data", nargs=2, default=(20140529, 6))
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

        # initialize LVD Readers
        reader = lvdread.LVDReader(fn_pc)
        
        # load parameters
        header = reader.get_header()
        param = reader.get_param()
        num_ch = header['AIUsedChannelCount']
        Fs_orig = header['AISampleRate']
        data_length = param['data_length']
        Fs = Fs_orig / dsfactor
        
        # load data from lvd
        wideband = load_data_in_chunks(reader, dsfactor=dsfactor, chunksize=5000000, verbose=True)

        # save data in hdf5
        fn_hdf5 = "{0}/{1}_rec{2}_pc{3}.hdf5".format(arg.hdf5dir, sess, rec, i_pc+1)
        with h5py.File(fn_hdf5, 'w') as f:
            f.create_dataset("wideband", data=wideband)
            f.create_dataset("Fs", data=Fs)
            f.create_dataset("Fs_orig", data=Fs_orig)
            f.create_dataset("file", data=fn_pc)
            f.create_dataset("downsample_factor", data=dsfactor)
        print 'PC{0} data file "{1}" converted to "{2}"'.format(i_pc + 1, fn_pc, fn_hdf5)
