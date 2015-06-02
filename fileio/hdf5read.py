'''
active_vision/prep/lvdread.py

Module for reading the contents of .lvd data files

Written by Richard Meyes (r.meyes@fz-juelich.de) and Junji Ito (j.ito@fz-juelich.de) on 2013.09.26
'''
#import used packages:
import h5py

import numpy as np


class HDF5Reader(object):
    float_params = ['AISampleRate', 'InputRangeLow', 'InputRangeHigh', 'SensorRangeLow', 'SensorRangeHigh']
    data_point_size = 8
    
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(self.file_path, 'r') as f:
            self.header = {}
            for key in f['header']:
                self.header[key] = f['header'][key][...]
            self.param = {}
            for key in f['param']:
                self.param[key] = f['param'][key][...]
                
    def __parse_channel_argument(self, channel):
        # parse channel argument:
        # channels can be specified either by index or channel name
        if channel is None:
            # when channel argument is omitted, all the channels are selected.
            chan = self.header['AIUsedChannelName']
        else:
            if isinstance(channel, list) or isinstance(channel, tuple):
                chan = list(channel)
            else:
                chan = [channel]
                
        # convert channel index number to channel name string
        for i, ch in enumerate(chan):
            if isinstance(ch, int):
                chan[i] = self.header['AIUsedChannelName'][ch]
        
        return chan

    def __parse_range_argument(self, samplerange, timerange):
        # parse range argument
        Fs = self.param['sampling_rate'] / self.param['downsample_factor']
#         Fs = self.param['sampling_rate'] / self.param['donwsample_factor']

        if samplerange:
            datarange = list(samplerange)
        elif timerange:
            if timerange[0] < 0 or self.param['data_duration'] < timerange[1]:
                raise ValueError("Time range must be within [0, {0}]".format(self.param['data_duration']))
            datarange = [int(timerange[0] * Fs), int(timerange[1] * Fs)]
        else:
            datarange = [0, int(self.param['data_duration'] * Fs)]
        
        return datarange
            
    def get_header(self):
        return self.header
    
    def get_param(self):
        return self.param

    def get_data(self, channel=None, samplerange=None, timerange=None):
        channelnames = self.__parse_channel_argument(channel)
        datarange = self.__parse_range_argument(samplerange, timerange)
        
        data = np.zeros((len(channelnames), datarange[1] - datarange[0]))
        with h5py.File(self.file_path, 'r') as f:
            for i_ch, chname in enumerate(channelnames):
                data_in_range = f['data'][chname][datarange[0]:datarange[1]]
                data[i_ch][:len(data_in_range)] = data_in_range

        return data


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import active_vision.fileio.lvdread as lvdread
    
    hdf5_path = "C:/Users/ito/datasets/osaka/RAWDATA/HIME/20140528/20140528_rec1_pc1.hdf5"
    lvd_path = "C:/Users/ito/datasets/osaka/RAWDATA/HIME/20140528/20140528_94701_rec1_pc1.lvd"
    
    hdf5_reader = HDF5Reader(hdf5_path)
    print hdf5_reader.get_header()
    print hdf5_reader.get_param()
#     data_hdf5 = hdf5_reader.get_data(channel=[0,], samplerange=(10000, 10100))
    data_hdf5 = hdf5_reader.get_data(channel=[0,], timerange=(1000, 1010))
    
    lvd_reader = lvdread.LVDReader(lvd_path)
    print lvd_reader.get_header()
    print lvd_reader.get_param()
#     data_lvd = lvd_reader.get_data(channel=[0,], samplerange=(100000, 101000))
    data_lvd = lvd_reader.get_data(channel=[0,], timerange=(1000, 1010))

    # plot for a check
    plt.subplot(211)
    plt.plot(np.arange(len(data_hdf5[0])) * 10, data_hdf5[0])
    plt.plot(data_lvd[0])
    plt.show()
