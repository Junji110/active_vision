'''
active_vision/prep/lvdread.py

Module for reading the contents of .lvd data files

Written by Richard Meyes (r.meyes@fz-juelich.de) and Junji Ito (j.ito@fz-juelich.de) on 2013.09.26
'''
#import used packages:
import numpy as np


class LVDReader(object):
    float_params = ['AISampleRate', 'InputRangeLow', 'InputRangeHigh', 'SensorRangeLow', 'SensorRangeHigh']
    data_point_size = 8
    
    def __init__(self, file_path):
        self.file_path = file_path
        with open(self.file_path, 'rb') as fd:
            fd.seek(0, 2)
            self.file_size = fd.tell()
        self.header, self.header_size = self.__parse_header()
        self.param = self.__gen_param()
    
    def __extract_header_string(self):
        with open(self.file_path, 'rb') as fd:
            header_len = np.fromfile(fd, '>u2', 2)[1] # first 4 bytes (2 x uint16, big endian) give the length of the header in bytes.
            header_str = fd.read(header_len)
        return header_str, header_len + 4
    
    def __parse_header(self):
        hd_str, hd_size = self.__extract_header_string()
        hd_token = hd_str.split(';')
        hd = {}
        for token in hd_token:
            if token == '': continue
            subtoken = token.split(':')
            if len(subtoken) != 2:
                raise ValueError('Wrongly formatted header string found: "{0}"'.format(token))
            key, value = [x.strip() for x in token.split(':')]
            try:
                hd[key] = eval(value)
            except NameError:
                hd[key] = value
            if key in self.float_params:
                hd[key] = float(hd[key])
        return hd, hd_size
    
    def __gen_param(self):
        param = {}
        # store or derive parameters required for accessing the data structure:
        param['file_size'] = self.file_size
        param['header_size'] = self.header_size
        param['num_channels'] = self.header['AIUsedChannelCount']
        param['chunk_length'] = self.header['AISampleChunk']
        param['sampling_rate'] = self.header['AISampleRate']
        param['num_data'] = (param['file_size'] - param['header_size']) / self.data_point_size
        param['data_length'] = param['num_data'] / param['num_channels']
        param['data_duration'] = float(param['data_length'] / param['sampling_rate'])
        param['block_length'] = param['chunk_length'] * param['num_channels']
        param['num_blocks'] = param['num_data'] / param['block_length']
        param['num_chunks'] = param['num_blocks'] * param['num_channels']
        param['last_chunk_length'] = (param['num_data'] - param['num_blocks'] * param['block_length']) / param['num_channels']
        return param
    
    def __chname2chidx(self, chname):
        try:
            return self.header['AIUsedChannelName'].index(chname)
        except ValueError:
            raise ValueError("Channel name '{0}' does not exist.".format(chname))

    def __fetch_chunk(self, fd, block, channel):
        dps = self.data_point_size
        chunk_len = self.param['chunk_length']
        last_chunk_len = self.param['last_chunk_length']
        header_size = self.param['header_size']
        num_channels = self.param['num_channels']
        num_blocks = self.param['num_blocks']
        
        if block == num_blocks - 1:
            this_chunk_len = last_chunk_len
        else:
            this_chunk_len = chunk_len
        chunk_head = header_size + long(block * chunk_len * num_channels * dps) + channel * chunk_len * dps
        fd.seek(chunk_head, 0)
        return np.fromfile(fd, '>d', this_chunk_len)
                
    def __fetch_channel_data(self, channel, samplerange):
        chunk_len = self.param['chunk_length']
        
        start_block = samplerange[0] / chunk_len
        start_point_in_chunk = samplerange[0] % chunk_len
        end_block = samplerange[1] / chunk_len
        end_point_in_chunk = samplerange[1] % chunk_len
        num_blocks = end_block - start_block + 1
        
        # read data in the unit of chunks
        data = np.empty(num_blocks * chunk_len)
        with open(self.file_path, 'rb') as fd:
            for i, blk in enumerate(xrange(start_block, end_block + 1)):
                chunk = self.__fetch_chunk(fd, blk, channel)
                data[i * chunk_len : i * chunk_len + len(chunk)] = chunk
        
        # only return the data for the requested range
        return data[start_point_in_chunk : (num_blocks - 1) * chunk_len + end_point_in_chunk]
        
    def __fetch_data(self, channel, samplerange):
        if samplerange[0] < 0 or samplerange[1] > self.param['data_length']:
            raise ValueError("Sample range must be within [0, {0}]".format(self.param['data_length']))
        
        data = np.empty((len(channel), samplerange[1] - samplerange[0]))
        for i, ch in enumerate(channel):
            data[i] = self.__fetch_channel_data(ch, samplerange)
        
        return data

    def get_header(self):
        return self.header
    
    def get_param(self):
        return self.param

    def get_data(self, channel=None, samplerange=None, timerange=None):
        # parse channel argument:
        # channels can be specified either by index or channel name
        if channel is None:
            # when channel argument is omitted, all the channels are selected.
            chan = range(self.param['num_channels'])
        else:
            if isinstance(channel, list) or isinstance(channel, tuple):
                chan = list(channel)
            else:
                chan = [channel]
                
            # convert channel name string to channel index number
            for i, ch in enumerate(chan):
                if isinstance(ch, str) or isinstance(ch, unicode):
                    chan[i] = self.__chname2chidx(ch)
                else:
                    chan[i] = int(ch)
        
        # parse range argument
        if samplerange is None and timerange is None:
            datarange = [0, self.param['data_length']]
        elif samplerange is None:
            if timerange[0] < 0 or self.param['data_duration'] < timerange[1]:
                raise ValueError("Time range must be within [0, {0}]".format(self.param['data_duration']))
                
            datarange = [int(timerange[0] * self.param['sampling_rate']), int(timerange[1] * self.param['sampling_rate'])]
        else:
            datarange = list(samplerange)
            
        return self.__fetch_data(channel=chan, samplerange=datarange)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
#    file_path = "C:/Users/ito/datasets/osaka/behavior/20130903/20130903_163903_rec1_pc3.lvd"
    file_path = "C:/Users/ito/datasets/osaka/behavior/20130905/20130905_143912_rec1_pc3.lvd"
    
    lvd_reader = LVDReader(file_path)
    print lvd_reader.get_header()
    data = lvd_reader.get_data(channel=['eyecoil_x', 'eyecoil_y'])
    
    # plot for a check
    plt.subplot(121)
    plt.plot(data[0], data[1])
    plt.subplot(222)
    plt.plot(data[0])
    plt.subplot(224)
    plt.plot(data[1])
    plt.show()
