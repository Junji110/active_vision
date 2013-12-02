'''
daqread.py

Module for reading Matlab Data Acquisition Toolbox (.daq) data file.
Based on Matlab script daqread.m (revision 1.17.2.8, Copyright 1998-2004 The
MathWorks, Inc.)

Written by Junji Ito (j.ito@fz-juelich.de) on 2013.04.14
'''
import os
import struct
import datetime
import warnings

import numpy as np


class DAQStruct(object):
    '''
    A blank class that allows dynamic assignment of properties, just like Matlab
    structure does
    '''
    pass

def daqread(filename, retmode='data-info', Samples=None, Time=None, Triggers=None, Channels=None, DataFormat='double', TimeFormat='vector'):
    '''
    Read Data Acquisition Toolbox (.daq) data file and returns a M-by-N data
    matrix, where M specifies the number of samples and N specifies the number
    of channels. If data from multiple triggers is read, the data from each
    trigger is separated by a NaN.
    
    "DATA, TIME, DAQINFO = daqread('filename.daq')" reads the data acquisition
    file, filename.daq, and returns a data matrix, time vector, and recording
    information. TIME is a vector, the same length of DATA indicating the
    relative time of each data sample relative to the first trigger.
    DAQINFO contains the information:

        DAQINFO.ObjInfo:
            a dictionary containing the information of the data acquisition
            object used to create the file.
        DAQINFO.HwInfo:
            a dictionary containing hardware information.

    Arguments
    ---------
    filename: string
        Name of the data acquisition file to be read
    
    retmode: string (default: 'data-info')
        Specifies return mode. The following return modes are implemented.
        
            'data-info':
                data matrix, time vector, and recording information (DAQINFO)
                are returned. This is the default mode.
            'info':
                only recording information is returned
            'data' or 'data1':
                only data matrix is returned
            'data2':
                data matrix and time vector are returned
            'data3':
                in addition to 'data2', absolute time of the first trigger is
                returned
            'data4':
                in addition to 'data3', event log, which is identical to
                DAQINFO['EventLog'], is returned
            'data5':
                in addition to 'data4', recording information (DAQINFO) is
                returned
    
    Samples: 2-integer list
        Specifies the range of sample numbers to be returned
    
    Time: 2-float list
        Specifies the time range of returned data
    
    Triggers: 2-integer list
        Specifies the range of triggers for returned data.
        
    (Samples, Time, and Triggers are mutually exclusive: if multiple of them are
    given, an error is raised.)
    
    Channels: list of int (channel index) or str (channel name)
        Specifies the channels of which data are returnd, either by channel
        index or channel name
    
    dataformat: 'double' or 'native' (default: 'double')
        Type of returned data values. When 'native' is set, channel properties of
        'NativeOffset' and 'NativeScaling' are ignored.
    
    timeformat: 'vector' or 'matrix' (default: 'vector')
        Format of returned time values. When set 'matrix', 'ChannelSkew'
        property of recording is taken into account.
        
    Examples
    --------
    To read all the data from the file, data.daq:
        data, time, daqinfo = daqread('data.daq')

    To read only samples 1000 to 2000 of channel indices 2, 4 and 7 in 
    native format from the file, data.daq:
        data = daqread('data.daq', retmode='data',\
                        Samples=[1000 2000], Channels=[2 4 7], dataformat='native')

    To read only the data which represents the first and second triggers on 
    all channels from the file, data.daq, together with the time vector:
        data, time = daqread('data.daq', retmode='data2', Triggers=[0 2])

    To obtain the property values of the channels from the file, data.daq:
        daqinfo = daqread('data.daq', retmode='info')
        chaninfo = daqinfo.ObjInfo['Channel']
        
    '''
    # check whether data range is specified
    if [Samples, Time, Triggers, Channels] == [None, None, None, None]:
        rangearg = False
    else:
        rangearg = True
    dataformat = DataFormat
    timeformat = TimeFormat
    
    # Error if the first input is not a string.
    if not isinstance(filename, str) and not isinstance(filename, unicode):
        raise ValueError('daq:daqread:invalidfile\nInvalid FILENAME specified to DAQREAD.')

    # Determine if an extension was given.  If not add a .daq.
    path_name, ext = os.path.splitext(filename)
    if not ext:
        filename = path_name + '.daq'

    # Open the specified file.
    try:
        fid = open(filename, 'rb')
    except IOError:
        raise IOError('daq:daqread:invalidfile\nUnable to open specified FILENAME: %s.' % (filename,))

    # Verify that file is DAQ file.
    FileKey = fid.read(32)
    # now check the file key
    if FileKey != "MATLAB Data Acquisition File.\x00\x19\x00":
        fid.close()
        raise IOError('daq:daqread:invalidfileerror\nFILENAME is not a valid Data Acquisition Toolbox file.')

    # Read in the creation time and engine time offset used for time calculations.
    headersize = np.fromfile(fid, 'int32', 1)[0]
    fileVer = np.fromfile(fid, 'int16', 2)  # this variable is never used, but needs to be read in just to proceed the file pointer
    creationTime = np.fromfile(fid, 'double', 6)
    engineOffset = np.fromfile(fid, 'double', 1)[0]

    # Store the headersize - 512.
    pos = headersize

    # Determine the size of the file.
    # pos = Original position of file position indicator.
    fid.seek(0, 2)
    fsize = fid.tell()
    fid.seek(pos, 0)

    # Determine the number of samples logged, the number of blocks 
    # logged and the state of the object at the end of the acquisition.
    samplesacquired, num_of_blocks, info, lastheaderloc, flag = localPreprocessFile(fid, fsize)
    fid.seek(pos, 0)
    
    # If the call to localPreprocessFile failed due to the last block not being 
    # written or an unknown error, initialize the lastheaderloc and num_of_blocks
    # variables.
    if flag:
        # Calculate the number of blocks.
        if not num_of_blocks:
            try:
                tempinfo = readObject(fid, pos);
                num_of_blocks = np.ceil(1.15 * fsize / (tempinfo.ObjInfo['BufferingConfig'][0][0] * 2) + 4).astype(int)
                fid.seek(pos, 0)
            except Exception as cacheerror:
                fid.close()
                raise IOError('daq:daqread:cannotread\nThe data file cannot be read.  The following error occurred while reading the file:\n' + cacheerror.message)
        # Set the amount of data to read in to the size of the file (fsize).
        if not lastheaderloc:
            lastheaderloc = fsize
    
    # Determine the block locations, types and sizes. CHART contains the fields:
    # firstheader : contains the location after reading the first header.
    # pos         : contains the block locations.
    # type        : contains the block type where 0:header, 1:data, 2:event.
    # blockSize   : contains the size of each block.
    # headerSize  : contains the size of each header.
    # 4 is subtracted for the end header, the object info header, the
    # hardware info header, and the engine info header.
    try:
        chart = localReadChart(fid, pos, lastheaderloc, num_of_blocks - 4)
    except Exception as cacheerror:
        fid.close()
        raise IOError('daq:daqread:cannotread\nThe data file cannot be read.  The following error occurred while reading the file:\n' + cacheerror.message)

    # To move through the data:
    # To get the header information - fid.seek(chart.firstheader + chart.pos)
    # To get the data information - fid.seek(chart.pos + chart.headerSize)

    # Read and parse the initial state of the object.  If the final state of the
    # object was successfully read into the variable info by localPreprocessFile,
    # parse the final state of the object.  Replace the object information in
    # objinfo with the final state information in out.
    try:
        objinfo, sampleTimes = localReadInfo(fid, chart, creationTime, engineOffset)
    except Exception as cacheerror:
        fid.close()
        raise IOError('daq:daqread:cannotread\nThe data file cannot be read.  The following error occurred while reading the file:\n' + cacheerror.message)
    if info:
        out = localHeaderFormat(info)
        out.ObjInfo['Running'] = 'Off'
        out.ObjInfo['Channel'] = objinfo.ObjInfo['Channel']
        out.ObjInfo['EventLog'] = objinfo.ObjInfo['EventLog']
        out.HwInfo['NativeDataType'] = out.HwInfo['NativeDataType'].lower()
        # special case for IOTech adaptor.  It reports a native data type of 4
        # byte real, which maps to single precision in MATLAB.
        if out.HwInfo['NativeDataType'] == 'real4':
            out.HwInfo['NativeDataType'] = 'single'
        objinfo = out

    # Determine the number of channels.
    try:
        num_chans = len(objinfo.ObjInfo['Channel'])
    except Exception as cacheerror:
        fid.close()
        raise IOError('daq:daqread:cannotread\nThe data file cannot be read.  The following error occurred while reading the file:\n' + cacheerror.message)

    # Initialize variables.
    nanLoc = []
    storetimerange = []
    extractEvents = 0

    # Initialize variables depending on the arguments.
    if retmode != 'info':
        if rangearg == False:
            # Initialize variables.
            samples = []
            timerange = []
            triggers = []
            channels = []
        else:
            # Parse input into samples, channels, etc.
            samples, timerange, triggers, channels, errmesg = localParseInput(objinfo, num_chans, samplesacquired, Samples, Time, Triggers, Channels)
            # Error if an invalid property was passed.
            if errmesg:
                fid.close()
                raise ValueError("daq:daqread:unexpected\n" + errmesg)
       
            # Determine if the eventlog needs to be modified because either
            # the number of samples, the number of triggers or the time
            # range was specified.
            if samples or triggers or timerange:
                extractEvents = 1

    # Depending on return mode, calculate data, time, absolute time, events and
    # info structure.

    # Read data.
    if retmode in ['data-info', 'data', 'data1', 'data2', 'data3', 'data4', 'data5'] and np.all(sampleTimes != -1):
        # Determine the number of SamplesPerTrigger (for placing NaNs).
        samplesPerTrigger = objinfo.ObjInfo['SamplesPerTrigger']
   
        # If SamplesPerTrigger is set to INF then the number of samples per
        # trigger is the number of samples acquired.
        if samplesPerTrigger == float('inf') and samplesacquired:
            samplesPerTrigger = samplesacquired
   
        # If the samples to read in is specified in time, convert to samples.
        if timerange:
            sampleRate = objinfo.ObjInfo['SampleRate']
            clockSource = objinfo.ObjInfo['ClockSource']
            if clockSource.lower != 'internal':
                warnings.warn("daq:daqread:fileread\nThe requested samples may be incorrect if the SampleRate property does not represent the external clock rate.  Try specifying the SAMPLES range.")  
          
            samples = np.floor(np.array(timerange) * sampleRate).astype(int).tolist()
            
        # If the samples to read in is specified in triggers, convert to samples.
        if triggers:
            if samplesPerTrigger != float('int'):
                if len(triggers) == 2:
                    samples = [samplesPerTrigger * triggers[0], samplesPerTrigger * triggers[1]]
                else:
                    samples = [samplesPerTrigger * triggers[0], samplesPerTrigger * (triggers[0] + 1)]
            else:
                fid.close()
                raise IOError('daq:daqread:fileread\nError reading the file.  Try specifying the SAMPLES range.')
      
        # If samples is empty define it.
        if len(samples) == 0 and samplesacquired:
            samples = [0, samplesacquired]
   
        # Determine if the maximum samples requested is more than the number of
        # samples available. 
        if len(samples) > 0 and samplesacquired:
            max_available = samplesacquired
            if samples[1] > max_available:
                samples[1] = max_available
                warnings.warn('daq:daqread:fileread\nMore samples than are available has been requested. Only %d samples are available.' % (max_available,))
   
        # Read the data in.
        try:
            [data, nanLoc] = localReadData(fid, chart, num_chans, samples, channels, objinfo)
        except Exception as cacheerror:
            # cache the error code that caused the error
            fid.close()
            # the following conditional is not implemented yet
            # Original code:
            #     if isinstance(cacheerror, MATLAB_pmaxsize_Error):
            if False:
                # Special case for file too large.  This is the most common case
                # where a customer is attempting to open a file that has more
                # than 536,870,911 data points on 32 bit MATLAB.
                raise IOError('daq:daqread:filetoolarge\nThe file is too large to be read into the program. Try specifying a sample range using SAMPLES.')
            else:
                # In all other cases, here's the improved error message, with
                # the read error message appended.
                raise IOError('daq:daqread:cannotread\nThe data file cannot be read.  The following error occurred while reading the file:\n' + cacheerror.message)
        
        # If the SamplesPerTrigger is Inf and couldn't be calculated from the
        # data file, calculate it from the data.
        if samplesPerTrigger == float('inf'):
            # TODO: this works only when "Samples" and "Time" are not given.
            # When they are given, "samplesPerTrigger" is wrongly estimated as
            # the number of samples within the specified range, and also
            # "samples" is wrongly replaced.
            triggerRepeat = len(nanLoc)
            samplesPerTrigger = np.ceil(len(data) / (triggerRepeat + 1)).astype(int)
            samples = [0, (triggerRepeat + 1) * samplesPerTrigger]
    else:
        data = []
    
    # Calculate time.
    if retmode in ['data-info', 'data2', 'data3', 'data4', 'data5'] and np.all(sampleTimes != -1):
        # Need ChannelSkew, TriggerDelay and TriggerDelayUnits to calculate
        # the time matrix or vector.
        channelSkew = objinfo.ObjInfo['ChannelSkew']
        triggerDelay = objinfo.ObjInfo['TriggerDelay']
        triggerDelayUnits = objinfo.ObjInfo['TriggerDelayUnits']
        sampleRate = objinfo.ObjInfo['SampleRate']
   
        # Get the trigger times.
        ttimes = sampleTimes
   
        # Determine which trigger occured.
        trig = localLocateTrigger(samples, samplesPerTrigger)
        ttimes = ttimes[trig] #ttimes(trig)

        # ttimes may be modified based on TriggerDelay.
        if triggerDelay != 0:
            if triggerDelayUnits == 'Samples':
                triggerDelay = triggerDelay / sampleRate
            else:
                #round trigger delay to nearest sample
                triggerDelay = np.floor(triggerDelay * sampleRate + 0.5) / sampleRate
      
            ttimes = ttimes + triggerDelay
   
        # The length of the vector will be the same as the length of the data.
        time = np.zeros(data.shape[0])
       
        # Initialize variables.
        startindex = 0
        triggerNum = 0
        starttime = ttimes[triggerNum]
   
        # Calculate the length of the first time block.
        if nanLoc:
            firstlength = nanLoc[0]
            endtime = starttime + samplesPerTrigger / sampleRate
            addNaN = 1
            starttime = starttime + (samples[0] - trig[0] * samplesPerTrigger) / sampleRate
        else:
            starttime = starttime + (samples[0] - trig[0] * samplesPerTrigger) / sampleRate
            firstlength = samples[1] - samples[0] + 1
            endtime = starttime + (samples[1] - samples[0]) / sampleRate
            addNaN = 0;
   
        # Fill in the first time block.
        time[startindex:firstlength - 1] = np.linspace(starttime, endtime, firstlength - 1, endpoint=False)
        if addNaN:
            time[firstlength] = NaN
        startindex = firstlength + 1
        triggerNum = triggerNum + 1
   
        # Fill in the middle time blocks.
        for i in range(1, len(ttimes) - 1):
            starttime = ttimes(triggerNum)
            time[startindex:startindex + samplesPerTrigger] = np.linspace(starttime, starttime + samplesPerTrigger / sampleRate, samplesPerTrigger, endpoint=False)
            time[startindex + samplesPerTrigger + 1] = NaN;
            startindex = startindex + samplesPerTrigger + 2
            triggerNum = triggerNum + 1
   
        # Fill in the last time block.
        if len(ttimes) > 1:
            starttime = ttimes[triggerNum]
            lastlength = len(time) - startindex
            time[startindex:startindex + lastlength] = np.linspace(starttime, starttime + lastlength / sampleRate, lastlength, endpoint=False)
            if startindex + lastlength == nanLoc[-1]:
                time[startindex + lastlength] = NaN
      
        # If TimeFormat is set to 'matrix', calculate time matrix using ChannelSkew.
        if timeformat.lower() == 'matrix':
            t1 = time
            time = np.zeros((len(t1), num_chans))
            time[:,0] = t1
            for i in range(1, num_chans):
                time[:, i] = t1 + channelSkew * i
    else:
        time = []

    # Calculate absolute time zero.
    if retmode in ['data3', 'data4', 'data5']:
        abstime = objinfo.ObjInfo['InitialTriggerTime']
    
    # Return just the event information.
    if retmode in ['data4', 'data5']:
        if extractEvents:
            eventinfo = localExtractEvents(objinfo.ObjInfo, samples, triggers, storetimerange)
        else:
            eventinfo = objinfo.ObjInfo['EventLog']
    
    # Return object information.
    if retmode in ['data-info', 'info', 'data5']:
        daqinfo = objinfo

    # Convert the data to double if specified.
    if retmode in ['data-info', 'data', 'data1', 'data2', 'data3', 'data4', 'data5'] and dataformat.lower() == 'double':
        try:
            if len(data) > 0:
                data = localConvertDataDouble(data, objinfo, channels)
        except:
            warnings.warn('daq:daqread:dataconversionwarning\nAn error occurred while converting data to double format. Try specifying ''native'' dataformat.')  
            fid.close()
        if nanLoc:
            data[nanLoc, :] = NaN
    
    # Close the file.
    fid.close()
    
    if retmode == 'data-info':
        return data, time, daqinfo
    elif retmode == 'info':
        return objinfo
    elif retmode in ['data', 'data1']:
        return data
    elif retmode == 'data2':
        return data, time
    elif retmode == 'data3':
        return data, time, abstime
    elif retmode == 'data4':
        return data, time, abstime, eventinfo
    elif retmode == 'data5':
        return data, time, abstime, eventinfo, daqinfo

def localPreprocessFile(fid, fsize):
    '''
    Determine the number of samples logged, the number of blocks logged and the
    state of the object at the end of the acquisition.
    '''
    # Initialize variables.
    flag = 0

    # Loop through and get the object information, hardware information and
    # engine information.
    try:
        # Read the end block to determine the number of samples acquired and
        # where the end object is stored.
        fid.seek(fsize - 16, 0)
        samplesacquired = np.fromfile(fid, 'int64', 1)[0]
        lastheaderloc = np.fromfile(fid, 'int64', 1)[0]
    
        # Position file indicator to the location of the last header information.
        fid.seek(lastheaderloc, 0)
        pos = fid.tell()
    
        # Loop through and get the object information, hardware information and
        # engine information.
        info = [DAQStruct() for _ in range(3)]
        for i in range(3):
            info[i].blocksize = np.fromfile(fid, 'int32', 1)[0]
            info[i].blocktype = np.fromfile(fid, 'int32', 1)[0]
            info[i].headersize = np.fromfile(fid, 'int32', 1)[0]
            info[i].number = np.fromfile(fid, 'uint32', 1)[0]
            info[i].hdr_typestr = fid.read(16)
            fid.seek(pos + info[i].headersize, 0)
            info[i].data = fid.read(info[i].blocksize - info[i].headersize)
            fid.seek(pos + info[i].blocksize, 0)
            pos = fid.tell()
    
        # The number of blocks equals the number of the last entry in the last 
        # header plus the end block.
        num_of_blocks = info[2].number + 1
        flag = 0
    except:
        # Reset all variables since they are most likely corrupted if an error
        # occurred somewhere while reading them.  Set the flag and return.
        samplesacquired = None
        num_of_blocks = None
        info = []
        lastheaderloc = None
        flag = 1

    return samplesacquired, num_of_blocks, info, lastheaderloc, flag

def localReadChart(fid, pos, fsize, blocks):
    '''
    Determine the location and types of the blocks.

    firstheader : contains the location after reading the first header.
    pos         : contains the block locations.
    type        : contains the block type where 0 - header, 1 - data
                  and 2 - event information.
    blockSize   : contains the size of each block.
    headerSize  : contains the size of each header.
    '''
    # Initialize variables.
    temp = DAQStruct()
    temp.pos = -np.ones(blocks).astype(long)
    temp.blockSize = -np.ones(blocks).astype(int)
    temp.type = -np.ones(blocks).astype(int)
    temp.headerSize = -np.ones(blocks).astype(int)

    # Read in the first block and the firstheader (which only has to be done once).
    temp.pos[0] = pos
    temp.blockSize[0] = np.fromfile(fid, 'int32', 1)[0]
    temp.type[0] = np.fromfile(fid, 'int32', 1)[0]
    temp.headerSize[0] = np.fromfile(fid, 'int32', 1)[0]
    temp.number = np.fromfile(fid, 'uint32', 1)[0]
    temp.firstheader = fid.tell() - pos

    # Adjust the file position indicator by the blocksize.
    fid.seek(pos + temp.blockSize[0], 0)
   
    # Get the new file position which is pos+info.blocksize.
    pos = fid.tell()

    # Create a counter.
    k = 1

    # Loop through and get the blocksize, headersize and type information.
    while pos < fsize:
        temp.pos[k] = pos
        temp.blockSize[k] = np.fromfile(fid, 'int32', 1)[0]
        temp.type[k] = np.fromfile(fid, 'int32', 1)[0]
        temp.headerSize[k] = np.fromfile(fid, 'int32', 1)[0]
        
        # Adjust the file position indicator by the blocksize.
        fid.seek(pos + temp.blockSize[k], 0)
        
        # Get the new file position which is pos+temp.blocksize.
        pos = fid.tell()
        
        # Determine if more space is needed to store the file information.
        # If so, increase it by twenty-five percent.
        if k >= blocks:
            add_blocks = np.ceil(.25 * blocks).astype(int)
            temp.pos.append(-np.ones((1, add_blocks)))
            temp.blockSize.append(-np.ones((1, add_blocks)))
            temp.type.append(-np.ones((1, add_blocks)))
            temp.headerSize.append(-np.ones((1, add_blocks)))
            blocks = blocks + add_blocks
        # Increment the counter.
        k = k + 1
    
    return temp

def localLocateTrigger(samples, samplesPerTrigger):
    '''
    Determine which triggers occurred.
    '''
    # Find the trigger values.
    if samplesPerTrigger == float('inf'):
        tloc = []
    else:
        tloc = range(np.ceil((samples[0] + 0.5) / samplesPerTrigger).astype(int) * samplesPerTrigger, samples[1], samplesPerTrigger)
    #num_trig = len(tloc)
    
    # If tloc is empty than a single trigger occurred - samples = [100 200];
    if not tloc:
        trig = [np.ceil((samples[0] + 0.5) / samplesPerTrigger).astype(int) - 1,]
        return trig

    # Determine if the trigger before tloc is needed - samples = [1000 2000];
    if samples[0] <= tloc[0]:
        tloc.insert(tloc[0] - samplesPerTrigger, 0)

    # If last sample is the last value of the data matrix remove the extra
    # trigger from tloc - samples = [1 4096];
    if samples[1] == tloc[-1]:
        tloc = tloc[0:-1]

    # Determine the trigger number: [0 1024 2048] ==> [0 1 2]
    trig = tloc / samplesPerTrigger
    
    return trig

def localReadData(fid, chart, num_chans, samples, channels, info):
    '''
    Read the data information.
    '''
    # Initialize variables.
    flag = 0
    startloc = 0
    nanLoc = []
    samplesPerTrigger = info.ObjInfo['SamplesPerTrigger']
    engineBlockSize = info.ObjInfo['BufferingConfig'][0][0]
    #bits = info.HwInfo['Bits']
    datatype = info.HwInfo['NativeDataType']

    # Data information has a type of 1.
    data_loc = np.where(chart.type == 1)[0]
    data_pos = chart.pos[data_loc]
    data_block = chart.blockSize[data_loc]
    data_header = chart.headerSize[data_loc]

    # Create a matrix of NaNs the size of the data matrix to be returned.
    if channels:
        numcols = len(channels)
    else:
        numcols = num_chans

    # Determine the number of triggers for the supplied samples range.
    if len(samples) > 0:
        if samplesPerTrigger == float('inf'):
            tloc = []
        else:
            tloc = range(np.ceil(samples[0] / samplesPerTrigger).astype(int) * samplesPerTrigger, samples[1], samplesPerTrigger)
        num_trig = len(tloc)
        # Add in the number of triggers (TriggerRepeat+1) to the number of rows.
        if num_trig == 0:
            numrows = samples[1] - samples[0]
        else:
            numrows = samples[1] - samples[0] + num_trig - 1
    else:
        # The number of samples could not be determined from the data file
        # (either SamplesPerTrigger or TriggerRepeat was inf and was not 
        # supplied as input).
#        numrows = (engineBlockSize / num_chans) * len(data_loc)
        numrows = engineBlockSize * len(data_loc)

    # Initialize data.
    
    data = np.zeros((numrows, numcols), dtype=datatype.lower())

    # Determine the datattype to use when reading in the block of information.
    datatype = datatype.lower()

    # Create a counter for the samples to read in.
    countsamples = np.array(samples)

    # Build up the data array.  
    dinfo = [DAQStruct() for _ in range(len(data_pos))]
    for i in range(len(data_pos)):
        # To get the header information - fseek(startlocation+chart.pos)
        startlocation = chart.firstheader
        fid.seek(startlocation + data_pos[i], 0)
        dinfo[i].hdr_starttime = np.fromfile(fid, 'double', 1)[0]
        dinfo[i].hdr_endtime = np.fromfile(fid, 'double', 1)[0]
        dinfo[i].hdr_startsample = np.fromfile(fid, 'int64', 1)[0]
        dinfo[i].hdr_intrigger = np.fromfile(fid, 'int32', 1)[0]
        dinfo[i].hdr_flags = np.fromfile(fid, 'uint32', 1)[0]
   
        # Position the file indicator and determine the number of points to read.
        fid.seek(data_pos[i] + data_header[i], 0)
        sizebytes = data.nbytes / (numrows * numcols)
        num_points = int(((data_block[i] - data_header[i]) / sizebytes) / num_chans)
   
        # Determine if the data block needs to be read in or if the for loop
        # should be incremented to the next data block.
        if len(samples) == 0:
            readblock = 1
        elif dinfo[i].hdr_startsample + 1 + num_points <= samples[0] and not (dinfo[i].hdr_startsample == 0 and num_points >= samples[1]):
            readblock = 0
            countsamples = countsamples - num_points
        else:
            readblock = 1
   
        if readblock:
            # Read the data block.
            block = np.fromfile(fid, datatype, num_chans * num_points)
            block.resize((num_points, num_chans))
      
            # Extract the requested channels.
            if channels:
                block = block[:, channels]
      
            if len(samples) > 0:
                # Extract the samples from block.
                mins = countsamples[0]
                maxs = countsamples[1]
                if num_points >= maxs:
                    # If the maximum samples to get is less than the number
                    # of samples in the block, extract the samples and return.
                    # Ex. samples = [1 100]; blocksize = 200;
                    block = block[mins:maxs, :]
                    flag = 1
                elif num_points >= mins:
                    # If the number of samples in the block is greater than the
                    # lower sample range, extract from the lower sample range to
                    # the number of samples in the block.
                    # Ex. range = [400 1000]; blocksize = 600;
                    block = block[mins:num_points, :]
                    # countsamples is reset to range from 1 to the remainder.
                    countsamples = [0, maxs - block.shape[0] - mins]
                    if samples[1] == 0:
                        flag = 1
                else:
                    # Extract no samples and read just the samples range by the blocksize.
                    # Ex. range = [400 1000]; blocksize = 200;
                    countsamples = [mins - block.shape[0], maxs - block.shape[0]]
        
                    block = np.array([])
            # Determine the number of rows in the data.
            blocksize = block.shape[0]
      
            # Concatenate the block into the data matrix.
            if len(block) > 0:
                # if num_trig > 1: #Special case for IOTech: We should never be adding
                # NaN unless we have multiple triggers. The problem seems to be
                # when we wrote this file we put flag=7 and this is forcing
                # mod(flag,2)==1 on all reads of blocks above 1.
                if i > 1 and np.mod(dinfo[i].hdr_flags, 2) == 1 and startloc != 0:
                    data[startloc, :] = NaN
                    nanLoc.append(startloc)
                    startloc = startloc + 1
                data[startloc:startloc + blocksize, :] = block[0:blocksize, :]
                startloc = startloc + blocksize
      
            # Return if all the samples requested has been read.
            if flag:
                return data, nanLoc
            if i == len(data_pos) and not samples:
                data = data[0:startloc, :]
                
    return data, nanLoc

def localConvertDataDouble(data, info, channels):
    ''' 
    Convert the data to double with engineering units.
    '''
    # Initialize variables.
    data = data.astype(np.float64)
    if not channels:
        channels = range(data.shape[1])

    # Need to loop through each column of data which represents one channel.
    for i in range(data.shape[1]):
        slope = info.ObjInfo['Channel'][channels[i]]['NativeScaling']
        intercept = info.ObjInfo['Channel'][channels[i]]['NativeOffset']

    # Convert the data.
    data[:, i] = slope * data[:, i] + intercept
    
    return data

def localReadInfo(fid, chart, creationTime, engineOffset):
    '''
    Read the headers and event information.
    '''

    # Read the header information - Header information has a type of 0. 
    header_loc = np.where(chart.type == 0)[0]
    header_pos = chart.pos[header_loc]
    header_block = chart.blockSize[header_loc]
    header_header = chart.headerSize[header_loc]
    startlocation = chart.firstheader

    # Index 1: Object information.
    # Index 2: Hardware information.
    # Index 3: Engine information.
    # Remaining headers are the number of channels.

    hinfo = [DAQStruct() for _ in range(len(header_loc))]
    for i in range(len(header_loc)):
        fid.seek(header_pos[i] + startlocation, 0)
        hinfo[i].hdr_typestr = fid.read(16)
        # Move the file position indicator by the size of the header.
        fid.seek(header_pos[i] + header_header[i], 0)
        # Read in the property information.  
        # size of property information =  blocksize - headersize.
        hinfo[i].data = fid.read(header_block[i] - header_header[i])

    # Read the event information - Event information has a type of 2. 
    event_loc = np.where(chart.type == 2)[0]
    event_pos = chart.pos[event_loc]
    #event_block = chart.blockSize[event_loc]
    event_header = chart.headerSize[event_loc]

    # Initialize einfo
    einfo = DAQStruct()
    einfo.hdr_entries = np.empty(len(event_pos), dtype='int32')
    einfo.data = []

    for j in range(len(event_pos)):
        # Adjust the file position indicator.
        fid.seek(event_pos[j] + startlocation, 0)
        # Read the number of events logged.
        einfo.hdr_entries[j] = np.fromfile(fid, 'int32', 1)[0]
   
        # Adjust the file position indicator by the size of the header.
        fid.seek(event_pos[j] + event_header[j], 0)
   
        # Loop through the events.
        for i in range(einfo.hdr_entries[j]):
            edata = DAQStruct()
            edata.timestamp = np.fromfile(fid, 'double', 1)[0]
            edata.samplestamp = np.fromfile(fid, 'int64', 1)[0]
            edata.logtype = np.fromfile(fid, 'int16', 1)[0]
            edata.entrysize = np.fromfile(fid, 'int16', 1)[0]
            #block alignment issue.
            dummy = np.fromfile(fid, 'int32', 1)
            edata.string = fid.read(edata.entrysize - 24)
            einfo.data.append(edata)

    # Convert the header information and event information to output format.
    out1 = localHeaderFormat(hinfo)
    eventinfo = localEventFormat(einfo, creationTime, engineOffset)

    # Add the eventinfo information to the EventLog field.
    out1.ObjInfo['EventLog'] = eventinfo

    # Determine the triggertimes from the event information and add to output.
    ttimes = localFindTriggerTime(eventinfo, out1.ObjInfo['SampleRate'])
    
    return out1, ttimes

def localHeaderFormat(header):
    '''
    Convert the header structure into the correct output format.
    '''
    # Index 0: Object information.
    # Index 1: Hardware information.
    # Index 2: Engine information.
    # Remaining headers are the number of channels.
    out = DAQStruct()
    Channel = []
    for i in range(len(header)):
        # exec statement produces a dictionary called 'x'.
        x = {}
        header[i].data = header[i].data.replace('1.#INF', 'inf')
        
        # Try to recover if one of the string properties contains a
        # carriage return.
        try:
            exec(mat2py(header[i].data))
        except:
            tempHeader = localCheckQuote(header[i].data)
            exec(mat2py(tempHeader))
        
        
        typestr = header[i].hdr_typestr
        typestr = ''.join(map(lambda x: x if x != '\x00' else ' ', struct.unpack('16c', typestr))) # convert all zeros (nulls) to blanks
        typestr = typestr.rstrip()
        if typestr == 'AnalogInput':
            out.ObjInfo = x
        elif typestr == 'DaqHwInfo':
            out.HwInfo = x
        elif typestr == 'Channel':
            # JI: the following conditional is inherited from the original
            # Matlab version, but this seems to be of no use.
            if 'Parent' in x.keys():
                prot = dict(x)
                del prot['Parent']
            Channel.append(x)

    # Add the Channel information to the Channel field.
    if len(header) > 3:
        out.ObjInfo['Channel'] = Channel
    
    return out

def localEventFormat(event, creationTime, engineOffset):
    '''
    Convert the event structure into the correct output format.
    '''
    out = []
    # Create the event structure.
    for i in range(len(event.data)):
        # Initialize variables.
        x = {}
   
        # Depending on the logtype, set the Type field of the event
        # structure.  eval is called if the event has additional
        # fields (other than TimeStamp and SampleStamp).
        logtype = event.data[i].logtype
        tmp = {}
        tmp['Data'] = {}
        if logtype == 0:
            tmp['Type'] = 'Start'
            tmp['Data']['AbsTime'] = localEventTime(event.data[i].timestamp, creationTime, engineOffset)
        elif logtype == 1:
            tmp['Type'] = 'Stop'
            tmp['Data']['AbsTime'] = localEventTime(event.data[i].timestamp, creationTime, engineOffset)
        elif logtype == 2:
            tmp['Type'] = 'Trigger'
            tmp['Data']['AbsTime'] = localEventTime(event.data[i].timestamp, creationTime, engineOffset)
            exec(mat2py(event.data[i].string))
        elif logtype == 3:
            tmp['Type'] = 'RunTimeError'
            tmp['Data']['AbsTime'] = localEventTime(event.data[i].timestamp, creationTime, engineOffset)
            x = event.data[i].string
        elif logtype == 4:
            tmp['Type'] = 'Overrange'
            exec(mat2py(event.data[i].string))
        elif logtype == 5:
            tmp['Type'] = 'DataMissed'
        elif logtype == 6:
            tmp['Type'] = 'SamplesAcquired'
        out.append(tmp)
   
        # Set the RelSample field (every event has at least this field).
        out[i]['Data']['RelSample'] = event.data[i].samplestamp
   
        # The Trigger, RunTimeError and Overrange events have additional
        # fields for the event structure.  Create them here.
        logtype = event.data[i].logtype
        if logtype == 2:  # Trigger.
            if x:
                if x['Channel'] == -1:
                    x['Channel'] = [];
                out[i]['Data']['Channel'] = x['Channel']
                out[i]['Data']['Trigger'] = x['Trigger']
        elif logtype == 3:  # RunTimeError.
            if x:
                out[i]['Data']['String'] = x
        elif logtype == 4:  # Overrange.
            if x:
                out[i]['Data']['Channel'] = x['Channel']
                out[i]['Data']['Overrange'] = x['Overrange']
                
    return out

def localEventTime(sec, creationTime, engineOffset):
    '''
    Convert the seconds logged to a clock.
    '''
    # Convert creationTime to datetime.
    usec = int((creationTime[5] - np.floor(creationTime[5])) * 1e6)
    ctime = datetime.datetime(microsecond=usec, *creationTime.astype(int))
    
    # convert the seconds logged to timedelta
    time = datetime.timedelta(seconds=sec-engineOffset)
    
    # Add two together
    time = ctime + time
    
    # convert the result to a vector [year, month, day, hour, min, sec] of float values
    timearray = np.array([time.year, time.month, time.day, time.hour, time.minute, time.second], dtype=float)
    timearray[5] += time.microsecond * 1e-6
    
    return timearray.tolist()

def readObject(fid, pos):
    '''
    Read the object information only.  This is needed if the last data block was
    not logged to the file.
    '''
    # Initialize variables.
    temp = [DAQStruct(),]
    temp[0].pos = pos
    temp[0].blockSize = np.fromfile(fid, 'int32', 1)[0]
    temp[0].type = np.fromfile(fid, 'int32', 1)[0]
    temp[0].headerSize = np.fromfile(fid, 'int32', 1)[0]
    temp[0].number = np.fromfile(fid, 'uint32', 1)[0]
    temp[0].hdr_typestr = fid.read(16) 
    temp[0].firstheader = fid.tell()

    # Move the file position indicator by the size of the header.
    fid.seek(pos + temp[0].headerSize, 0)
    # Read in the property information.  
    # size of property information =  blocksize - headersize.
    temp[0].data = fid.read(temp[0].blockSize - temp[0].headerSize)

    # Convert the data to the output structure.
    out1 = localHeaderFormat(temp)
    
    return out1

def localFindTriggerTime(events, sampleRate):
    '''
    Determine the trigger times from the event information.
    '''
    # If a trigger event did not occur, want to return the object information
    # but no data.
    try:
        out = np.array([events[i]['Data']['RelSample'] for i in range(len(events)) if events[i]['Type'] == 'Trigger'])
        
        # ttimes is used to calculate the time vector.   
        ttimes = out / sampleRate
    except:
        ttimes = -1
    
    return ttimes

def localExtractEvents(objinfo, samples, triggers, timerange):
    '''
    Determine the trigger times from the event information.
    '''
    # Initialize variables.
    events = objinfo['EventLog']
    samplesPerTrigger = objinfo['SamplesPerTrigger']
    eventtimes = []

    # If the original call to daqread was in terms of samples, convert
    # samples to triggers.
    if not triggers and not timerange:
        # SamplesPerTrigger = 1000. samples = [1000 4000];  triggers = [2 5];

        # Calculate the first trigger.
        if samples[0] == 0:
            triggers.append(0)
        else:
            triggers.append(np.ceil(samples[0] / samplesPerTrigger).astype(int))

        # Calculate the second trigger.
        triggers.append(np.ceil(samples[1] / samplesPerTrigger).astype(int))

    # Extract the correct event information either in terms of triggers
    # or in terms of time.
    if not triggers:
        # Data was specified in terms of time.
   
        # Extract all the event times.
        for i in range(len(events)):
            eventtimes.append(events[i]['Data']['TimeStamp'])
   
        # Find the location of the first time.
        index = [i for i in range(len(eventtimes)) if eventtimes[i] < timerange[0]]
        value = []
        if not index:
            value[0].append(0)
        else:
            value[0].append(index[-1] + 1)

        # Find the location of the second time.
        index = [i for i in range(len(eventtimes)) if eventtimes[i] > timerange[1]]
        if not index:
            value[1].append(len(events))
        else:
            value[1].append(index[0])

        # Extract only those events for the specified times.
        out = events[value[0]:value[1]]
    else:
        # Data was specified in either triggers or samples.  But samples
        # has been converted to triggers.  Triggers will be used.
   
        # Determine the location of the trigger events.
        trigloc = [i for i in range(len(events)) if events[i]['Type'] == 'Trigger']

        # Extract the trigger events and any other events that may have occurred 
        # in between triggers.
        trigevent = events[min(trigloc):max(trigloc) + 1]

        # Determine the new location of the trigger events in the structure.
        newtrigloc = [i for i in range(len(trigevent)) if trigevent[i]['Type'] == 'Trigger']
   
        # Create a temp event structure which contains only the information
        # requested.
        if len(triggers) == 1:
            # In case Triggers was defined as 2.
            out =  trigevent[newtrigloc[triggers[0]]]
        else:
            out =  trigevent[newtrigloc[triggers[0]]:newtrigloc[triggers[1]] + 1]
    
    return out
   
def localParseInput(objinfo, num_chans, samplesacquired, Samples, Time, Triggers, Channels):
    ''' 
    Parse the input 
    '''
    # Initialize variables.
    errmesg = ''
    samp = [] if Samples is None else Samples
    timer = [] if Time is None else Time
    trigger = [] if Triggers is None else Triggers
    chan = [] if Channels is None else Channels

    # Determine what properties were specified and their value.
    if len(samp) > 0:
        if len(samp) != 2:
            errmesg = " 'Samples' must be specified as a two-element range."
            return samp, timer, trigger, chan, errmesg
        elif not isinstance(samp[0], int) or not isinstance(samp[1], int):
            errmesg = " 'Samples' must contain integer values."
            return samp, timer, trigger, chan, errmesg
        elif min(samp) < 0 or samp[1] <= samp[0]:
            errmesg = " Invalid 'Samples' range requested."
            return samp, timer, trigger, chan, errmesg
        elif len(timer) > 0:
            errmesg = " The 'Time' and 'Samples' properties are mutually exclusive."
            return samp, timer, trigger, chan, errmesg
        elif len(trigger) > 0:
            errmesg = " The 'Triggers' and 'Samples' properties are mutually exclusive."
            return samp, timer, trigger, chan, errmesg
    elif len(timer) > 0:
        if len(timer) != 2:
            errmesg = " 'Time' must be specified as a two-element range."
            return samp, timer, trigger, chan, errmesg
        elif min(timer) < 0 or timer[1] <= timer[0]:
            errmesg = " Invalid ''Time'' range requested."
            return samp, timer, trigger, chan, errmesg
        elif len(trigger) > 0:
            errmesg = " The 'Triggers' and 'Time' properties are mutually exclusive."
            return samp, timer, trigger, chan, errmesg
    elif len(trigger) > 0:
        if len(trigger) > 2:
            errmesg = " 'Triggers' must be specified as a single trigger or a two-element range."
            return samp, timer, trigger, chan, errmesg
        elif not np.all([isinstance(x, int) for x in trigger]):
            errmesg = " 'Triggers' must contain integer values."
            return samp, timer, trigger, chan, errmesg
        elif min(trigger) < 0:
            errmesg = " Invalid 'Triggers' index."
            return samp, timer, trigger, chan, errmesg
        elif trigger[1] <= trigger[0]:
            errmesg = " Invalid 'Triggers' range requested."
            return samp, timer, trigger, chan, errmesg
    
    if len(chan) > 0:
        if np.all([isinstance(x, str) for x in chan]):
            index = localFindChannel(objinfo, chan)
            if len(index) != len(chan):
                errmesg = " Invalid channel name specified."
                return samp, timer, trigger, chan, errmesg
            else:
                chan = index
        
        # Channel Index begins from 1, while Python list index begins from 0, 
        # meaning Channel n is the (n-1)-th element of channel list. Here chan 
        # is converted to represent channel list index instead of Channel Index
        # per se.
        chan = [x - 1 for x in chan]
        
        if min(chan) < 0:
            errmesg = " Invalid 'Channels' index."
            return samp, timer, trigger, chan, errmesg
        elif max(chan) >= num_chans:
            errmesg = " 'Channels' index exceeds the number of available channels: %d" % (num_chans,)
            return samp, timer, trigger, chan, errmesg
        elif not np.all([isinstance(x, int) for x in chan]):
            errmesg = " 'Channels' must contain integer values."
            return samp, timer, trigger, chan, errmesg
        elif len(set(chan)) != len(chan):
            errmesg = " A single channel is specified more then once."
            return samp, timer, trigger, chan, errmesg
        
    return samp, timer, trigger, chan, errmesg

def localFindChannel(objinfo, name):
    '''
    Convert the channelnames to channel indices.
    '''
    # Get all the Channel Names and indices.
    channelnames = [x['ChannelName'] for x in objinfo.ObjInfo['Channel']]
    index = [x['Index'] for x in objinfo.ObjInfo['Channel']]
    temp = []

    # Loop through and determine if the specified name equals one
    # of the channelnames.
    for i in range(len(name)):
        try:
            i2 = channelnames.index(name[i])
            temp.append(index[i2])
        except ValueError:
            pass
    
    return temp
   
def localCheckQuote(codestr):
    '''
    If the string cannot be evaled check if the correct number of quotes have been used.
    '''
    lstr = codestr
    
    # Find the single quote and semi-colon locations.
    quoteLoc = [i for i in range(len(lstr)) if lstr[i] == "'"]
    if len(lstr) < 3:
        semiLoc = []
    else:
        semiLoc = [i for i in range(len(lstr)-3) if lstr[i:i+3] == ";x."]
    
    # Loop through to verify that there are an even number of quotes
    # before each ';x.'.  If that is not the case, add a single quote
    # before the 'x; and after the ';'.  If a single quote is added,
    # the locations of single quotes and ';x.' will have to be recalculated.
    for i in range(len(semiLoc)):
#        index = [j for j in range(len(quoteLoc)) if quoteLoc[j] < semiLoc[j]]
        index = [j for j in range(len(quoteLoc)) if quoteLoc[j] < semiLoc[i]]
        if np.mod(len(index), 2) != 0:
            lstr = lstr[0:semiLoc[i]] + "'" + lstr[semiLoc[i]:]
            quoteLoc = [j for j in range(len(lstr)) if lstr[j] == "'"]
            semiLoc = [j for j in range(len(lstr)) if lstr[j] == ";"]
            i = i - 1
    
    return lstr

def mat2py(matcode):
    '''
    Convert a string of Matlab code into a python code. The input string is
    assumed to be a concatenation of substitution statements in the form of:
    
        x.*** = $$$; ($$$ is either a number, a string, or an array)
    
    The function convertes this to a Python code:
    
        x['***'] = $$$;
    
    When $$$ is an array, it is converted to a proper Python expression.
    '''
    # input string is split at ';'s into single statements
    statements = extract_statements(matcode)
    
    pycode = ''
    for stat in statements:
        # obtain left- and right-hand sides of substitution. if stat is not a
        # substitution, this raises ValueError
        lhs, rhs = split_terms(stat)
        
        # LHS: convert x.*** to x['***']
        lhs = mstruct2pydict(lhs)
        
        # RHS: identify data type and convert if necessary
        if rhs[0] in ["'", '"']: # string
            pass
        elif isnumber(rhs): # number (int or float)
            if rhs in ['Inf', 'inf', 'INF']:
                rhs = "float('inf')"
            else:
                pass
        elif rhs[0] in ['[', '{']: # array
            rhs = marray2pylist(rhs)
        else:
            pass
        
        pycode += lhs + '=' + rhs + ';'
    
    return pycode

def extract_statements(codestr):
    strlen = len(codestr)
    stat_ini = 0
    statements = []
    for i, char in enumerate(codestr):
        if char is ';':
            if i == strlen - 1 or codestr[i + 1] in ['x', '\n']:
                statements.append(codestr[stat_ini:i])
                stat_ini = i + 1
    return statements
                
def split_terms(statstr):
    terms = statstr.split('=')
    if len(terms) != 2:
        raise ValueError("Input string is not a substitution.")
    return terms[0].strip(), terms[1].strip()
                
def mstruct2pydict(mstruct):
    tokens = mstruct.split('.')
    pydict = tokens[0]
    for tok in tokens[1:]:
        pydict = pydict + "['" + tok + "']"
    return pydict
            
def isnumber(numstr):
    try:
        int(numstr)
        return True
    except ValueError:
        try:
            float(numstr)
            return True
        except ValueError:
            return False

def marray2pylist(marray):
    if [marray[0], marray[-1]] not in [["[" , "]"], ["{", "}"]]:
        raise ValueError("Input string is not a MATLAB array")
    
    # extract raws
    raws = marray[1:-1].split(';')
    nraws = len(raws) - 1
    if raws[0] == '':
        # this happens when marray is empty
        pass
    elif raws[-1] == '':
        raws = raws[:-1]
        
    # construct pylist raw by raw
    pylist = '['
    for raw in raws:
        terms = raw.strip().split()
        
        # re-assemble the terms separated by blanks within quotes
        if "'" in raw:
            tmp = []
            quoteloc = -1
            for i in range(len(terms)):
                if quoteloc > -1:
                    if terms[i][-1] == "'":
                        tmp.append(' '.join(terms[quoteloc:i+1]))
                        quoteloc = -1
                else:
                    if terms[i][0] == "'" and terms[i][-1] != "'":
                        quoteloc = i
                    else:
                        tmp.append(terms[i])
            terms = tmp
            
        pylist += '[' + ','.join(terms) + '],'
    pylist += ']'
    
    if nraws == 0:
        # [a b c] --> [a, b, c] rather than [[a, b, c],]
        return pylist[1:-2]
    else:
        # [a b c;] --> [[a, b, c],]
        # [a b c; d e f] --> [[a, b, c], [d, e, f]]
        return pylist


if __name__ == '__main__':
#    filename = "C:/Users/ito/datasets/osaka/behavior/20130306/20130306_133232_rec1_pc3.daq"
    filename = "C:/Users/ito/datasets/osaka/behavior/20130306/20130306_134636_rec3_pc3.daq"
#    filename = "N:/datasets/osaka/behavior/20130306/20130306_133232_rec1_pc3.daq"
#    filename = "N:/datasets/osaka/behavior/20130306/20130306_134636_rec3_pc3.daq"
    
#    data, time, objinfo = daqread(filename, 'data-info')
#    objinfo = daqread(filename, 'info')
    
#    data, time, objinfo = daqread(filename, 'data-info', Samples=[44000, 56000])
    data, time, objinfo = daqread(filename, 'data-info', Time=[2., 3.])
#    data, time, objinfo = daqread.daqread(filename, Time=[100, 2000], Channels=['eyecoil_x', 'eyecoil_y'])
    
    print '\n=== object information ==='
    for key, var in objinfo.ObjInfo.iteritems():
        if isinstance(var, list):
            print key, ":"
            for listvar in var:
                print "\t", listvar 
        else:
            print key, ":", var
    print '\n=== hardware information ==='
    for key in objinfo.HwInfo.iterkeys():
        print key, ":", objinfo.HwInfo[key]
    quit()