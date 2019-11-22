#!/usr/bin/env python3
'''
This script is meant to contain the overhead code for organizing and working with
analysis files for BEACON analysis.  Given a reader and some kwargs it will
load the correct analysis file (or create one if necessary).

I anticipate having wrapper functions for certain things such as adding a datasets
or overwriting existing datasets.
'''

import sys
import os
import inspect
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from pprint import pprint
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import numpy
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info


analysis_data_dir = '/home/dsouthall/scratch-midway2/beacon/'
#os.environ['BEACON_ANALYSIS_DIR'] + 'data/'


def loadTriggerTypes(reader):
    '''
    Will get a list of trigger types corresponding to all eventids for the given reader
    trigger_type:
    1 Software
    2 RF
    3 GPS
    '''
    #trigger_types = numpy.zeros(reader.N())
    try:
        N = reader.head_tree.Draw("trigger_type","","goff") 
        trigger_types = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print('Error while trying to copy header elements to attrs.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return trigger_types

def getTimes(reader):
    '''
    This pulls timing information for each event from the reader object..
    
    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.

    Returns
    -------
    times : numpy.ndarray of floats
        The raw_approx_trigger_time values for each event from the Tree.
    subtimes : numpy.ndarray of floats
        The raw_approx_trigger_time_nsecs values for each event from the Tree. 
    trigtimes : numpy.ndarray of floats
        The trig_time values for each event from the Tree.
    '''
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    trigtimes = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
    eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)

    return times, subtimes, trigtimes, eventids



def createFile(reader,redo_defaults=False):
    '''
    This will make an hdf5 file for the run specified by the reader.
    If the file already exists then this will check if the file has
    all of the baseline datasets that are currently expected, and if
    not it will do this prepwork.  It will not overwrite.

    This will returnt the filename of the file.

    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.
    redo_defaults : bool
        If True then this will replace the values of any default fields by determining them again.
        This will not effect any additional datasets added.  This is by defualt False, but sometimes
        if an error was made it may be worthwhile to enable and rerun. 
    '''
    try:
        run = int(reader.run)
        N = reader.N()
        filename = analysis_data_dir + 'run%i_analysis_data.h5'%run

        header_keys_to_copy = []
        h = interpret.getHeaderDict(reader)

        initial_expected_datasets = numpy.array(['eventids','trigger_types','times','subtimes','trigtimes','inband_peak_freq_MHz']) #expand as more things are added.  This should only include datasets that this function will add.
        initial_expected_attrs    = numpy.array(['N','run'])
        if os.path.exists(filename):
            print('%s already exists, checking if setup is up to date.'%filename )

            with h5py.File(filename, 'a') as file:
                
                try:
                    if redo_defaults == False:
                        attempt_list = initial_expected_datasets[~numpy.isin(initial_expected_datasets,list(file.keys()))]
                    else:
                        attempt_list = initial_expected_datasets

                    if numpy.any(numpy.isin(attempt_list,['eventids','times','subtimes','trigtimes','inband_peak_freq_MHz'])):
                        times_loaded = True
                        times, subtimes, trigtimes, eventids = getTimes(reader)


                    for key in attempt_list:
                        print('Attempting to add content for key: %s'%key)
                        if key == 'eventids':
                            #if ('eventids' in list(file.keys())) == False:
                            del file['eventids']
                            file.create_dataset('eventids', (N,), dtype=numpy.uint32, compression='gzip', compression_opts=9, shuffle=True)
                            file['eventids'][...] = eventids
                        elif key == 'trigger_types':
                            if ('trigger_types' in list(file.keys())) == False:
                                file.create_dataset('trigger_types', (N,), dtype=numpy.uint8, compression='gzip', compression_opts=9, shuffle=True)
                            file['trigger_types'][...] = loadTriggerTypes(reader)
                        elif key == 'times':
                            if ('times'  in list(file.keys())) == False:
                                file.create_dataset('times', (N,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                            file['times'][...] = times
                        elif key == 'subtimes':
                            if ('subtimes'  in list(file.keys())) == False:
                                file.create_dataset('subtimes', (N,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                            file['subtimes'][...] = subtimes
                        elif key == 'trigtimes':
                            if ('trigtimes'  in list(file.keys())) == False:
                                file.create_dataset('trigtimes', (N,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                            file['trigtimes'][...] = trigtimes
                        elif key == 'inband_peak_freq_MHz':
                            #Used for gating obvious backgrounds like known CW
                            if ('inband_peak_freq_MHz'  in list(file.keys())) == False:
                                file.create_dataset('inband_peak_freq_MHz', (N,8), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                            reader.setEntry(eventids[0])
                            t = reader.t()/1e9
                            freqs = numpy.fft.rfftfreq(len(t),t[1]-t[0])/1e6
                            band_cut = numpy.logical_and(freqs > 10.0, freqs < 100.0)
                            print('Calculating Spectral Frequency\n')
                            print_every = int(N/1000)
                            for event_index,eventid in enumerate(eventids):
                                if event_index%print_every == 0:
                                    sys.stdout.write('\r%i/%i'%(event_index+1,len(eventids)))
                                    sys.stdout.flush()
                                reader.setEntry(eventid)
                                wfs = numpy.zeros((8,len(t)))
                                for channel in range(8):
                                    wfs[channel] = reader.wf(channel)

                                fft = numpy.fft.rfft(wfs,axis=1)
                                for channel in range(8):
                                    file['inband_peak_freq_MHz'][event_index,channel] = freqs[band_cut][numpy.argmax(fft[channel][band_cut])]

                            print('\n')
                        else:
                            print('key: %s currently has no hardcoded support in this loop.'%key)

                    if redo_defaults == False:
                        attempt_list = initial_expected_attrs[~numpy.isin(initial_expected_attrs,list(file.attrs.keys()))]
                    else:
                        attempt_list = initial_expected_attrs

                    for key in attempt_list:
                        print('Attempting to add content for key: %s'%key)
                        if key == 'N':
                            file.attrs['N'] = N
                        elif key == 'run':
                            file.attrs['run'] = run
                        else:
                            print('key: %s currently has no hardcoded support in this loop.'%key)
                    file.close()
                except Exception as e:
                    file.close()
                    print('\nError in %s'%inspect.stack()[0][3])
                    print('Error while trying to copy header elements to attrs.')
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

        else:
            print('Creating %s.'%filename )
            with h5py.File(filename, 'w') as file:
                #Prepare attributes for the file.
                file.attrs['N'] = N
                file.attrs['run'] = run
                for key in header_keys_to_copy:
                    try:
                        file.attrs[key] = h[key]
                    except Exception as e:
                        file.close()
                        print('\nError in %s'%inspect.stack()[0][3])
                        print('Error while trying to copy header elements to attrs.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

                #Create datasets that don't require analysis, but might be useful in analysis.
                #When adding things to here, ensure they are also added and handled above as well
                #for when the file already exists. 
                times, subtimes, trigtimes, eventids = getTimes(reader)
                file.create_dataset('eventids', (N,), dtype=numpy.uint32, compression='gzip', compression_opts=9, shuffle=True)
                file['eventids'][...] = eventids

                file.create_dataset('times', (N,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                file['times'][...] = times

                file.create_dataset('subtimes', (N,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                file['subtimes'][...] = subtimes

                file.create_dataset('trigtimes', (N,), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                file['trigtimes'][...] = trigtimes


                file.create_dataset('trigger_types', (N,), dtype=numpy.uint8, compression='gzip', compression_opts=9, shuffle=True)
                file['trigger_types'][...] = loadTriggerTypes(reader)

                #Used for gating obvious backgrounds like known CW
                file.create_dataset('inband_peak_freq_MHz', (N,8), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                reader.setEntry(eventids[0])
                t = reader.t()/1e9
                freqs = numpy.fft.rfftfreq(len(t),t[1]-t[0])/1e6
                band_cut = numpy.logical_and(freqs > 10.0, freqs < 100.0)
                print('Calculating Spectral Frequency\n')
                print_every = int(N/1000)
                for event_index,eventid in enumerate(eventids):
                    if event_index%print_every == 0:
                        sys.stdout.write('\r%i/%i'%(event_index+1,len(eventids)))
                        sys.stdout.flush()
                    reader.setEntry(eventid)
                    wfs = numpy.zeros((8,len(t)))
                    for channel in range(8):
                        wfs[channel] = reader.wf(channel)

                    fft = numpy.fft.rfft(wfs,axis=1)
                    for channel in range(8):
                        file['inband_peak_freq_MHz'][event_index,channel] = freqs[band_cut][numpy.argmax(fft[channel][band_cut])]



        return filename

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def getEventTimes(reader,plot=False):
    '''
    This will hopefully do the appropriate math to determine the real time
    of each event in a run.
    '''
    try:
        #Get Values from Header Tree
        N = reader.head_tree.Draw("trig_time:readout_time:pps_counter","","goff") 
        trig_time = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
        readout_time_head = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
        pps_counter = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)

        #Not using Tree because I want these per event. (there will be repeated values)
        readout_time_status = numpy.zeros(reader.N())
        latched_pps_time = numpy.zeros(reader.N())
        for eventid in range(reader.N()):
            reader.setEntry(eventid)
            readout_time_status[eventid] =  getattr(reader.status(),'readout_time')
            latched_pps_time[eventid] =  getattr(reader.status(),'latched_pps_time')

        unique_latched_pps, indices = numpy.unique(latched_pps_time,return_index=True)
        sample_rate = numpy.diff(unique_latched_pps)
        inter_trig_time = (trig_time[indices][:-1] + trig_time[indices][1:]) / 2.0

        bad_derivatives = sample_rate > 3.15e7 #Ones where a sample was missed or something went wrong.

        f_interpolate_rate = scipy.interpolate.interp1d(inter_trig_time[~bad_derivatives],sample_rate[~bad_derivatives],bounds_error=False,fill_value='extrapolate')

        fractional_second = (trig_time - latched_pps_time)/f_interpolate_rate(trig_time) #THe fraction into the second (beyond pps_counter) this event was.
        second = pps_counter + fractional_second
        #This will then need to be correlated with readout time to get the real world signal that these corresponds to.
        #Right now these correspond to the number of second since the pps immediately prior to the run starting. 
        #Will also need to make some exceptions to handle the first few events which appear weird for some reason. 

        if plot == True:

            plt.figure()
            plt.plot(inter_trig_time[~bad_derivatives],sample_rate[~bad_derivatives])
            plt.xlabel('inter_trig_time')
            plt.ylabel('Sample Rate (Hz)')

            plt.figure()
            plt.plot(trig_time,f_interpolate_rate(trig_time))
            plt.xlabel('trig_time')
            plt.ylabel('Interpolated Clock Rate (Hz)')

            plt.figure()
            plt.plot(trig_time,latched_pps_time)
            plt.xlabel('trig_time')
            plt.ylabel('latched_pps_time')

            plt.figure()
            plt.plot(second)
            plt.ylabel('second (s)')
            plt.xlabel('eventid')

        return seconds

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__=="__main__":
    datapath = os.environ['BEACON_DATA']

    if len(sys.argv) >= 2:
        run = int(sys.argv[1])
        run_label = 'run%i'%run
        redo_defaults = False
        if len(sys.argv) == 3:
            if str(sys.argv[2]) == 'redo':
                redo_defaults = True
                print('WARNING, REDOING DEFAULTS')
        try:
            reader = Reader(datapath,run)
            createFile(reader,redo_defaults=redo_defaults)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    else:
        plt.close('all')
        redo_defaults = False

        runs = [1645]#numpy.arange(1645,1650)

        bins = numpy.linspace(10,100,91)
        width = numpy.diff(bins)
        center = (bins[:-1] + bins[1:]) / 2
        hist = numpy.zeros((8,len(bins)-1))
        for run_index, run in enumerate(runs):
            run = int(run)
            reader = Reader(datapath,run)
            #'''
            trigger_types = loadTriggerTypes(reader)
            print('\nReader:')
            d = interpret.getReaderDict(reader)
            pprint(d)
            print('\nHeader:')
            h = interpret.getHeaderDict(reader)
            pprint(h)
            print('\nStatus:')
            s = interpret.getStatusDict(reader)
            pprint(s)
            #'''
            filename = createFile(reader,redo_defaults=redo_defaults)
            with h5py.File(filename, 'r') as file:
                cut = file['trigger_types'][:] == 2
                for channel in range(8):
                    hist[channel] += numpy.histogram(file['inband_peak_freq_MHz'][:,int(channel)][cut],bins=bins)[0]

                getEventTimes(reader,plot=True) #WHAT I AM CURRENTLY WORKING ON

        if False:
            fig, ax = plt.subplots()
            
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            
            for channel in range(8):
                ax.bar(center, hist[channel], align='center', width=width,label='ch%i'%channel,alpha=0.7)

            plt.legend()
            plt.ylabel('Counts')
            plt.xlabel('Peak Inband Freq (MHz)')
            #ax.set_xticks(bins)
