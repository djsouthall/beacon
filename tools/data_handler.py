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
                            if ('eventids' in list(file.keys())) == False:
                                file.create_dataset('eventids', (N,), dtype='i', compression='gzip', compression_opts=9, shuffle=True)
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
                file.create_dataset('eventids', (N,), dtype='i', compression='gzip', compression_opts=9, shuffle=True)
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
        runs = numpy.array([1645])#numpy.arange(1645,1700)

        for run in runs:
            run = int(run)

            reader = Reader(datapath,run)
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

            filename = createFile(reader,redo_defaults=redo_defaults)

            with h5py.File(filename, 'r') as file:
                cut = file['trigger_types'][:] == 2
                for channel in range(8):
                    peaks = file['inband_peak_freq_MHz'][:,int(channel)][cut]
                    e = numpy.where(cut)[0]
                    plt.figure()
                    plt.hist(peaks,bins=200)
                    plt.xlabel('MHz')
                    plt.ylabel('Counts')
                    print(e[peaks == 0])
