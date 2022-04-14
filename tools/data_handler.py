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
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.  This shoild be the raw one and not the sine subtracted one. 
import numpy
import h5py
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.signal

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import pdb


def loadTriggerTypes(reader):
    '''
    Will get a list of trigger types corresponding to all eventids for the given reader
    trigger_type:
    1 Software
    2 RF
    3 GPS
    '''
    #trigger_type = numpy.zeros(reader.N())
    '''
    N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff") 
    #ROOT.gSystem.ProcessEvents()
    subtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
    times = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
    trigtimes = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
    eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)
    '''


    try:
        N = reader.head_tree.Draw("trigger_type","","goff") 
        trigger_type = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N).astype(int)
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print('Error while trying to copy header elements to attrs.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e
    return trigger_type

def getTimes(reader):
    '''
    This pulls timing information for each event from the reader object..
    
    Parameters
    ----------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.

    Returns
    -------
    raw_approx_trigger_time : numpy.ndarray of floats
        The raw_approx_trigger_time values for each event from the Tree.
    raw_approx_trigger_time_nsecs : numpy.ndarray of floats
        The raw_approx_trigger_time_nsecs values for each event from the Tree. 
    trig_time : numpy.ndarray of floats
        The trig_time values for each event from the Tree.
    '''
    try:
        N = reader.head_tree.Draw("raw_approx_trigger_time_nsecs:raw_approx_trigger_time:trig_time:Entry$","","goff") 
        #ROOT.gSystem.ProcessEvents()
        raw_approx_trigger_time_nsecs = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
        raw_approx_trigger_time = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N) 
        trig_time = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)
        eventids = numpy.frombuffer(reader.head_tree.GetV4(), numpy.dtype('float64'), N).astype(int)

        return raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print('Error while trying to copy header elements to attrs.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e

def getEventTimes(reader,plot=False,smooth_window=101):
    '''
    This will hopefully do the appropriate math to determine the real time
    of each event in a run.

    Smoothing will be performed on the rates using the specified smooth window.
    To disable this set the smooth window to None.
    '''
    try:
        #Get Values from Header Tree
        N = reader.head_tree.Draw("readout_time:readout_time_ns","","goff") 
        readout_time_head = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N) + numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N)/1e9  

        N = reader.head_tree.Draw("trig_time:pps_counter:trigger_type","","goff") 
        trig_time = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), N)
        pps_counter = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), N)
        trigger_type = numpy.frombuffer(reader.head_tree.GetV3(), numpy.dtype('float64'), N)

        #Get Values from Status Tree
        N = reader.status_tree.Draw("latched_pps_time","","goff") 
        _latched_pps_time = numpy.frombuffer(reader.status_tree.GetV1(), numpy.dtype('float64'), N)
        
        #latched_pps_time are not present for every event, but I want an array which contains the most recent latched_pps_time
        #per event.  Below searchsorted returns the index of the _latched_pps_time that each trigtime would be placed to remain
        #sorted.  This means that these indices correspond the the values in _latched_pps_time that are just larger than trig_time
        #a(as if they were bumped to the right, and the trig time was inserted there, it would still be sorted).  This picking those
        #indices - 1 you get the _latched_pps_time for each event that is just to the left of the trig_time for each trig_time.  
        latched_pps_time = _latched_pps_time[numpy.searchsorted(_latched_pps_time,trig_time,side='left') - 1]
        

        '''
        #Not using Tree because I want these per event. (there will be repeated values)
        readout_time_status = numpy.zeros(reader.N())
        #latched_pps_time = numpy.zeros(reader.N())
        for eventid in range(reader.N()):
            reader.setEntry(eventid)
            readout_time_status[eventid] =  getattr(reader.status(),'readout_time')
            #latched_pps_time[eventid] =  getattr(reader.status(),'latched_pps_time')

        '''

        #The next few steps aim to obtain the instantaneous clock rate.  Some work has to be done to ignore anomolies like skipped latched_pps_times for instance.
        unique_latched_pps, indices = numpy.unique(latched_pps_time,return_index=True)

        #Sorting unique values by index
        indices = numpy.sort(indices)
        unique_latched_pps = latched_pps_time[indices]
        
        if plot == True:
            plt.figure()
            plt.ylabel('Calculated Clock Rate MHz)',fontsize=16)
            plt.xlabel('trig_time',fontsize=16)
            plt.plot((trig_time[indices][:-1] + trig_time[indices][1:]) / 2.0 ,  numpy.diff(unique_latched_pps)/1e6,label='Before removing anomolies')

        #Sometimes the beginning of the run can not be properly handled.  The latched pps will start very high.
        #I ignore these values and then hope to get the resulting values later by interpolating at a later stage. 
        problematic_latched_pps_cut = numpy.append(numpy.diff(unique_latched_pps) < 0,False)


        indices = indices[~problematic_latched_pps_cut]
        unique_latched_pps = unique_latched_pps[~problematic_latched_pps_cut]


        #Getting first pass sample rate
        sample_rate = numpy.diff(unique_latched_pps) #Results in array with len = len(unique_latched_pps) - 1

        #Calculating trig times between original trig times to correspond to len = len(unique_latched_pps) - 1
        inter_trig_time = (trig_time[indices][:-1] + trig_time[indices][1:]) / 2.0 
                
        bad_derivatives = sample_rate > 3.15e7 #Ones where a latched_pps_time sample was missed or something went wrong, so the rate is artificially about double.

        if plot == True:
            plt.plot(inter_trig_time[~bad_derivatives],  sample_rate[~bad_derivatives]/1e6,label='Negative and Doubled\nClock Rates Removed')
            plt.legend()

            plt.figure()
            plt.ylabel('Calculated Clock Rate MHz)',fontsize=16)
            plt.xlabel('trig_time',fontsize=16)
            plt.plot(inter_trig_time[~bad_derivatives],  sample_rate[~bad_derivatives]/1e6,label='Negative and Doubled\nClock Rates Removed',c=plt.rcParams['axes.prop_cycle'].by_key()['color'][1])
            plt.legend()

        #The below will smooth out the rate to let interpolation and extrapolation work better than with the jittering rate.
        if smooth_window is not None:
            #Smoothing out rate
            hamming_filter = scipy.signal.hamming(smooth_window)
            hamming_filter = hamming_filter/sum(hamming_filter)
            padded = numpy.append(numpy.append(numpy.ones(smooth_window//2)*sample_rate[~bad_derivatives][0],sample_rate[~bad_derivatives]),numpy.ones(smooth_window//2)*sample_rate[~bad_derivatives][-1])
            smoothed_rate = numpy.convolve(padded,hamming_filter, mode='valid')
        else:
            smoothed_rate = sample_rate[~bad_derivatives]
        
        #The following interpolates the smoothed rate.
        f_interpolate_rate = scipy.interpolate.interp1d(inter_trig_time[~bad_derivatives],smoothed_rate,bounds_error=False,fill_value=(smoothed_rate[0],smoothed_rate[-1]))

        #This will extrapolate the leading edge where some problems may have occurred in latched_pps_time.
        latched_pps_time[trig_time <= latched_pps_time] = scipy.interpolate.interp1d(trig_time[trig_time > latched_pps_time],latched_pps_time[trig_time > latched_pps_time],bounds_error=False,fill_value='extrapolate')(trig_time[trig_time <= latched_pps_time])

        fractional_second = (trig_time - latched_pps_time)/f_interpolate_rate(trig_time) #THe fraction into the second (beyond pps_counter) this event was.
        fractional_second -= numpy.floor(fractional_second) #Corrects for cases where the fractional second > 1 (for skipped latched_pps)
        second = pps_counter + fractional_second

        actual_event_time_seconds = numpy.floor(numpy.mean(readout_time_head - second)) + second


        if False:
            plt.figure()
            ax = plt.subplot(2,1,1)
            plt.plot(trig_time,trig_time,label='trig_time')
            plt.plot(trig_time,latched_pps_time,label='latched_pps_time')
            for v in inter_trig_time[bad_derivatives]:
                plt.axvline(v,c='r')
            plt.legend()
            plt.subplot(2,1,2,sharex=ax)
            plt.plot(trig_time, trig_time - latched_pps_time,label='trig_time - latched_pps_time')
            plt.plot(trig_time, f_interpolate_rate(trig_time),label='rate')
            for v in inter_trig_time[bad_derivatives]:
                plt.axvline(v,c='r')
            plt.legend()
            plt.ylabel('trig_time - latched_pps_time')
            plt.xlabel('trigtime')

        if plot == True:

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(readout_time_head - second)
            plt.axhline(numpy.mean(readout_time_head - second),label='mean = %f\nstd = %f'%(numpy.mean(readout_time_head - second), numpy.std(readout_time_head - second)),linestyle='--',c='r')
            plt.ylabel('readout_time_head - second')
            plt.xlabel('eventid')
            plt.legend()
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            bins = numpy.linspace(min(readout_time_head - second),max((readout_time_head - second)),100)

            for t_index, t in enumerate([1,2,3]):
                if t_index == 0:
                    ax = plt.subplot(2,3,3+t)
                else:
                    plt.subplot(2,3,3+t,sharex=ax,sharey=ax)

                cut = trigger_type == t
                y = (readout_time_head - second)[cut]
                plt.hist(y,bins=bins,alpha=0.8,label='Trigger type = %i'%t)
                mean = numpy.mean(y)
                std = numpy.std(y)
                plt.axvline(mean,label='mean = %f\nstd = %f'%(mean,std),linestyle='--',c='r')
                plt.legend()
                plt.xlabel('readout_time_head - second')
                plt.ylabel('Counts')
                plt.yscale('log', nonposy='clip')

                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


            plt.figure()
            plt.plot(inter_trig_time[~bad_derivatives],sample_rate[~bad_derivatives],label='Original')
            if smooth_window is not None:
                plt.plot(inter_trig_time[~bad_derivatives],smoothed_rate,label='Smoothed with len(hamming)=%i'%smooth_window)
            plt.legend()
            
            plt.xlabel('inter_trig_time')
            plt.ylabel('Sample Rate (Hz)')

            '''
            plt.figure()
            plt.plot(trig_time,f_interpolate_rate(trig_time))
            plt.xlabel('trig_time')
            plt.ylabel('Interpolated Clock Rate (Hz)')

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(trig_time,latched_pps_time,label='All latched pps as stored')
            plt.xlabel('trig_time')
            plt.ylabel('All latched_pps_time')
            plt.legend(loc='upper center')
            plt.subplot(2,1,2)
            plt.plot(trig_time[indices],latched_pps_time[indices],label='Only latched pps used in\ncalculating seconds')
            plt.xlabel('trig_time')
            plt.ylabel('Used latched_pps_time')
            plt.legend(loc='upper center')
            '''

            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(pps_counter)
            plt.ylabel('pps_counter (s)')
            plt.xlabel('eventid')

            plt.subplot(2,1,2)
            plt.plot(fractional_second)
            plt.ylabel('fractional_second (s)')
            plt.xlabel('eventid')

            plt.figure()
            plt.plot(second)
            plt.ylabel('second (s)')
            plt.xlabel('eventid')

        return actual_event_time_seconds

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e



def createFile(reader, verbose=True, redo_defaults=False, check_defaults=False, analysis_data_dir=None):
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
    check_defaults : bool
        If True and the file exists, then this will open the file ante check if the expected datasets exist.  By default
        this is False, as hdf5 files seem to have a habit of corrupting.  By limiting unecessary file openings I hope
        to reduce that.  This needs to be True if you want outdated_datasets_to_remove to be checked.
    '''
    try:
        run = int(reader.run)
        N = reader.N()
        if N > 0:
            if analysis_data_dir is None:
                analysis_data_dir = os.environ['BEACON_PROCESSED_DATA']
                
            filename = os.path.join(analysis_data_dir,'run%i_analysis_data.h5'%run)

            header_keys_to_copy = []
            h = interpret.getHeaderDict(reader)

            initial_expected_datasets = numpy.array(['eventids','trigger_type','raw_approx_trigger_time','raw_approx_trigger_time_nsecs','trig_time','calibrated_trigtime','p2p','std'])#,'inband_peak_freq_MHz' #expand as more things are added.  This should only include datasets that this function will add.
            initial_expected_attrs    = numpy.array(['N','run'])

            #outdated_datasets_to_remove should include things that were once in each file but should no longer be.  They might be useful if a dataset
            #is given a new name and you want to delete the old dataset for instance. 
            #If a data set is in both initial_expected_datasets and outdated_datasets_to_remove it will be removed FIRST, then recalculated.
            #This has been used in the past to correct a type error in storing values (storing std as int instead of float),
            #where the data set cannot only be recalculated, but must have the type changed.  
            outdated_datasets_to_remove = numpy.array(['trigger_tpes','times','subtimes','trigtimes','hpol_impulsivity','vpol_impulsivity']) #Currently working on similarity.  Will want the old one to be in here after complete.

            if os.path.exists(filename):

                if check_defaults == True or redo_defaults == True:
                    print('%s already exists, checking if setup is up to date.'%filename )
                    with h5py.File(filename, 'a') as file:
                        try:
                            remove_list = outdated_datasets_to_remove[numpy.isin(outdated_datasets_to_remove,list(file.keys()))]
                            for key in remove_list:
                                print('Removing old dataset: %s'%key)
                                try:
                                    del file[key]
                                except:
                                    continue

                            if redo_defaults == False:
                                attempt_list = initial_expected_datasets[~numpy.isin(initial_expected_datasets,list(file.keys()))]
                            else:
                                attempt_list = initial_expected_datasets

                            if numpy.any(numpy.isin(attempt_list,['eventids','raw_approx_trigger_time','raw_approx_trigger_time_nsecs','trig_time','inband_peak_freq_MHz','p2p','std'])):
                                times_loaded = True
                                raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids = getTimes(reader)


                            for key_index, key in enumerate(attempt_list):
                                if key_index == 0:
                                    print('attempt_list = ',attempt_list)
                                print('Attempting to add content for key: %s'%key)
                                if key == 'eventids':
                                    if ('eventids' in list(file.keys())) == False:
                                        file.create_dataset('eventids', (N,), dtype=numpy.uint32, compression='gzip', compression_opts=9, shuffle=True)
                                    file['eventids'][...] = eventids
                                elif key == 'trigger_type':
                                    if ('trigger_type' in list(file.keys())) == False:
                                        file.create_dataset('trigger_type', (N,), dtype=numpy.uint8, compression='gzip', compression_opts=9, shuffle=True)
                                    file['trigger_type'][...] = loadTriggerTypes(reader)
                                elif key == 'raw_approx_trigger_time':
                                    if ('raw_approx_trigger_time'  in list(file.keys())) == False:
                                        file.create_dataset('raw_approx_trigger_time', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                                    file['raw_approx_trigger_time'][...] = raw_approx_trigger_time
                                elif key == 'raw_approx_trigger_time_nsecs':
                                    if ('raw_approx_trigger_time_nsecs'  in list(file.keys())) == False:
                                        file.create_dataset('raw_approx_trigger_time_nsecs', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                                    file['raw_approx_trigger_time_nsecs'][...] = raw_approx_trigger_time_nsecs
                                elif key == 'trig_time':
                                    if ('trig_time'  in list(file.keys())) == False:
                                        file.create_dataset('trig_time', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                                    file['trig_time'][...] = trig_time
                                elif key == 'calibrated_trigtime':
                                    if ('calibrated_trigtime'  in list(file.keys())) == False:
                                        file.create_dataset('calibrated_trigtime', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                                    file['calibrated_trigtime'][...] = getEventTimes(reader)
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
                                elif key == 'p2p':
                                    if ('p2p'  in list(file.keys())) == False:
                                        file.create_dataset('p2p', (N,8), dtype=numpy.uint32, compression='gzip', compression_opts=9, shuffle=True)
                                    reader.setEntry(eventids[0])
                                    t = reader.t()/1e9
                                    print('Calculating Peak to Peak\n')
                                    print_every = int(N/1000)
                                    for event_index,eventid in enumerate(eventids):
                                        if event_index%print_every == 0:
                                            sys.stdout.write('\r%i/%i'%(event_index+1,len(eventids)))
                                            sys.stdout.flush()

                                        reader.setEntry(eventid)
                                        for channel in range(8):
                                            wf = reader.wf(channel)
                                            file['p2p'][event_index,channel] = numpy.max(wf) - numpy.min(wf)

                                    print('\n')
                                elif key == 'std':
                                    if ('std'  in list(file.keys())) == False:
                                        file.create_dataset('std', (N,8), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                                    reader.setEntry(eventids[0])
                                    t = reader.t()/1e9
                                    print('Calculating std\n')
                                    print_every = int(N/1000)
                                    for event_index,eventid in enumerate(eventids):
                                        if event_index%print_every == 0:
                                            sys.stdout.write('\r%i/%i'%(event_index+1,len(eventids)))
                                            sys.stdout.flush()

                                        reader.setEntry(eventid)
                                        for channel in range(8):
                                            wf = reader.wf(channel).astype(numpy.float64)
                                            file['std'][event_index,channel] = numpy.std(wf)

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
                            return e

                else:
                    if verbose:
                        print('%s already exists, a check of contents was NOT performed.'%filename )
            else:
                print('Creating %s.'%filename )
                with h5py.File(filename, 'w') as file:
                    try:
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
                                return e

                        #Create datasets that don't require analysis, but might be useful in analysis.
                        #When adding things to here, ensure they are also added and handled above as well
                        #for when the file already exists. 
                        raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids = getTimes(reader)
                        file.create_dataset('eventids', (N,), dtype=numpy.uint32, compression='gzip', compression_opts=9, shuffle=True)
                        file['eventids'][...] = eventids

                        file.create_dataset('raw_approx_trigger_time', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                        file['raw_approx_trigger_time'][...] = raw_approx_trigger_time

                        file.create_dataset('raw_approx_trigger_time_nsecs', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                        file['raw_approx_trigger_time_nsecs'][...] = raw_approx_trigger_time_nsecs

                        file.create_dataset('trig_time', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                        file['trig_time'][...] = trig_time

                        file.create_dataset('trigger_type', (N,), dtype=numpy.uint8, compression='gzip', compression_opts=9, shuffle=True)
                        file['trigger_type'][...] = loadTriggerTypes(reader)

                        file.create_dataset('calibrated_trigtime', (N,), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
                        file['calibrated_trigtime'][...] = getEventTimes(reader)

                        #Used for gating obvious backgrounds like known CW
                        file.create_dataset('inband_peak_freq_MHz', (N,8), dtype='f', compression='gzip', compression_opts=9, shuffle=True)
                        file.create_dataset('p2p', (N,8), dtype=numpy.uint32, compression='gzip', compression_opts=9, shuffle=True)
                        file.create_dataset('std', (N,8), dtype=numpy.float64, compression='gzip', compression_opts=9, shuffle=True)
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
                            wfs = numpy.zeros((8,len(t)),dtype=numpy.float64)
                            for channel in range(8):
                                wfs[channel] = reader.wf(channel)
                                file['p2p'][event_index,channel] = numpy.max(wfs[channel]) - numpy.min(wfs[channel])
                                file['std'][event_index,channel] = numpy.std(wfs[channel])

                            fft = numpy.fft.rfft(wfs,axis=1)
                            for channel in range(8):
                                file['inband_peak_freq_MHz'][event_index,channel] = freqs[band_cut][numpy.argmax(fft[channel][band_cut])]
                        file.close()
                    except Exception as e:
                        file.close()
                        print('\nError in %s'%inspect.stack()[0][3])
                        print('Error while trying to generate new file.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                        return e

                                    



            return filename

        else:
            print('Empty Tree, returning None')
            return None

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        return e


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
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
            createFile(reader,redo_defaults=redo_defaults)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)

    else:
        try:
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

                try:
                    print(reader.status())
                except Exception as e:
                    print('Status Tree not present.  Returning Error.')
                    print('\nError in %s'%inspect.stack()[0][3])
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    sys.exit(1)
                #'''
                trigger_type = loadTriggerTypes(reader)
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
                getEventTimes(reader,plot=True) #WHAT I AM CURRENTLY WORKING ON
                '''
                filename = createFile(reader,redo_defaults=redo_defaults)
                with h5py.File(filename, 'r') as file:
                    cut = file['trigger_type'][:] == 2
                    for channel in range(8):
                        hist[channel] += numpy.histogram(file['inband_peak_freq_MHz'][:,int(channel)][cut],bins=bins)[0]
                '''


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
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            sys.exit(1)
    sys.exit(0)
