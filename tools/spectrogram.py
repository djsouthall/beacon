import numpy
import os
import sys
from pprint import pprint
import gc
import multiprocessing
import concurrent.futures
from multiprocessing import cpu_count
from datetime import datetime

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_handler import loadTriggerTypes, getEventTimes


import matplotlib.pyplot as plt
plt.ion()


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

def getFirstDateTime(reader):
    dt = datetime.fromtimestamp(getEventTimes(reader))
    return dt

def getSpectData(datapath,run,event_limit,max_time_min=None,bin_size=10,trigger_type=1,group_fft=False, channels=numpy.arange(8)):
    '''
    This function obtains the data for a spectrogram.

    Parameters
    ----------
    datapath : str
        The path to the data where the runs are stored.  This is the same as the input to
        the reader class.
    run : int
        The run number to be loaded.
    event_limit : int
        This limits the number of events to load.  Loads from beginning of run to end, so
        reducing this speeds up the calculation by cutting off the later portions of the
        run.
    bin_size : int
        This is the number of seconds to include in each time slice of the spectrogram.  The
        average spectra will be computed per bin.  Default is 10.
    trigger_type : int
        This is the trigger type of events included in the spectrogram.  The default is 1.
    group_fft : bool
        This enables the fft calculation to be performed simultaneously for all events, rather
        than per waveform as they are loaded in.  This may be faster but requires more memory.
        Default is False.

    Returns
    -------
    reader : examples.beacon_data_reader.Reader
        This is the reader for the selected run.
    freqs : numpy.ndarray of floats
        This is the list of frequencies for corresponding to the y-axis of the spectrogram data.
    spectra_dbish_binned : dict
        This is the data corresponding to the spectrogram.  Each entry in the dictionary contains
        the spectrogram data for a particular channel.  This are returned in dB-like units.  I.e.
        they are calculated as if the waveforms were in volts, but in reality the waveforms are in
        adu.  Some there is some offset from these values to true dB units.
    '''
    channels = channels.astype(int)
    reader = Reader(datapath,run)

    draw = reader.head_tree.Draw("trig_time:trigger_type","","goff")
    ttypes = loadTriggerTypes(reader)
    trigtimes_s = getEventTimes(reader)

    cut = numpy.isin(ttypes, numpy.asarray(trigger_type))

    # import pdb; pdb.set_trace()
    if max_time_min is not None:
        cut = numpy.logical_and(cut, trigtimes_s < (max_time_min*60 + min(trigtimes_s)))
        # cut = numpy.logical_and(numpy.logical_and(cut, trigtimes_s < (10.97*60 + min(trigtimes_s))), trigtimes_s > (10.92*60 + min(trigtimes_s)))

    trigtimes_s = trigtimes_s[cut]
    eventids = numpy.arange(reader.N())[cut]

    print(eventids)

    if event_limit is not None:
        if event_limit < len(eventids):
            eventids = eventids[0:event_limit]
            trigtimes_s = trigtimes_s[0:event_limit]
    event_limit = len(eventids)

    print('\nReader:')
    d = tools.interpret.getReaderDict(reader)
    pprint(d)
    print('\nHeader:')
    h = tools.interpret.getHeaderDict(reader)
    pprint(h)
    print('\nStatus:')
    s = tools.interpret.getStatusDict(reader)
    pprint(s)


    if reader.N() == 0:
        print('No events found in the selected run.')
    else:
        def rfftWrapper(channel, waveform_times, *args, **kwargs):
            spec = numpy.fft.rfft(*args, **kwargs)
            real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
            if len(numpy.shape(spec)) != 1:
                real_power_multiplier[:,[0,-1]] = 1.0
            else:
                real_power_multiplier[[0,-1]] = 1.0
            spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
            return channel, spec_dbish
        
        waveform_times = reader.t()
        freq_step = 1.0/(len(waveform_times)*(numpy.diff(waveform_times)[0]*1e-9))
        freqs = numpy.arange(len(waveform_times)//2 + 1)*freq_step 
        freq_nyquist = 1/(2.0*numpy.diff(waveform_times)[0]*1e-9)

        if group_fft == True:
            waveforms = {}
        spectra_dbish = {}
        
        for channel in channels:
            if group_fft == True:
                waveforms['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length),dtype=int)
            spectra_dbish['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length//2 + 1),dtype=float)

        print('')

        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('\r(%i/%i)'%(event_index+1,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid) 
            for channel in channels:
                if group_fft == True:
                    waveforms['ch%i'%channel][event_index] = reader.wf(int(channel))
                else:
                    spectra_dbish['ch%i'%channel][event_index] = rfftWrapper('ch%i'%channel, waveform_times, reader.wf(int(channel)))[1]
                    # import pdb; pdb.set_trace()
                # if numpy.any(numpy.isnan(spectra_dbish['ch%i'%channel])):
                #     import pdb; pdb.set_trace()
        if group_fft == True:
            with concurrent.futures.ThreadPoolExecutor(max_workers = cpu_count()) as executor:
                thread_results = []
                for channel in channels:
                    thread_results.append( executor.submit(rfftWrapper,'ch%i'%channel, waveform_times, waveforms['ch%i'%channel]) )
                    
            print('Weaving threads')
            sys.stdout.flush()

            for index, future in enumerate(concurrent.futures.as_completed(thread_results)):
                spectra_dbish[future.result()[0]] = future.result()[1]
                print('Channel %i FFTs Completed'%(index+1))
        
        bin_edges = numpy.arange(min(trigtimes_s),max(trigtimes_s)+bin_size,bin_size)
        bin_L_2d = numpy.tile( bin_edges[:-1] , (len(trigtimes_s),1))
        bin_R_2d = numpy.tile( numpy.roll(bin_edges,-1)[:-1] , (len(trigtimes_s),1))
        trigtimes_s_2d = numpy.tile(trigtimes_s,(len(bin_edges) - 1, 1)).T

        cut_2d = numpy.logical_and(trigtimes_s_2d >= bin_L_2d, trigtimes_s_2d < bin_R_2d).T

        # plt.figure()
        # plt.imshow(cut_2d)

        del bin_L_2d
        del bin_R_2d
        del trigtimes_s_2d

        spectra_dbish_binned = {}
        for channel in channels:
            spectra_dbish_binned['ch%i'%channel] = numpy.zeros((len(freqs),len(bin_edges)-1))
            for index,cut in enumerate(cut_2d):
                spectra_dbish_binned['ch%i'%channel][:,index] = numpy.mean( spectra_dbish['ch%i'%channel][cut], axis=0 )
                # if numpy.any(numpy.isnan(spectra_dbish_binned['ch%i'%channel][:,index])):
                #     import pdb; pdb.set_trace()

            # if numpy.any(numpy.isnan(spectra_dbish_binned['ch%i'%channel])):
            #     import pdb; pdb.set_trace()
            spectra_dbish_binned['ch%i'%channel] = numpy.flipud(numpy.ma.array(spectra_dbish_binned['ch%i'%channel], mask=numpy.isnan(spectra_dbish_binned['ch%i'%channel])))
        

        time_range = (0,(max(bin_edges)-min(bin_edges))/60.0)
        return reader, freqs, spectra_dbish_binned, time_range

if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = os.environ['BEACON_DATA']#sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 367 #Selects which run to examine
    event_limit = 40000
    channels = numpy.array([1])#numpy.arange(8)

    import time

    #runs = numpy.arange(2000,6501,250)
    #runs = numpy.array([4700,5140])
    # runs = numpy.arange(1600,6000,250)
    # runs = numpy.arange(5135,5140,1)
    # runs = numpy.array([3000, 4000, 5140, 5911])
    runs = numpy.array([run])
    cmap = plt.cm.brg
    colors = cmap(numpy.arange(len(runs))/(len(runs)-1))

    spectral_figure = plt.figure()
    spectral_ax = plt.gca()

    cmaps=[ 'coolwarm']#['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)

    Z = []
    skipped_runs = []
    datetimes = []

    trigtimes_s = getEventTimes(reader)

    for run_index, run in enumerate(runs):
        try:
            reader, freqs, spectra_dbish_binned, time_range = getSpectData(datapath,run,event_limit,bin_size=10,group_fft=False, channels=channels)

            gc.collect()

            for cmap in cmaps:
                for channel in channels:
                    if True:
                        f = plt.figure(figsize=(12,6))
                        # ax = plt.subplot(2,1,1)
                        plt.title('Run %i, Channel %i'%(run,channel),fontsize=28)


                        plt.imshow(spectra_dbish_binned['ch%i'%channel],extent = [0,(max(trigtimes_s)-min(trigtimes_s))/60.0,min(freqs)/1e6,max(freqs)/1e6],aspect='auto',cmap=cmap)
                        #plt.xlim(0,100)
                        plt.ylabel('Freq (MHz)',fontsize=20)
                        plt.xlabel('Readout Time (min)',fontsize=20)
                        cb = plt.colorbar()
                        cb.set_label('dB (arb)',fontsize=20)
                        #cb.set_label('Power (~dB)',fontsize=20)
                        #f.savefig('./spectrogram_run%i_ch%i_20MHz-100MHz_cmap%s.pdf'%(run,channel,cmap), bbox_inches='tight')
                        #plt.close(f)

                    if True:
                        #ax = plt.subplot(2,1,2)
                        if run_index == 0:
                            spectral_ax.set_xlabel('Freqs (MHz)')

                        avg = numpy.mean(spectra_dbish_binned['ch%i'%channel], axis=1)[::-1]


                        Z.append(avg)
                        dt = getFirstDateTime(reader)
                        datetimes.append(dt)
                        spectral_ax.plot(freqs/1e6, avg, c=colors[run_index], label='Run %i, ~ %s'%(run, str(dt.date())))
                        spectral_ax.set_xlim(0,150)
                        spectral_ax.set_ylim(-10,40)
        except Exception as e:
            print(e)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            print('Skipping %i'%run)
            skipped_runs.append(run)

    spectral_ax.legend()

    Z = numpy.asarray(Z)
    Z[numpy.isinf(Z)] = numpy.nan

    run_cut = ~numpy.isin(runs, skipped_runs)
    runs = runs[run_cut]
    datetimes = numpy.asarray(datetimes)[run_cut]


    X, Y = numpy.meshgrid(freqs/1e6, runs)

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # surf = ax.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, vmin=numpy.nanmin(Z), vmax=numpy.nanmax(Z), linewidth=1, antialiased=False)
    # # ax.plot_wireframe(X, Y, Z, rcount=len(runs), ccount=0, color='k', alpha=0.4, linewidth=0.5)


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(X, Y, Z, rcount=len(runs), ccount=0, color='k', alpha=1.0, linewidth=0.5)

    draw = reader.head_tree.Draw("trig_time:trigger_type","","goff")
    ttypes = trigger_type = numpy.frombuffer(reader.head_tree.GetV2(), numpy.dtype('float64'), draw).astype(int)
    trigtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), draw)[ttypes == trigger_type]

    raw_approx_trigger_time, raw_approx_trigger_time_nsecs, trig_time, eventids = getTimes(reader)
    t = getEventTimes(reader)
    t_mins = (t - min(t))/60

    cut_48 =  numpy.logical_and(numpy.logical_and(t_mins >= 18.4, t_mins <= 18.67), ttypes==1)

    print(run)
    print(eventids[cut_48][0:10])

    cut_42 =  numpy.logical_and(numpy.logical_and(t_mins >= 21.24, t_mins <= 21.92), ttypes==1)

    print(eventids[cut_42][0:10])

    # [14120 14121 14122 14123 14124 14125 14126 14127 14128 14129]
    # [16180 16181 16182 16183 16184 16185 16186 16187 16188 16189]
