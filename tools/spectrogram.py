import numpy
import os
import sys
from pprint import pprint
import gc
import multiprocessing
import concurrent.futures
from multiprocessing import cpu_count

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.

import matplotlib.pyplot as plt
plt.ion()

def getSpectData(datapath,run,event_limit,bin_size=10,trigger_type=1,group_fft=False, channels=numpy.arange(8)):
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
    ttypes = trigger_type = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), draw).astype(int)
    trigtimes = numpy.frombuffer(reader.head_tree.GetV1(), numpy.dtype('float64'), draw)[ttypes == trigger_type]
    eventids = numpy.arange(reader.N())[ttypes == trigger_type]


    if event_limit is not None:
        if event_limit < len(eventids):
            eventids = eventids[0:event_limit]
            trigtimes = trigtimes[0:event_limit]
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
        readout_times = []
        
        for channel in channels:
            if group_fft == True:
                waveforms['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length),dtype=int)
            spectra_dbish['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length//2 + 1),dtype=float)

        print('')

        for event_index, eventid in enumerate(eventids):
            sys.stdout.write('\r(%i/%i)'%(eventid+1,len(eventids)))
            sys.stdout.flush()
            reader.setEntry(eventid) 
            readout_times.append(getattr(reader.header(),'readout_time'))
            for channel in channels:
                if group_fft == True:
                    waveforms['ch%i'%channel][event_index] = reader.wf(int(channel))
                else:
                    spectra_dbish['ch%i'%channel][event_index] = rfftWrapper('ch%i'%channel, waveform_times, reader.wf(int(channel)))[1]
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
        
        bin_edges = numpy.arange(min(readout_times),max(readout_times)+bin_size,bin_size)
        bin_L_2d = numpy.tile( bin_edges[:-1] , (len(readout_times),1))
        bin_R_2d = numpy.tile( numpy.roll(bin_edges,-1)[:-1] , (len(readout_times),1))
        readout_times_2d = numpy.tile(readout_times,(len(bin_edges) - 1, 1)).T

        cut_2d = numpy.logical_and(readout_times_2d >= bin_L_2d, readout_times_2d < bin_R_2d).T

        del bin_L_2d
        del bin_R_2d
        del readout_times_2d

        spectra_dbish_binned = {}
        for channel in channels:
            spectra_dbish_binned['ch%i'%channel] = numpy.zeros((len(freqs),len(bin_edges)-1))
            for index,cut in enumerate(cut_2d):
                spectra_dbish_binned['ch%i'%channel][:,index] = numpy.mean( spectra_dbish['ch%i'%channel][cut], axis=0 )
            spectra_dbish_binned['ch%i'%channel] = numpy.flipud(numpy.ma.array(spectra_dbish_binned['ch%i'%channel], mask=numpy.isnan(spectra_dbish_binned['ch%i'%channel])))
        

        time_range = (0,(max(bin_edges)-min(bin_edges))/60.0)
        return reader, freqs, spectra_dbish_binned, time_range

if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = os.environ['BEACON_DATA']#sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = int(sys.argv[1]) if len(sys.argv) > 1 else 367 #Selects which run to examine
    event_limit = None

    import time

    reader, freqs, spectra_dbish_binned, time_range = getSpectData(datapath,run,event_limit,bin_size=100,group_fft=False)

    gc.collect()

    cmaps=[ 'coolwarm']#['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    for cmap in cmaps:
        for channel in range(8):
            f = plt.figure(figsize=(12,6))
            ax = plt.gca()
            plt.title('Run %i, Channel %i'%(run,channel),fontsize=28)
            plt.imshow(spectra_dbish_binned['ch%i'%channel],extent = [0,(reader.head_tree.GetMaximum('readout_time')-reader.head_tree.GetMinimum('readout_time'))/60.0,min(freqs)/1e6,max(freqs)/1e6],aspect='auto',cmap=cmap)
            #plt.xlim(0,100)
            plt.ylabel('Freq (MHz)',fontsize=20)
            plt.xlabel('Readout Time (min)',fontsize=20)
            cb = plt.colorbar()
            cb.set_label('dB (arb)',fontsize=20)
            #cb.set_label('Power (~dB)',fontsize=20)
            #f.savefig('./spectrogram_run%i_ch%i_20MHz-100MHz_cmap%s.pdf'%(run,channel,cmap), bbox_inches='tight')
            #plt.close(f)

    #,cmap='RdGy'
