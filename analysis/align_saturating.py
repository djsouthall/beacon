'''
This script uses the events found by find_good_saturating_signals.py to determine some good
expected time delays between antennas.  These can be used then as educated guesses for time
differences in the antenna_timings.py script. 
'''

import numpy
import scipy.spatial
import scipy.signal
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
plt.ion()

#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.
known_pulser_ids = {
    'run793':{
        'eventids':numpy.array([ 98958, 102883, 101954, 110633, 106755, 102863, 110415, 110465, 110372,  98075,\
                                 97898,  94907,  98402, 110747, 102796,  99008,  98001,  98061, 102560, 110606,\
                                 98490, 106395,  95429,  98221, 102745, 102344,  99025,  97850,  98835, 110347,\
                                110584, 105617,  97796,  98799,  98862,  98844,  98983, 106727, 105101, 100552,\
                                 93623, 108037, 106412, 102165,  99891, 109519,  97727, 110075, 109530, 108953,\
                                 97600,  97702, 107065, 104092,  94728, 100619,  98639,  99391,  96539, 109268,\
                                 99165, 100856,  93987, 101844, 103264, 101564, 100344,  93551, 108833,  93954,\
                                104061, 110160, 109106,  97669, 109310,  93438, 101024, 105723, 106431,  96802,\
                                 99406, 101578,  97436, 106103, 105560, 103241, 100756,  96715, 109905,  96701,\
                                109942, 103994, 102927,  95365, 100724,  96378, 100670, 105890, 101183, 107502,\
                                 94604, 106471,  95357, 107786, 106486,  97078,  93685,  95556, 104258, 103939,\
                                 94492, 103399,  96137]),
        'clock_rate':31249809.22371152
            }
}



def rfftWrapper(waveform_times, *args, **kwargs):
    spec = numpy.fft.rfft(*args, **kwargs)
    real_power_multiplier = 2.0*numpy.ones_like(spec) #The factor of 2 because rfft lost half of the power except for dc and Nyquist bins (handled below).
    if len(numpy.shape(spec)) != 1:
        real_power_multiplier[:,[0,-1]] = 1.0
    else:
        real_power_multiplier[[0,-1]] = 1.0
    spec_dbish = 10.0*numpy.log10( real_power_multiplier*spec * numpy.conj(spec) / len(waveform_times)) #10 because doing power in log.  Dividing by N to match monutau. 
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    return freqs, spec_dbish

def makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=False):
    dt = waveform_times[1] - waveform_times[0]
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    b, a = scipy.signal.butter(filter_order, crit_freq_low_pass_MHz*1e6, 'low', analog=True)
    d, c = scipy.signal.butter(filter_order, crit_freq_high_pass_MHz*1e6, 'high', analog=True)

    filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
    filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
    filter_x = freqs
    filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
    return filter_y, freqs

def loadSignals(reader,eventid,filter_y):
    try:
        reader.setEntry(eventid)
        raw = numpy.zeros((8,reader.header().buffer_length))
        filter_y = numpy.tile(filter_y,(8,1))
        for channel in range(8):
            #Load waveform
            raw[channel] = reader.wf(channel)
        #Upsample
        upsampled = scipy.signal.resample(raw,2*(numpy.shape(filter_y)[1]-1),axis=1)
        #Apply filter
        upsampled = numpy.fft.irfft(numpy.multiply(filter_y,numpy.fft.rfft(upsampled,axis=1)),axis=1)
    except Exception as e:
        print('Error in loadSignals')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    return raw, upsampled


expected_time_differences = [((0, 1), 13.508319251205194), ((0, 2), 17.170337798788978), ((0, 3), 33.424724259292816), ((1, 2), 30.67865704999417), ((1, 3), 46.93304351049801), ((2, 3), 16.254386460503838)]
max_time_differences = [((0, 1), 22.199031286793637), ((0, 2), 35.09576572192467), ((0, 3), 41.32087648347992), ((1, 2), 33.39469529610869), ((1, 3), 47.45680773722745), ((2, 3), 17.40978039384259)]


if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    use_known_ids = True
    
    resample_factor = 100

    #Filter settings
    crit_freq_low_pass_MHz = 75
    crit_freq_high_pass_MHz = 15
    filter_order = 6
    plot_filter = True
    use_envelopes = False
    use_raw = True
    bins = 200
    expected_timing_pm_tol = 20 #ns
    corr_plot = True

    for run_index, run in enumerate(runs):
        eventids = numpy.sort(known_pulser_ids['run%i'%run]['eventids'])
        reader = Reader(datapath,run)

        waveform_times = reader.t()
        waveforms_upsampled = {}
        waveforms_raw = {}

        #Prepare filter
        reader.setEntry(eventids[0])
        wf = reader.wf(0)
        if use_raw:
            if resample_factor != 1:
                print('\n!!!\nUsing raw waveforms for alignment.  Setting the resample factor to 1.\n!!!\n') 
                resample_factor = 1

        wf , waveform_times = scipy.signal.resample(wf,len(wf)*resample_factor,t=reader.t())
        dt = waveform_times[1] - waveform_times[0]
        filter_y,freqs = makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=plot_filter)

        try:
            for channel in range(8):
                waveforms_upsampled['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length*resample_factor))
                waveforms_raw['ch%i'%channel] = numpy.zeros((len(eventids),reader.header().buffer_length))

            print('Loading Waveforms_upsampled')

            #plt.figure()
            #ax = plt.subplot(4,2,1)

            for waveform_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\r'%(waveform_index,len(eventids)))
                sys.stdout.flush()
                reader.setEntry(eventid)

                raw_wfs, upsampled_wfs = loadSignals(reader,eventid,filter_y)

                for channel in range(8):
                    if use_envelopes == True:
                        waveforms_upsampled['ch%i'%channel][waveform_index] = numpy.abs(scipy.signal.hilbert(upsampled_wfs[channel])) 
                        waveforms_raw['ch%i'%channel][waveform_index] = numpy.abs(scipy.signal.hilbert(raw_wfs[channel]))                     
                    else:
                        waveforms_upsampled['ch%i'%channel][waveform_index] = upsampled_wfs[channel]
                        waveforms_raw['ch%i'%channel][waveform_index] = raw_wfs[channel]

            delays_even = numpy.zeros((len(eventids),4,4))
            max_corrs_even = numpy.zeros((len(eventids),4,4))
            delays_odd = numpy.zeros((len(eventids),4,4))
            max_corrs_odd = numpy.zeros((len(eventids),4,4))
            print('Cross correlating events between antennas.')

            if corr_plot:
                fig_even = plt.figure()
                plt.suptitle('Hpol')
                fig_odd = plt.figure()
                plt.suptitle('Vpol')


            for event_index in range(len(eventids)):
                sys.stdout.write('\r(%i/%i)'%(event_index,len(eventids)))
                sys.stdout.flush()


                for i in range(4):
                    if use_raw:
                        x_even = waveforms_raw['ch%i'%(2*i)][event_index]/numpy.std(waveforms_raw['ch%i'%(2*i)][event_index])
                        x_odd = waveforms_raw['ch%i'%(2*i+1)][event_index]/numpy.std(waveforms_raw['ch%i'%(2*i+1)][event_index])
                    else:
                        x_even = waveforms_upsampled['ch%i'%(2*i)][event_index]/numpy.std(waveforms_upsampled['ch%i'%(2*i)][event_index])
                        x_odd = waveforms_upsampled['ch%i'%(2*i+1)][event_index]/numpy.std(waveforms_upsampled['ch%i'%(2*i+1)][event_index])
                    for j in range(4):
                        if j >= i:
                            continue
                        else:
                            if use_raw:
                                y_even = waveforms_raw['ch%i'%(2*j)][event_index]/numpy.std(waveforms_raw['ch%i'%(2*j)][event_index])
                                y_odd = waveforms_raw['ch%i'%(2*j+1)][event_index]/numpy.std(waveforms_raw['ch%i'%(2*j+1)][event_index])
                            else:
                                y_even = waveforms_upsampled['ch%i'%(2*j)][event_index]/numpy.std(waveforms_upsampled['ch%i'%(2*j)][event_index])
                                y_odd = waveforms_upsampled['ch%i'%(2*j+1)][event_index]/numpy.std(waveforms_upsampled['ch%i'%(2*j+1)][event_index])

                            expected_time_difference = numpy.array(expected_time_differences)[[i in x[0] and j in x[0] for x in expected_time_differences]][0][1]

                            corr_even = scipy.signal.correlate(x_even,y_even)/(len(x_even)) #should be roughly normalized between -1,1
                            corr_odd = scipy.signal.correlate(x_odd,y_odd)/(len(x_odd)) #should be roughly normalized between -1,1
                            
                            time_shifts = numpy.arange(-len(x_even)+1,len(x_even))*dt
                            
                            time_shift_cut = numpy.logical_and(numpy.abs(time_shifts) < expected_time_difference + expected_timing_pm_tol,numpy.abs(time_shifts) > expected_time_difference - expected_timing_pm_tol)
                            #time_shift_cut = numpy.ones_like(time_shifts,dtype=bool)

                            max_corrs_even[event_index][i][j] = numpy.max(corr_even[time_shift_cut])
                            delays_even[event_index][i][j] = time_shifts[time_shift_cut][numpy.argmax(corr_even[time_shift_cut])]

                            max_corrs_odd[event_index][i][j] = numpy.max(corr_odd[time_shift_cut])
                            delays_odd[event_index][i][j] = time_shifts[time_shift_cut][numpy.argmax(corr_odd[time_shift_cut])]

                            '''
                            even_index = numpy.where((numpy.cumsum(corr_even**2)/sum(corr_even**2) > 0.50))[0][0]
                            odd_index = numpy.where((numpy.cumsum(corr_odd**2)/sum(corr_odd**2) > 0.50))[0][0]

                            max_corrs_even[event_index][i][j] = corr_even[even_index]
                            delays_even[event_index][i][j] = time_shifts[even_index]

                            max_corrs_odd[event_index][i][j] = corr_odd[odd_index]
                            delays_odd[event_index][i][j] = time_shifts[odd_index]
                            '''
                            if corr_plot:

                                plt.figure(fig_even.number)
                                plt.subplot(6,1,numpy.where([[i in x[0] and j in x[0] for x in expected_time_differences]][0])[0][0] + 1)
                                plt.plot(time_shifts,corr_even,alpha=0.8)

                                plt.figure(fig_odd.number)
                                plt.subplot(6,1,numpy.where([[i in x[0] and j in x[0] for x in expected_time_differences]][0])[0][0] + 1)
                                plt.plot(time_shifts,corr_odd,alpha=0.8)


            mean_corrs_even = numpy.mean(max_corrs_even,axis=0)
            mean_corrs_odd = numpy.mean(max_corrs_odd,axis=0)

            time_differences_even = []
            time_differences_odd = []
            for i in range(4):
                for j in range(4):
                    if j >= i:
                        continue
                    else:
                        expected_time_difference = numpy.array(expected_time_differences)[[i in x[0] and j in x[0] for x in expected_time_differences]][0][1]
                        max_time_difference = numpy.array(max_time_differences)[[i in x[0] and j in x[0] for x in max_time_differences]][0][1]

                        plt.figure()
                        plt.suptitle('Cross Correlation Times Between Antenna %i and %i'%(i,j))

                        ax = plt.subplot(2,1,1)
                        n, bins, patches = plt.hist(delays_even[:,i,j],label=('Channel %i and %i'%(2*i,2*j)),bins=bins)
                        best_delay_even = (bins[numpy.argmax(n)+1] + bins[numpy.argmax(n)])/2.0
                        time_differences_even.append(((i,j),best_delay_even))

                        plt.xlabel('Delay (ns)',fontsize=16)
                        plt.ylabel('Counts',fontsize=16)
                        plt.axvline(expected_time_difference,c='r',linestyle='--',label='Expected Time Difference = %f'%expected_time_difference)
                        plt.axvline(-expected_time_difference,c='r',linestyle='--')
                        plt.axvline(max_time_difference,c='g',linestyle='--',label='max Time Difference = %f'%max_time_difference)
                        plt.axvline(-max_time_difference,c='g',linestyle='--')

                        plt.axvline(best_delay_even,c='c',linestyle='--',label='Best Time Difference = %f'%best_delay_even)
                        plt.legend(fontsize=16)


                        plt.subplot(2,1,2,sharex=ax)
                        n, bins, patches = plt.hist(delays_odd[:,i,j],label=('Channel %i and %i'%(2*i+1,2*j+1)),bins=bins)
                        best_delay_odd = (bins[numpy.argmax(n)+1] + bins[numpy.argmax(n)])/2.0
                        time_differences_odd.append(((i,j),best_delay_odd))
                        plt.xlabel('Delay (ns)',fontsize=16)
                        plt.ylabel('Counts',fontsize=16)
                        plt.axvline(expected_time_difference,c='r',linestyle='--',label='Expected Time Difference = %f'%expected_time_difference)
                        plt.axvline(-expected_time_difference,c='r',linestyle='--')
                        plt.axvline(max_time_difference,c='g',linestyle='--',label='max Time Difference = %f'%max_time_difference)
                        plt.axvline(-max_time_difference,c='g',linestyle='--')
                        plt.axvline(best_delay_odd,c='c',linestyle='--',label='Best Time Difference = %f'%best_delay_odd)
                        plt.legend(fontsize=16)

            print(time_differences_even)
            print(time_differences_odd)
            '''
            for event_index in range(1):
                sys.stdout.write('\r(%i/%i)'%(event_index,len(eventids)))
                sys.stdout.flush()
                for i in range(4):
                    if use_raw:
                        x_even = waveforms_raw['ch%i'%(2*i)][event_index]/numpy.std(waveforms_raw['ch%i'%(2*i)][event_index])
                        x_odd = waveforms_raw['ch%i'%(2*i+1)][event_index]/numpy.std(waveforms_raw['ch%i'%(2*i+1)][event_index])
                    else:
                        x_even = waveforms_upsampled['ch%i'%(2*i)][event_index]/numpy.std(waveforms_upsampled['ch%i'%(2*i)][event_index])
                        x_odd = waveforms_upsampled['ch%i'%(2*i+1)][event_index]/numpy.std(waveforms_upsampled['ch%i'%(2*i+1)][event_index])
                    for j in range(4):
                        if j >= i:
                            continue
                        else:
                            delays_even[event_index][i][j] = int(delays_even[event_index][i][j]/dt)
                            delays_odd[event_index][i][j] = int(delays_odd[event_index][i][j]/dt)
            '''

        except Exception as e:
            print('Error in main loop.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)






