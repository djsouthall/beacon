'''
This script uses the events found by find_good_saturating_signals.py to determine some good
expected time delays between antennas.  These can be used then as educated guesses for time
differences in the antenna_timings.py script. 
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
import tools.info as info

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.

known_pulser_ids = loadPulserEventids()
ignorable_pulser_ids = loadPulserIgnorableEventids()


def makeFilter(waveform_times,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order, plot_filter=False):
    dt = waveform_times[1] - waveform_times[0]
    freqs = numpy.fft.rfftfreq(len(waveform_times), d=(waveform_times[1] - waveform_times[0])/1.0e9)
    b, a = scipy.signal.butter(filter_order, crit_freq_low_pass_MHz*1e6, 'low', analog=True)
    d, c = scipy.signal.butter(filter_order, crit_freq_high_pass_MHz*1e6, 'high', analog=True)

    filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
    filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
    filter_x = freqs
    filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)
    if plot_filter == True:
        plt.figure()
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),color='k',label='final filter')
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass')
        plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass')
        plt.title('Butterworth filter frequency response')
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Amplitude [dB]')
        plt.margins(0, 0.1)
        plt.grid(which='both', axis='both')
        plt.axvline(crit_freq_low_pass_MHz, color='magenta',label='LP Crit') # cutoff frequency
        plt.axvline(crit_freq_high_pass_MHz, color='cyan',label='HP Crit') # cutoff frequency
        plt.xlim(0,200)
        plt.ylim(-50,10)
        plt.legend()
    return filter_y, freqs

#expected_time_differences = [((0, 1), 13.508319251205194), ((0, 2), 17.170337798788978), ((0, 3), 33.424724259292816), ((1, 2), 30.67865704999417), ((1, 3), 46.93304351049801), ((2, 3), 16.254386460503838)]
#max_time_differences = [((0, 1), 22.199031286793637), ((0, 2), 35.09576572192467), ((0, 3), 41.32087648347992), ((1, 2), 33.39469529610869), ((1, 3), 47.45680773722745), ((2, 3), 17.40978039384259)]

#expected_time_differences_hpol = [((0, 1), -13.508319251205194), ((0, 2), 17.170337798788978), ((0, 3), 33.424724259292816), ((1, 2), 30.67865704999417), ((1, 3), 46.93304351049801), ((2, 3), 16.254386460503838)] #Using physical location of antennas
#expected_time_differences_vpol = [((0, 1), -13.508319251205194), ((0, 2), 17.170337798788978), ((0, 3), 33.424724259292816), ((1, 2), 30.67865704999417), ((1, 3), 46.93304351049801), ((2, 3), 16.254386460503838)] #Using physical location of antennas
expected_time_differences_hpol = [((0, 1), -13.490847233240856), ((0, 2), 19.2102184307073), ((0, 3), 30.782767964694813), ((1, 2), 32.70106566394816), ((1, 3), 44.27361519793567), ((2, 3), 11.572549533987512)] #Using minimizing phase centers
expected_time_differences_vpol = [((0, 1), -9.889400300196257), ((0, 2), 17.233687802734266), ((0, 3), 38.68880234261974), ((1, 2), 27.123088102930524), ((1, 3), 48.578202642815995), ((2, 3), 21.45511453988547)] #Using minimizing phase centers

max_time_differences = [((0, 1), -22.199031286793637), ((0, 2), 35.09576572192467), ((0, 3), 41.32087648347992), ((1, 2), 33.39469529610869), ((1, 3), 47.45680773722745), ((2, 3), 17.40978039384259)]

def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))

if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    # If your data is elsewhere, pass it as an argument

    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    runs = numpy.array([793])#numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793]) #Selects which run to examine
    ignore_eventids = True
    #Filter settings
    final_corr_length = 2**17 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 75
    crit_freq_high_pass_MHz = 15
    filter_order = 6

    #Plotting info
    plot = True

    for run_index, run in enumerate(runs):
        if 'run%i'%run in list(known_pulser_ids.keys()):
            try:
                eventids = numpy.sort(known_pulser_ids['run%i'%run])
                if ignore_eventids == True:
                    if 'run%i'%run is in list(ignorable_pulser_ids.keys()):
                        eventids = eventids[~numpy.isin(eventids,ignorable_pulser_ids['run%i'%run])]

                reader = Reader(datapath,run)
                reader.setEntry(eventids[0])
                
                waveform_times = reader.t()
                dt = waveform_times[1]-waveform_times[0]
                waveform_times_padded_to_power2 = numpy.arange(2**(numpy.ceil(numpy.log2(len(waveform_times)))))*dt #Rounding up to a factor of 2 of the len of the waveforms
                waveform_times_corr = numpy.arange(2*len(waveform_times_padded_to_power2))*dt #multiplying by 2 for cross correlation later.
                
                filter_y,freqs = makeFilter(waveform_times_corr,crit_freq_low_pass_MHz, crit_freq_high_pass_MHz, filter_order,plot_filter=True)
                #filter_y = numpy.ones_like(filter_y)
                
                df = freqs[1] - freqs[0] #Note that this is the df for the padded correlation ffts and would not be the same as the one for the normal waveform ffts which have not been doubled in length. 
                final_dt = 1e9/(2*(final_corr_length//2 + 1)*df) #ns #This is the time step resulting from the cross correlation.  

                time_shifts = numpy.arange(-(final_corr_length-1)//2,(final_corr_length-1)//2 + 1)*final_dt #This results in the maxiumum of an autocorrelation being located at a time shift of 0.0


                hpol_pairs = list(itertools.combinations((0,2,4,6), 2))
                vpol_pairs = list(itertools.combinations((1,3,5,7), 2))
                #pairs = numpy.vstack((hpol_pairs,vpol_pairs)) 
                pairs = numpy.vstack((hpol_pairs,vpol_pairs,(0,0))) #0,0 is test case for autocorrelation 

                indices = numpy.zeros((len(pairs),len(eventids)))
                max_corrs = numpy.zeros((len(pairs),len(eventids)))

                for event_index, eventid in enumerate(eventids):
                    sys.stdout.write('(%i/%i)\r'%(event_index,len(eventids)))
                    sys.stdout.flush()
                    reader.setEntry(eventid)

                    raw_wfs = numpy.zeros((8,len(waveform_times_corr)))
                    for channel in range(8):
                        raw_wfs[channel][0:reader.header().buffer_length] = reader.wf(channel)
                    
                    ffts = numpy.fft.rfft(raw_wfs,axis=1) #Now upsampled
                    ffts = numpy.multiply(ffts,filter_y) #Now filtered

                    corrs_fft = numpy.multiply((ffts[pairs[:,0]].T/numpy.std(ffts[pairs[:,0]],axis=1)).T,(numpy.conj(ffts[pairs[:,1]]).T/numpy.std(numpy.conj(ffts[pairs[:,1]]),axis=1)).T) / (len(waveform_times_corr)//2 + 1)
                    corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=final_corr_length),axes=1) * (final_corr_length//2) #Upsampling and keeping scale

                    max_indices = numpy.array(list(zip(numpy.arange(len(pairs)),numpy.argmax(corrs,axis=1))))
                    super_indices = numpy.array(list(zip(numpy.arange(len(pairs),dtype=int),event_index*numpy.ones(len(pairs),dtype=int))))
                    indices[super_indices[:,0], super_indices[:,1]] = numpy.argmax(corrs,axis=1)

                    max_corrs[super_indices[:,0], super_indices[:,1]] = numpy.max(corrs,axis=1)

                
                #PLOTTING
                if plot:
                    time_differences_hpol = []
                    time_differences_errors_hpol = []

                    time_differences_vpol = []
                    time_differences_errors_vpol = []


                    hpol_delays = time_shifts[indices[0:6].astype(int)]
                    hpol_corrs = max_corrs[0:6]

                    vpol_delays = time_shifts[indices[6:12].astype(int)]
                    vpol_corrs = max_corrs[6:12]
                    
                    #time_bins = numpy.arange(-(final_corr_length-1)//2 - 1,(final_corr_length-1)//2 + 1)*final_dt + final_dt/2 

                    for pair_index in range(6):
                        bins_pm_ns = 2.5
                        fig = plt.figure()

                        #HPOL Plot
                        pair = hpol_pairs[pair_index]

                        i = pair[0]//2 #Antenna numbers
                        j = pair[1]//2 #Antenna numbers


                        expected_time_difference_hpol = numpy.array(expected_time_differences_hpol)[[i in x[0] and j in x[0] for x in expected_time_differences_hpol]][0][1]
                        max_time_difference = numpy.array(max_time_differences)[[i in x[0] and j in x[0] for x in max_time_differences]][0][1]
                        
                        bin_bounds = [expected_time_difference_hpol - bins_pm_ns,expected_time_difference_hpol + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_hpol)*50))
                        time_bins = numpy.arange(bin_bounds[0],bin_bounds[1],final_dt)

                        plt.suptitle('t(Ant%i) - t(Ant%i)'%(i,j))
                        ax = plt.subplot(2,1,1)
                        n, bins, patches = plt.hist(hpol_delays[pair_index],label=('Channel %i and %i'%(2*i,2*j)),bins=time_bins)

                        x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0

                        #best_delay_hpol = numpy.mean(hpol_delays[pair_index])
                        best_delay_hpol = (bins[numpy.argmax(n)+1] + bins[numpy.argmax(n)])/2.0

                        #Fit Gaussian
                        popt, pcov = curve_fit(gaus,x,n,p0=[100,best_delay_hpol,2.0])
                        popt[2] = abs(popt[2]) #I want positive sigma.

                        plot_x = numpy.linspace(min(x),max(x),1000)
                        plt.plot(plot_x,gaus(plot_x,*popt),'--',label='fit')

                        time_differences_hpol.append(((i,j),popt[1]))
                        time_differences_errors_hpol.append(((i,j),popt[2]))

                        plt.xlabel('HPol Delay (ns)',fontsize=16)
                        plt.ylabel('Counts',fontsize=16)
                        plt.axvline(expected_time_difference_hpol,c='r',linestyle='--',label='Expected Time Difference = %f'%expected_time_difference_hpol)
                        #plt.axvline(-expected_time_difference_hpol,c='r',linestyle='--')
                        #plt.axvline(max_time_difference,c='g',linestyle='--',label='max Time Difference = %f'%max_time_difference)
                        #plt.axvline(-max_time_difference,c='g',linestyle='--')

                        plt.axvline(best_delay_hpol,c='c',linestyle='--',label='Peak Bin Value = %f'%best_delay_hpol)
                        plt.axvline(popt[1],c='y',linestyle='--',label='Fit Center = %f'%popt[1])
                        plt.legend(fontsize=16)



                        #VPOL Plot

                        expected_time_difference_vpol = numpy.array(expected_time_differences_vpol)[[i in x[0] and j in x[0] for x in expected_time_differences_vpol]][0][1]
                        max_time_difference = numpy.array(max_time_differences)[[i in x[0] and j in x[0] for x in max_time_differences]][0][1]
                        
                        bin_bounds = [expected_time_difference_vpol - bins_pm_ns,expected_time_difference_vpol + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_vpol)*50))
                        time_bins = numpy.arange(bin_bounds[0],bin_bounds[1],final_dt)

                        plt.subplot(2,1,2)
                        n, bins, patches = plt.hist(vpol_delays[pair_index],label=('Channel %i and %i'%(2*i+1,2*j+1)),bins=time_bins)

                        x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0

                        #best_delay_vpol = numpy.mean(vpol_delays[pair_index])
                        best_delay_vpol = (bins[numpy.argmax(n)+1] + bins[numpy.argmax(n)])/2.0

                        #Fit Gaussian
                        popt, pcov = curve_fit(gaus,x,n,p0=[100,best_delay_vpol,2.0])
                        popt[2] = abs(popt[2]) #I want positive sigma.

                        plot_x = numpy.linspace(min(x),max(x),1000)
                        plt.plot(plot_x,gaus(plot_x,*popt),'--',label='fit')

                        time_differences_vpol.append(((i,j),popt[1]))
                        time_differences_errors_vpol.append(((i,j),popt[2]))

                        plt.xlabel('VPol Delay (ns)',fontsize=16)
                        plt.ylabel('Counts',fontsize=16)
                        plt.axvline(expected_time_difference_vpol,c='r',linestyle='--',label='Expected Time Difference = %f'%expected_time_difference_vpol)
                        #plt.axvline(-expected_time_difference_vpol,c='r',linestyle='--')
                        #plt.axvline(max_time_difference,c='g',linestyle='--',label='max Time Difference = %f'%max_time_difference)
                        #plt.axvline(-max_time_difference,c='g',linestyle='--')
                        plt.axvline(best_delay_vpol,c='c',linestyle='--',label='Peak Bin Value = %f'%best_delay_vpol)
                        plt.axvline(popt[1],c='y',linestyle='--',label='Fit Center = %f'%popt[1])
                        plt.legend(fontsize=16)



                    print('time_differences_hpol')
                    print(time_differences_hpol)
                    print('time_differences_errors_hpol')
                    print(time_differences_errors_hpol)

                    print('time_differences_vpol')
                    print(time_differences_vpol)
                    print('time_differences_errors_vpol')
                    print(time_differences_errors_vpol)


            except Exception as e:
                print('Error in main loop.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


