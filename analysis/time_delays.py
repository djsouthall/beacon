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

known_pulser_ids = info.loadPulserEventids()
ignorable_pulser_ids = info.loadIgnorableEventids()

def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))


class TimeDelayCalculator:
    '''
    Takes a run reader and does the math required for determining time delays between the antennas.

    Parameters
    ----------
    reader : examples.beacon_data_reader.reader
        The run reader you wish to examine time delays for.
    final_corr_length : int
        Should be given as a power of 2.  This is the goal length of the cross correlations, and can set the time resolution
        of the time delays.
    crit_freq_low_pass_MHz : float
        Sets the critical frequency of the low pass filter to be applied to the data.
    crit_freq_high_pass_MHz
        Sets the critical frequency of the high pass filter to be applied to the data.
    low_pass_filter_order
        Sets the order of the low pass filter to be applied to the data.
    high_pass_filter_order
        Sets the order of the high pass filter to be applied to the data.
    
    See Also
    --------
    examples.beacon_data_reader.reader
    '''
    def __init__(self, reader, final_corr_length=2**17, crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None):
        try:
            self.reader = reader
            self.buffer_length = reader.header().buffer_length
            self.final_corr_length = final_corr_length #Should be a factor of 2 for fastest performance
            self.crit_freq_low_pass_MHz = crit_freq_low_pass_MHz
            self.crit_freq_high_pass_MHz = crit_freq_high_pass_MHz
            self.low_pass_filter_order = low_pass_filter_order
            self.high_pass_filter_order = high_pass_filter_order

            self.hpol_pairs = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
            self.vpol_pairs = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
            self.pairs = numpy.vstack((self.hpol_pairs,self.vpol_pairs)) 

            self.prepForFFTs()

        except Exception as e:
            print('Error in TimeDelayCalculator.__init__()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

           
    def prepForFFTs(self):
        '''
        This will get timing information from the reader and use it to determine values such as timestep
        that will be used when performing ffts later.  
        '''
        try:
            self.eventid = 0
            self.reader.setEntry(self.eventid)

            #Below are for the original times of the waveforms and correspond frequencies.
            self.waveform_times_original = self.reader.t()
            self.dt_ns_original = self.waveform_times_original[1]-self.waveform_times_original[0] #ns
            self.freqs_original = numpy.fft.rfftfreq(len(self.waveform_times_original), d=self.dt_ns_original/1.0e9)
            self.df_original = self.freqs_original[1] - self.freqs_original[0]

            #Below are for waveforms padded to a power of 2 to make ffts faster.  This has the same timestep as original.
            self.waveform_times_padded_to_power2 = numpy.arange(2**(numpy.ceil(numpy.log2(len(self.waveform_times_original)))))*self.dt_ns_original #Rounding up to a factor of 2 of the len of the waveforms
            self.freqs_padded_to_power2 = numpy.fft.rfftfreq(len(self.waveform_times_padded_to_power2), d=self.dt_ns_original/1.0e9)
            self.df_padded_to_power2 = self.freqs_padded_to_power2[1] - self.freqs_padded_to_power2[0]

            #Below are for waveforms that were padded to a power of 2, then multiplied by two to add zero padding for cross correlation.
            self.waveform_times_corr = numpy.arange(2*len(self.waveform_times_padded_to_power2))*self.dt_ns_original #multiplying by 2 for cross correlation later.
            self.freqs_corr = numpy.fft.rfftfreq(len(self.waveform_times_corr), d=self.dt_ns_original/1.0e9)
            self.df_corr = self.freqs_corr[1] - self.freqs_corr[0]
            self.dt_ns_corr = 1.0e9/(2*(self.final_corr_length//2 + 1)*self.df_corr)
            self.corr_time_shifts = numpy.arange(-(self.final_corr_length-1)//2,(self.final_corr_length-1)//2 + 1)*self.dt_ns_corr #This results in the maxiumum of an autocorrelation being located at a time shift of 0.0

            #Prepare Filters
            self.filter_original = self.makeFilter(self.freqs_original,plot_filter=False)
            self.filter_padded_to_power2 = self.makeFilter(self.freqs_padded_to_power2,plot_filter=False)
            self.filter_corr = self.makeFilter(self.freqs_corr,plot_filter=False)

        except Exception as e:
            print('Error in Antenna.setPhaseCenter()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def makeFilter(self,freqs, plot_filter=False):
        '''
        This will make a frequency domain filter based on the given specifications. 
        '''
        try:
            filter_x = freqs
            if numpy.logical_and(self.low_pass_filter_order is not None, self.crit_freq_low_pass_MHz is not None):
                b, a = scipy.signal.butter(self.low_pass_filter_order, self.crit_freq_low_pass_MHz*1e6, 'low', analog=True)
                filter_x_low_pass, filter_y_low_pass = scipy.signal.freqs(b, a,worN=freqs)
            else:
                filter_x_low_pass = filter_x
                filter_y_low_pass = numpy.ones_like(filter_x)

            if numpy.logical_and(self.high_pass_filter_order is not None, self.crit_freq_high_pass_MHz is not None):
                d, c = scipy.signal.butter(self.high_pass_filter_order, self.crit_freq_high_pass_MHz*1e6, 'high', analog=True)
                filter_x_high_pass, filter_y_high_pass = scipy.signal.freqs(d, c,worN=freqs)
            else:
                filter_x_high_pass, = filter_x
                ilter_y_high_pass = numpy.ones_like(filter_x)

            filter_y = numpy.multiply(filter_y_low_pass,filter_y_high_pass)

            if plot_filter == True:
                fig = plt.figure()
                fig.canvas.set_window_title('Filter')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y)),color='k',label='final filter')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_low_pass)),color='r',linestyle='--',label='low pass')
                plt.plot(filter_x/1e6, 20 * numpy.log10(abs(filter_y_high_pass)),color='orange',linestyle='--',label='high pass')
                plt.title('Butterworth filter frequency response')
                plt.xlabel('Frequency [MHz]')
                plt.ylabel('Amplitude [dB]')
                plt.margins(0, 0.1)
                plt.grid(which='both', axis='both')
                if self.crit_freq_low_pass_MHz is not None:
                    plt.axvline(self.crit_freq_low_pass_MHz, color='magenta',label='LP Crit') # cutoff frequency
                if self.crit_freq_high_pass_MHz is not None:
                    plt.axvline(self.crit_freq_high_pass_MHz, color='cyan',label='HP Crit') # cutoff frequency
                plt.xlim(0,200)
                plt.ylim(-50,10)
                plt.legend()

            return filter_y
        except Exception as e:
            print('Error in TimeDelayCalculator.makeFilter')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateTimeDelays(self,eventid):
        try:
            self.reader.setEntry(eventid)

            raw_wfs = numpy.zeros((8,len(self.waveform_times_corr)))
            for channel in range(8):
                raw_wfs[channel][0:self.buffer_length] = self.reader.wf(channel)
            
            ffts = numpy.fft.rfft(raw_wfs,axis=1) #Now upsampled
            ffts = numpy.multiply(ffts,self.filter_corr) #Now filtered

            corrs_fft = numpy.multiply((ffts[self.pairs[:,0]].T/numpy.std(ffts[self.pairs[:,0]],axis=1)).T,(numpy.conj(ffts[self.pairs[:,1]]).T/numpy.std(numpy.conj(ffts[self.pairs[:,1]]),axis=1)).T) / (len(self.waveform_times_corr)//2 + 1)
            corrs = numpy.fft.fftshift(numpy.fft.irfft(corrs_fft,axis=1,n=final_corr_length),axes=1) * (final_corr_length//2) #Upsampling and keeping scale

            indices = numpy.argmax(corrs,axis=1)
            max_corrs = numpy.max(corrs,axis=1)

            return self.corr_time_shifts[indices], max_corrs, self.pairs
        except Exception as e:
            print('Error in TimeDelayCalculator.calculateTimeDelays')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateMultipleTimeDelays(self,eventids):
        try:
            if ~numpy.all(eventids == numpy.sort(eventids)):
                print('eventids NOT SORTED, WOULD BE FASTER IF SORTED.')
            timeshifts = []
            corrs = []
            print('Calculating time delays:')
            for event_index, eventid in enumerate(eventids):
                sys.stdout.write('(%i/%i)\t\t\t\r'%(event_index+1,len(eventids)))
                sys.stdout.flush()
                time_shift, corr_value, pairs = self.calculateTimeDelays(eventid)
                timeshifts.append(time_shift)
                corrs.append(corr_value)
            sys.stdout.write('\n')
            sys.stdout.flush()
            return numpy.array(timeshifts).T, numpy.array(corrs).T, self.pairs

        except Exception as e:
            print('Error in TimeDelayCalculator.calculateMultipleTimeDelays')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)




if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = os.environ['BEACON_DATA']

    if len(sys.argv) == 2:
        if str(sys.argv[1]) in ['vpol', 'hpol']:
            site = str(sys.argv[1])
        else:
            print('Given site not in options.  Defaulting to vpol')
            site = 'day5'

    if site == 'day5':
        expected_time_differences_hpol = [((0, 1), -12.370633960780424), ((0, 2), 19.082452170088118), ((0, 3), 34.91110427374633), ((1, 2), 31.45308613086854), ((1, 3), 47.281738234526756), ((2, 3), 15.828652103658214)]
        expected_time_differences_vpol = [((0, 1), -12.370633960780424), ((0, 2), 19.082452170088118), ((0, 3), 34.91110427374633), ((1, 2), 31.45308613086854), ((1, 3), 47.281738234526756), ((2, 3), 15.828652103658214)]
        max_time_differences = [((0, 1), -12.370633960780424), ((0, 2), 19.082452170088118), ((0, 3), 34.91110427374633), ((1, 2), 31.45308613086854), ((1, 3), 47.281738234526756), ((2, 3), 15.828652103658214)]
        runs = numpy.array([782,783,784,785,788,789])
    elif site == 'day6':
        expected_time_differences_hpol = [((0, 1), -13.490847233240856), ((0, 2), 19.2102184307073), ((0, 3), 30.782767964694813), ((1, 2), 32.70106566394816), ((1, 3), 44.27361519793567), ((2, 3), 11.572549533987512)] #Using minimizing phase centers
        expected_time_differences_vpol = [((0, 1), -9.889400300196257), ((0, 2), 17.233687802734266), ((0, 3), 38.68880234261974), ((1, 2), 27.123088102930524), ((1, 3), 48.578202642815995), ((2, 3), 21.45511453988547)] #Using minimizing phase centers
        max_time_differences = [((0, 1), -22.199031286793637), ((0, 2), 35.09576572192467), ((0, 3), 41.32087648347992), ((1, 2), 33.39469529610869), ((1, 3), 47.45680773722745), ((2, 3), 17.40978039384259)]
        runs = numpy.array([792,793])# SHOULD ALL BE FOR THE SAME SOURCE AS THEY WILL BE ADDED IN HIST
    pols = []
    pol_info = info.loadPulserPolarizations()
    for run in runs:
        try:
            pols.append(pol_info['run%i'%run])
        except:
            continue
    
    ignore_eventids = True
    #Filter settings
    final_corr_length = 2**17 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = 75
    crit_freq_high_pass_MHz = 15
    low_pass_filter_order = 6
    high_pass_filter_order = 6

    #Plotting info
    plot = True

    all_hpol_delays = {} #This will store the time delay data across all runs
    all_vpol_delays = {} #This will store the time delay data across all runs

    all_hpol_corrs = {} #This will store the time delay data across all runs
    all_vpol_corrs = {} #This will store the time delay data across all runs

    hpol_pairs  = numpy.array(list(itertools.combinations((0,2,4,6), 2)))
    vpol_pairs  = numpy.array(list(itertools.combinations((1,3,5,7), 2)))
    pairs       = numpy.vstack((hpol_pairs,vpol_pairs)) 

    for pair in pairs:
        if pair in hpol_pairs:
            all_hpol_delays[str(pair)] = numpy.array([])
            all_hpol_corrs[str(pair)] = numpy.array([])
        elif pair in vpol_pairs:
            all_vpol_delays[str(pair)] = numpy.array([])
            all_vpol_corrs[str(pair)] = numpy.array([])


    for run_index, run in enumerate(runs):
        if 'run%i'%run in list(known_pulser_ids.keys()):
            try:
                print('run%i\n'%run)
                eventids = numpy.sort(known_pulser_ids['run%i'%run])
                if ignore_eventids == True:
                    if 'run%i'%run in list(ignorable_pulser_ids.keys()):
                        eventids = eventids[~numpy.isin(eventids,ignorable_pulser_ids['run%i'%run])]

                reader = Reader(datapath,run)
                reader.setEntry(eventids[0])
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order)
                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(eventids)

                for pair_index, pair in enumerate(pairs):
                    if pair in hpol_pairs:
                        all_hpol_delays[str(pair)] = numpy.append(all_hpol_delays[str(pair)],time_shifts[pair_index])
                        all_hpol_corrs[str(pair)] = numpy.append(all_hpol_delays[str(pair)],corrs[pair_index])
                    elif pair in vpol_pairs:
                        all_vpol_delays[str(pair)] = numpy.append(all_vpol_delays[str(pair)],time_shifts[pair_index])
                        all_vpol_corrs[str(pair)] = numpy.append(all_vpol_delays[str(pair)],corrs[pair_index])


            except Exception as e:
                print('Error in main loop.')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


    if plot:
        try:
            #PLOTTING
            time_differences_hpol = []
            time_differences_errors_hpol = []

            time_differences_vpol = []
            time_differences_errors_vpol = []

            hpol_delays = numpy.array([value for key,value in all_hpol_delays.items()])
            hpol_corrs = numpy.array([value for key,value in all_hpol_corrs.items()])

            vpol_delays = numpy.array([value for key,value in all_vpol_delays.items()])
            vpol_corrs = numpy.array([value for key,value in all_vpol_corrs.items()])            
            
            for pair_index in range(6):
                bins_pm_ns = 10.0
                fig = plt.figure()

                #HPOL Plot
                pair = hpol_pairs[pair_index]

                i = pair[0]//2 #Antenna numbers
                j = pair[1]//2 #Antenna numbers

                fig.canvas.set_window_title('%i-%i Time Delays'%(i,j))

                expected_time_difference_hpol = numpy.array(expected_time_differences_hpol)[[i in x[0] and j in x[0] for x in expected_time_differences_hpol]][0][1]
                max_time_difference = numpy.array(max_time_differences)[[i in x[0] and j in x[0] for x in max_time_differences]][0][1]
                
                bin_bounds = [expected_time_difference_hpol - bins_pm_ns,expected_time_difference_hpol + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_hpol)*50))
                time_bins = numpy.arange(bin_bounds[0],bin_bounds[1],tdc.dt_ns_corr)

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
                
                bin_bounds = [expected_time_difference_vpol - bins_pm_ns, expected_time_difference_vpol + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_vpol)*50))
                time_bins = numpy.arange(bin_bounds[0],bin_bounds[1],tdc.dt_ns_corr)

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
            print('Error in plotting.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
