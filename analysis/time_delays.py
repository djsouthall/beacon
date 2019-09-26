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
from objects.fftmath import TimeDelayCalculator

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

if __name__ == '__main__':
    #plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = os.environ['BEACON_DATA']

    if len(sys.argv) == 2:
        if str(sys.argv[1]) in ['day5', 'day6']:
            site = str(sys.argv[1])
        else:
            print('Given site not in options.  Defaulting to day5')
            site = 'day5'
    else:
        print('No site given.  Defaulting to day5')
        site = 'day5'


    if site == 'day5':
        expected_time_differences_hpol  =  [((0, 1), -12.370633960780424), ((0, 2), 19.082452170088118), ((0, 3), 34.91110427374633), ((1, 2), 31.45308613086854), ((1, 3), 47.281738234526756), ((2, 3), 15.828652103658214)]
        expected_time_differences_vpol  =  [((0, 1), -14.20380849900198), ((0, 2), 19.858237233245745), ((0, 3), 33.29384923010252), ((1, 2), 34.062045732247725), ((1, 3), 47.4976577291045), ((2, 3), 13.435611996856778)]
        max_time_differences  =  [((0, 1), -10.781845180090386), ((0, 2), 17.31943377487096), ((0, 3), 38.26081567689971), ((1, 2), 28.101278954961344), ((1, 3), 49.042660856990096), ((2, 3), 20.941381902028752)]
        runs = numpy.array([782,783,784,785,788,789])
    elif site == 'day6':
        expected_time_differences_hpol  =  [((0, 1), -13.508319251205194), ((0, 2), 17.170337798788978), ((0, 3), 33.424724259292816), ((1, 2), 30.67865704999417), ((1, 3), 46.93304351049801), ((2, 3), 16.254386460503838)]
        expected_time_differences_vpol  =  [((0, 1), -13.579241831790796), ((0, 2), 19.154835384631042), ((0, 3), 30.67772905831862), ((1, 2), 32.73407721642184), ((1, 3), 44.256970890109415), ((2, 3), 11.522893673687577)]
        max_time_differences  =  [((0, 1), -9.892282610593384), ((0, 2), 17.22764581086949), ((0, 3), 38.683274875513234), ((1, 2), 27.119928421462873), ((1, 3), 48.57555748610662), ((2, 3), 21.455629064643745)]
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
                bins_pm_ns = 20.0
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
