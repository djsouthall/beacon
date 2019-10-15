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
from mpl_toolkits.mplot3d import Axes3D
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

    site = 1 #1,2,3

    if site == 1:
        runs = numpy.array([1507])

        expected_time_differences_physical  =  [((0, 1), -55.818082350306895), ((0, 2), 82.39553727998077), ((0, 3), 18.992683496782092), ((1, 2), 138.21361963028767), ((1, 3), 74.81076584708899), ((2, 3), -63.40285378319868)]
        max_time_differences_physical  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), 184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]
        expected_time_differences_hpol  =  [((0, 1), -55.818082350306895), ((0, 2), 82.39553727998077), ((0, 3), 18.992683496782092), ((1, 2), 138.21361963028767), ((1, 3), 74.81076584708899), ((2, 3), -63.40285378319868)]
        max_time_differences_hpol  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), 184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]
        expected_time_differences_vpol  =  [((0, 1), -55.818082350306895), ((0, 2), 82.39553727998077), ((0, 3), 18.992683496782092), ((1, 2), 138.21361963028767), ((1, 3), 74.81076584708899), ((2, 3), -63.40285378319868)]
        max_time_differences_vpol  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), 184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]

        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
        pulser_location = info.loadPulserLocationsENU()['run1507'] #ENU
    elif site == 2:
        runs = numpy.array([1509])
    
        expected_time_differences_physical  =  [((0, 1), -96.21228508039667), ((0, 2), 21.36317970746586), ((0, 3), -56.5419782996255), ((1, 2), 117.57546478786253), ((1, 3), 39.67030678077117), ((2, 3), -77.90515800709136)]
        max_time_differences_physical  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]
        expected_time_differences_hpol  =  [((0, 1), -96.21228508039667), ((0, 2), 21.36317970746586), ((0, 3), -56.5419782996255), ((1, 2), 117.57546478786253), ((1, 3), 39.67030678077117), ((2, 3), -77.90515800709136)]
        max_time_differences_hpol  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]
        expected_time_differences_vpol  =  [((0, 1), -96.21228508039667), ((0, 2), 21.36317970746586), ((0, 3), -56.5419782996255), ((1, 2), 117.57546478786253), ((1, 3), 39.67030678077117), ((2, 3), -77.90515800709136)]
        max_time_differences_vpol  =  [((0, 1), -129.37157110284315), ((0, 2), 152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), 139.82851662731397), ((1, 3), 104.08254793314117), ((2, 3), -81.41340851163496)]

        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
        pulser_location = info.loadPulserLocationsENU()['run1509'] #ENU

    elif site == 3:
        runs = numpy.array([1511])

        expected_time_differences_physical  =  [((0, 1), -103.1812449168724), ((0, 2), -142.99918760162836), ((0, 3), -183.12401361615616), ((1, 2), -39.81794268475596), ((1, 3), -79.94276869928376), ((2, 3), -40.1248260145278)]
        max_time_differences_physical  =  [((0, 1), -129.37157110284315), ((0, 2), -152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), -139.82851662731397), ((1, 3), -104.08254793314117), ((2, 3), -81.41340851163496)]
        expected_time_differences_hpol  =  [((0, 1), -103.1812449168724), ((0, 2), -142.99918760162836), ((0, 3), -183.12401361615616), ((1, 2), -39.81794268475596), ((1, 3), -79.94276869928376), ((2, 3), -40.1248260145278)]
        max_time_differences_hpol  =  [((0, 1), -129.37157110284315), ((0, 2), -152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), -139.82851662731397), ((1, 3), -104.08254793314117), ((2, 3), -81.41340851163496)]
        expected_time_differences_vpol  =  [((0, 1), -103.1812449168724), ((0, 2), -142.99918760162836), ((0, 3), -183.12401361615616), ((1, 2), -39.81794268475596), ((1, 3), -79.94276869928376), ((2, 3), -40.1248260145278)]
        max_time_differences_vpol  =  [((0, 1), -129.37157110284315), ((0, 2), -152.48216718324971), ((0, 3), -184.0463943150346), ((1, 2), -139.82851662731397), ((1, 3), -104.08254793314117), ((2, 3), -81.41340851163496)]

        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=1)
        pulser_location = info.loadPulserLocationsENU()['run1511'] #ENU

    if True:
        fig = plt.figure()
        fig.canvas.set_window_title('Antenna Locations')
        ax = fig.add_subplot(111, projection='3d')

        for i, a in antennas_physical.items():
            ax.scatter(a[0], a[1], a[2], marker='o',label=str(i))

        ax.scatter(pulser_location[0], pulser_location[1], pulser_location[2], marker='o',label='Pulser Site %i'%site)

        ax.set_xlabel('E (m)')
        ax.set_ylabel('N (m)')
        ax.set_zlabel('Relative Elevation (m)')
        plt.legend()

    for run in runs:
        try:
            pols.append(pol_info['run%i'%run])
        except:
            continue
    
    ignore_eventids = True
    #Filter settings
    final_corr_length = 2**18 #Should be a factor of 2 for fastest performance
    crit_freq_low_pass_MHz = None #This new pulser seems to peak in the region of 85 MHz or so
    crit_freq_high_pass_MHz = None
    low_pass_filter_order = None
    high_pass_filter_order = None

    align_method = 0 #0 = argmax, 1 = argmax of hilbert

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
                eventids = {}
                eventids['hpol'] = numpy.sort(known_pulser_ids['run%i'%run]['hpol'])
                eventids['vpol'] = numpy.sort(known_pulser_ids['run%i'%run]['vpol'])
                all_eventids = numpy.sort(numpy.append(eventids['hpol'],eventids['vpol']))

                if ignore_eventids == True:
                    if 'run%i'%run in list(ignorable_pulser_ids.keys()):
                        eventids['hpol'] = eventids['hpol'][~numpy.isin(eventids['hpol'],ignorable_pulser_ids['run%i'%run])]
                        eventids['vpol'] = eventids['vpol'][~numpy.isin(eventids['vpol'],ignorable_pulser_ids['run%i'%run])]
                        all_eventids = all_eventids[~numpy.isin(all_eventids,ignorable_pulser_ids['run%i'%run])]

                reader = Reader(datapath,run)
                reader.setEntry(all_eventids[0])
                tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order)
                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(all_eventids,align_method=align_method)

                #??? Am  I appropriately selecting the correct events?  This is such a weird way to do this.  Why am I not doing it more seperately :c

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

                bin_bounds = [numpy.min((numpy.mean(hpol_delays[pair_index]),numpy.mean(vpol_delays[pair_index]))) - bins_pm_ns, numpy.max((numpy.mean(hpol_delays[pair_index]),numpy.mean(vpol_delays[pair_index]))) + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_vpol)*50))
                time_bins = numpy.arange(bin_bounds[0],bin_bounds[1],tdc.dt_ns_upsampled)




                fig = plt.figure()

                #HPOL Plot
                pair = hpol_pairs[pair_index]

                i = pair[0]//2 #Antenna numbers
                j = pair[1]//2 #Antenna numbers

                fig.canvas.set_window_title('%i-%i Time Delays'%(i,j))

                expected_time_difference_hpol = numpy.array(expected_time_differences_hpol)[[i in x[0] and j in x[0] for x in expected_time_differences_hpol]][0][1]
                max_time_difference = numpy.array(max_time_differences_physical)[[i in x[0] and j in x[0] for x in max_time_differences_physical]][0][1]
                
                #bin_bounds = [expected_time_difference_hpol - bins_pm_ns,expected_time_difference_hpol + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_hpol)*50))
                bin_bounds = [numpy.mean(hpol_delays[pair_index]) - bins_pm_ns,numpy.mean(hpol_delays[pair_index]) + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_hpol)*50))
                time_bins = numpy.arange(bin_bounds[0],bin_bounds[1],tdc.dt_ns_upsampled)

                plt.suptitle('t(Ant%i) - t(Ant%i)'%(i,j))
                ax = plt.subplot(2,1,1)
                n, bins, patches = plt.hist(hpol_delays[pair_index],label=('Channel %i and %i'%(2*i,2*j)),bins=time_bins)

                x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0

                #best_delay_hpol = numpy.mean(hpol_delays[pair_index])
                best_delay_hpol = (bins[numpy.argmax(n)+1] + bins[numpy.argmax(n)])/2.0

                plt.xlabel('HPol Delay (ns)',fontsize=16)
                plt.ylabel('Counts',fontsize=16)
                plt.axvline(expected_time_difference_hpol,c='r',linestyle='--',label='Expected Time Difference = %f'%expected_time_difference_hpol)
                #plt.axvline(-expected_time_difference_hpol,c='r',linestyle='--')
                plt.axvline(max_time_difference,c='g',linestyle='--',label='max Time Difference = %f'%max_time_difference)
                #plt.axvline(-max_time_difference,c='g',linestyle='--')

                plt.axvline(best_delay_hpol,c='c',linestyle='--',label='Peak Bin Value = %f'%best_delay_hpol)

                try:
                    #Fit Gaussian
                    popt, pcov = curve_fit(gaus,x,n,p0=[100,best_delay_hpol,2.0])
                    popt[2] = abs(popt[2]) #I want positive sigma.

                    plot_x = numpy.linspace(min(x),max(x),1000)
                    plt.plot(plot_x,gaus(plot_x,*popt),'--',label='fit')

                    time_differences_hpol.append(((i,j),popt[1]))
                    time_differences_errors_hpol.append(((i,j),popt[2]))

                    plt.axvline(popt[1],c='y',linestyle='--',label='Fit Center = %f'%popt[1])
                except Exception as e:
                    print('Error trying to fit hpol data.')
                    print(e)
                    time_differences_hpol.append(((i,j),-999))
                    time_differences_errors_hpol.append(((i,j),-999))

                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

                plt.legend(fontsize=16)



                #VPOL Plot

                expected_time_difference_vpol = numpy.array(expected_time_differences_vpol)[[i in x[0] and j in x[0] for x in expected_time_differences_vpol]][0][1]
                max_time_difference = numpy.array(max_time_differences_physical)[[i in x[0] and j in x[0] for x in max_time_differences_physical]][0][1]
                
                #bin_bounds = [expected_time_difference_vpol - bins_pm_ns,expected_time_difference_vpol + bins_pm_ns ]#numpy.sort((0, numpy.sign(expected_time_difference_vpol)*50))

                plt.subplot(2,1,2)
                n, bins, patches = plt.hist(vpol_delays[pair_index],label=('Channel %i and %i'%(2*i+1,2*j+1)),bins=time_bins)

                x = (bins[1:len(bins)] + bins[0:len(bins)-1] )/2.0

                #best_delay_vpol = numpy.mean(vpol_delays[pair_index])
                best_delay_vpol = (bins[numpy.argmax(n)+1] + bins[numpy.argmax(n)])/2.0

                plt.xlabel('VPol Delay (ns)',fontsize=16)
                plt.ylabel('Counts',fontsize=16)
                plt.axvline(expected_time_difference_vpol,c='r',linestyle='--',label='Expected Time Difference = %f'%expected_time_difference_vpol)
                #plt.axvline(-expected_time_difference_vpol,c='r',linestyle='--')
                plt.axvline(max_time_difference,c='g',linestyle='--',label='max Time Difference = %f'%max_time_difference)
                #plt.axvline(-max_time_difference,c='g',linestyle='--')
                plt.axvline(best_delay_vpol,c='c',linestyle='--',label='Peak Bin Value = %f'%best_delay_vpol)

                try:
                    #Fit Gaussian
                    popt, pcov = curve_fit(gaus,x,n,p0=[100,best_delay_vpol,2.0])
                    popt[2] = abs(popt[2]) #I want positive sigma.

                    plot_x = numpy.linspace(min(x),max(x),1000)
                    plt.plot(plot_x,gaus(plot_x,*popt),'--',label='fit')

                    time_differences_vpol.append(((i,j),popt[1]))
                    time_differences_errors_vpol.append(((i,j),popt[2]))
                    plt.axvline(popt[1],c='y',linestyle='--',label='Fit Center = %f'%popt[1])

                except Exception as e:
                    print('Error trying to fit hpol data.')
                    print(e)
                    time_differences_vpol.append(((i,j),-999))
                    time_differences_errors_vpol.append(((i,j),-999))

                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

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
