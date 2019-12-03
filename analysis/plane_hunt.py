'''
This script uses the time delay calculators to look for signals bouncing off of planes.
The characteristic of this signals that we look for is their progression through the sky,
i.e. a repeated signal that moves gradually with time in a sensible flight path.
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
from tools.data_handler import createFile
from objects.fftmath import TimeDelayCalculator
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from pprint import pprint
import itertools
import warnings
import h5py
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

cable_delays = info.loadCableDelays()


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.

known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
ignorable_pulser_ids = info.loadIgnorableEventids()
cm = plt.cm.get_cmap('plasma')

def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))

def countSimilar(delays,similarity_atol=5,verbose=True):
    '''
    Given a set of delays this function will determine which ones are similar to eachother.
    Essentially each event will be given a similarity metric that describes how many other
    events are within a certain number of nanoseconds from it.  Because it considers all
    time delays per event, it should only gain similarity when things are from similar
    sources.

    Each row in delays should denote a single event with 6 columns representing the
    6 baseline time delays for a single polarization.

    Similarity_atol is applied in ns for each delay in the comparison.
    '''
    try:

        n_rows = delays.shape[0]
        

        similarity_count = numpy.zeros(n_rows) #zeros for each column (event)

        delays_rolled = delays.copy()

        for roll in numpy.arange(1,n_rows):
            if verbose == True:
                if roll%100 == 0:
                    sys.stdout.write('(%i/%i)\t\t\t\r'%(roll,n_rows))
                    sys.stdout.flush()
            delays_rolled = numpy.roll(delays_rolled,1,axis=0) #Comparing each event to the next event.
            comparison = numpy.isclose(delays,delays_rolled,atol=similarity_atol)
            if False:
                similarity_count += numpy.all(comparison,axis=1) #All time delays are within tolerance between the two events.
            else:
                similarity_count += numpy.sum(comparison,axis=1) >= 5 #5/6 time delays are within tolerance between the two events.

        return similarity_count
    except Exception as e:
        file.close()
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    plt.close('all')
    datapath = os.environ['BEACON_DATA']
    run = 1652

    runs = numpy.arange(1652,1670)
    runs = [1652]

    times = numpy.array([])
    delays = numpy.array([])

    for run_index, run in enumerate(runs):
        try:
            reader = Reader(datapath,run)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
            if filename is not None:
                with h5py.File(filename, 'r') as file:
                    try:
                        #print(list(file.keys()))
                        rough_dir_cut = file['hpol_max_corr_dir_ENU_zenith'][...] < 95.0 #Above horizen.  Don't trust calibration enough for this to be perfect.
                        
                        rf_cut = file['trigger_type'][...] == 2 #This is RF triggers.
                        inband_cut = ~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...],axis=1) < 49, file['trigger_type'][...] > 48) #Cutting out known CW
                        total_loading_cut = numpy.logical_and(numpy.logical_and(rf_cut,inband_cut),rough_dir_cut)

                        eventids = file['eventids'][total_loading_cut]
                        calibrated_trigtime = file['calibrated_trigtime'][total_loading_cut]

                        delays = numpy.vstack((file['hpol_t_%isubtract%i'%(0,1)][total_loading_cut],file['hpol_t_%isubtract%i'%(0,2)][total_loading_cut],file['hpol_t_%isubtract%i'%(0,3)][total_loading_cut],file['hpol_t_%isubtract%i'%(1,2)][total_loading_cut],file['hpol_t_%isubtract%i'%(1,3)][total_loading_cut],file['hpol_t_%isubtract%i'%(2,3)][total_loading_cut])).T
                        similarity_count = countSimilar(delays)

                        ignore_unreal = numpy.any(abs(delays) > 300,axis=1) #might be too strict for ones that don't have 4 visible pulses.
                        similarity_cut = similarity_count < 10 #Less than this number of similar events in a run to show up.

                        cut = numpy.logical_and(~ignore_unreal,similarity_cut) #Expand for more cuts

                        if run_index == 0:
                            times = calibrated_trigtime[cut]
                            total_delays = delays[cut,:]
                        else:
                            times =  numpy.vstack((times,calibrated_trigtime[cut]))
                            total_delays =  numpy.vstack((total_delays,delays[cut,:]))

                        if False:
                            remove_top_n_bins = 0
                            n_bins = 500
                            remove_indices = numpy.array([])


                            plt.figure()
                            bins = numpy.linspace(numpy.min(delays[:,0:3]),numpy.max(delays[:,0:3]),n_bins+1)
                            plt.subplot(3,1,1)
                            n, outbins, patches = plt.hist(delays[cut,0],bins=bins,label = 't0 - t1')
                            plt.xlabel('t0 - t1 (ns)')
                            plt.ylabel('Counts')
                            plt.yscale('log', nonposy='clip')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            top_left_edges = numpy.argsort(n)[::-1][0:remove_top_n_bins]
                            for left_bin_edge_index in top_left_edges:
                                if left_bin_edge_index == len(n):
                                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(delays[:,0] >= outbins[left_bin_edge_index],delays[:,0] <= outbins[left_bin_edge_index + 1]) )[0]) #Different bin boundary condition on right edge. 
                                else:
                                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(delays[:,0] >= outbins[left_bin_edge_index],delays[:,0] < outbins[left_bin_edge_index + 1]) )[0])



                            plt.subplot(3,1,2)
                            n, outbins, patches = plt.hist(delays[cut,1],bins=bins,label = 't0 - t2')
                            plt.xlabel('t0 - t2 (ns)')
                            plt.ylabel('Counts')
                            plt.yscale('log', nonposy='clip')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            top_left_edges = numpy.argsort(n)[::-1][0:remove_top_n_bins]
                            for left_bin_edge_index in top_left_edges:
                                if left_bin_edge_index == len(n):
                                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(delays[:,1] >= outbins[left_bin_edge_index],delays[:,1] <= outbins[left_bin_edge_index + 1]) )[0]) #Different bin boundary condition on right edge. 
                                else:
                                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(delays[:,1] >= outbins[left_bin_edge_index],delays[:,1] < outbins[left_bin_edge_index + 1]) )[0])


                            plt.subplot(3,1,3)
                            n, outbins, patches = plt.hist(delays[cut,2],bins=bins,label = 't0 - t3')
                            plt.xlabel('t0 - t3 (ns)')
                            plt.ylabel('Counts')
                            plt.yscale('log', nonposy='clip')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            top_left_edges = numpy.argsort(n)[::-1][0:remove_top_n_bins]
                            for left_bin_edge_index in top_left_edges:
                                if left_bin_edge_index == len(n):
                                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(delays[:,2] >= outbins[left_bin_edge_index],delays[:,2] <= outbins[left_bin_edge_index + 1]) )[0]) #Different bin boundary condition on right edge. 
                                else:
                                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(delays[:,2] >= outbins[left_bin_edge_index],delays[:,2] < outbins[left_bin_edge_index + 1]) )[0])

                            #remove_indices cuts from the time delays before [cut] is taken.  
                            remove_indices = numpy.sort(numpy.unique(remove_indices))
                            cut[remove_indices.astype(int)] = False

                        if False:
                            fig = plt.figure()
                            scatter = plt.scatter(delays[cut,0],delays[cut,1],c=(calibrated_trigtime[cut]-min(calibrated_trigtime))/60,cmap=cm)
                            cbar = fig.colorbar(scatter)
                            plt.xlabel('t0 - t1')
                            plt.ylabel('t0 - t2')

                            fig = plt.figure()
                            scatter = plt.scatter(delays[cut,0],delays[cut,2],c=(calibrated_trigtime[cut]-min(calibrated_trigtime))/60,cmap=cm)
                            cbar = fig.colorbar(scatter)
                            cbar.ax.set_ylabel('Time from Start of Run (min)', rotation=270)
                            plt.xlabel('t0 - t1')
                            plt.ylabel('t0 - t3')

                        if True:
                            
                            if False:
                                c = file['inband_peak_freq_MHz'][...][total_loading_cut,0][cut]
                                suptitle = 'Color Corresponds to Peak Frequency In Band (MHz)'
                                norm = None
                            elif False:
                                uniqueness = 1 - similarity_count/max(similarity_count)
                                c = uniqueness[cut]
                                suptitle = 'Uniqueness (1 = Unique, 0 = Common)'
                                norm = None
                            elif False:
                                c = similarity_count[cut]
                                suptitle = 'Number of Similar Events in Run'
                                if max(similarity_count[cut]) >= 100:
                                    norm = LogNorm()
                                else:
                                    norm = None
                            elif True:
                                c = file['hpol_max_corr_dir_ENU_azimuth'][...][total_loading_cut][cut]
                                suptitle = 'hpol_max_corr_dir_ENU_azimuth (deg)'
                                norm = None

                            fig = plt.figure()
                            ax = plt.subplot(3,1,1)
                            scatter1 = plt.scatter((calibrated_trigtime[cut]-min(calibrated_trigtime)), delays[cut,0],label = 't0 - t1',c=c,cmap=cm,norm=norm)
                            #cbar = ax.colorbar(scatter)
                            plt.xlabel('Calibrated trigtime (s)')
                            plt.ylabel('t0 - t1 (ns)')
                            plt.legend()
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                            
                            divider = make_axes_locatable(ax)
                            cax = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(scatter1, cax=cax, orientation='vertical')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)



                            ax2 = plt.subplot(3,1,2,sharex=ax)
                            scatter2 = plt.scatter((calibrated_trigtime[cut]-min(calibrated_trigtime)), delays[cut,1],label = 't0 - t2',c=c,cmap=cm,norm=norm)
                            #cbar = ax.colorbar(scatter)
                            plt.xlabel('Calibrated trigtime (s)')
                            plt.ylabel('t0 - t2 (ns)')
                            plt.minorticks_on()
                            plt.legend()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            divider = make_axes_locatable(ax2)
                            cax2 = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(scatter2, cax=cax2, orientation='vertical')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            ax3 = plt.subplot(3,1,3,sharex=ax)
                            scatter3 = plt.scatter((calibrated_trigtime[cut]-min(calibrated_trigtime)), delays[cut,2],label = 't0 - t3',c=c,cmap=cm,norm=norm)
                            #cbar = ax.colorbar(scatter)
                            plt.xlabel('Calibrated trigtime (s)')
                            plt.ylabel('t0 - t3 (ns)')
                            plt.minorticks_on()
                            plt.legend()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            divider = make_axes_locatable(ax3)
                            cax3 = divider.append_axes('right', size='5%', pad=0.05)
                            fig.colorbar(scatter3, cax=cax3, orientation='vertical')
                            plt.minorticks_on()
                            plt.grid(b=True, which='major', color='k', linestyle='-')
                            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                            plt.suptitle(suptitle)



                    except Exception as e:
                        file.close()
                        print('Error in plotting.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                file.close()
            else:
                print('filename is None, indicating empty tree.  Skipping run %i'%run)

        except Exception as e:
            print('Error in plotting.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
