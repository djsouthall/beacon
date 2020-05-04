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
import glob

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret #Must be imported before matplotlib or else plots don't load.
import tools.clock_correct as cc
from objects.fftmath import TemplateCompareTool
import tools.info as info
from tools.data_handler import createFile
from objects.fftmath import TimeDelayCalculator
import tools.get_plane_tracks as pt
from tools.correlator import Correlator


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from matplotlib.widgets import Button

from pprint import pprint
import itertools
import warnings
import h5py
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

datapath = os.environ['BEACON_DATA']

cable_delays = info.loadCableDelays()


#The below are not ALL pulser points, but a set that has been precalculated and can be used
#if you wish to skip the calculation finding them.

known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
ignorable_pulser_ids = info.loadIgnorableEventids()
cm = plt.cm.get_cmap('plasma')

def txtToClipboard(txt):
    df=pd.DataFrame([txt])
    df.to_clipboard(index=False,header=False)

def gaus(x,a,x0,sigma):
    return a*numpy.exp(-(x-x0)**2.0/(2.0*sigma**2.0))

def countSimilar(delays,similarity_atol=2,verbose=True):
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


class SelectFromCollection(object):
    """
    This is an adapted bit of code that prints out information
    about the lasso'd data points. 

    Parameters
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        Axes to interact with.

    collection : :class:`matplotlib.collections.Collection` subclass
        Collection you want to select from.
    """

    def __init__(self, ax, collection, eventids, runnum, total_hpol_delays,total_vpol_delays,alpha_other=0.3):

        self.canvas = ax.figure.canvas
        self.eventids = eventids
        self.runnum = runnum
        self.id = numpy.array(list(zip(self.runnum,self.eventids)))
        self.collection = collection
        self.alpha_other = alpha_other

        self.xys = collection.get_offsets()
        self.Npts = len(self.xys)

        self.lasso = LassoSelector(ax, onselect=self.onselect)
        self.ind = []

        self.total_hpol_delays = total_hpol_delays
        self.total_vpol_delays = total_vpol_delays




    def onselect(self, verts):
        path = Path(verts)
        self.ind = numpy.nonzero(path.contains_points(self.xys))[0]
        print('Selected run/eventids:')
        print(repr(self.id[self.ind]))
        txtToClipboard(repr(self.id[self.ind]))
        event_info = self.id[self.ind]
        #print(event_info)
        print('Coordinates:')
        print(repr(self.xys[self.ind]))
        self.canvas.draw_idle()

        self.alignSelectedEvents()

    def plotTimeDelays(self,times,total_hpol_delays,total_vpol_delays):
        '''
        This is intended to reproduce the plot that Kaeli used to make, with the time
        delays being plotted all on the same plot. 
        '''
        times = times - times[0]
        for pol in ['Hpol','Vpol']:
            _fig = plt.figure()
            _fig.canvas.set_window_title('%s Delays'%pol)
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            plt.title(pol)
            plt.ylabel('Time Delay (ns)')
            plt.xlabel('Readout Time (s)')

            for pair_index, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                if pol == 'Hpol':
                    y = total_hpol_delays[:,pair_index]
                else:
                    y = total_vpol_delays[:,pair_index]

                #plt.plot(times, y,linestyle = '--',alpha=0.5)
                plt.scatter(times, y,label='A%i and A%i'%(pair[0],pair[1]))
            plt.legend(loc='upper right')

    def alignSelectedEvents(self, plot_aligned_wf=False,save_template=False,plot_timedelays=True):
        '''
        My plan is for this to be called when some events are circled in the plot.
        It will take those wf, align them, and plot the averaged waveforms.  No
        filters will be applied. 
        '''
        if plot_timedelays == True:
            runs, counts = numpy.unique(self.id[self.ind][:,0],return_counts=True)
            run = runs[numpy.argmax(counts)]
            print('Only calculating template from run with most points circled: run %i with %i events circled'%(run,max(counts)))
            eventids = self.id[self.ind][:,1][self.id[self.ind][:,0] == run]
            coords = self.xys[self.ind]

            self.plotTimeDelays(self.xys[self.ind][:,0]*60,self.total_hpol_delays[self.ind],self.total_vpol_delays[self.ind])

        _reader = Reader(datapath,run)
        
        crit_freq_low_pass_MHz = None
        low_pass_filter_order = None
        
        crit_freq_high_pass_MHz = None# 45
        high_pass_filter_order = None# 12
        
        waveform_index_range = (None,None)
        
        final_corr_length = 2**18

        tct = TemplateCompareTool(_reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=True)
        tdc = TimeDelayCalculator(_reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=True)
        self.cor = Correlator(_reader,  upsample=2**15, n_phi=360, n_theta=360, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=True)
        
        if True:
            print('TRYING TO MAKE CORRELATOR PLOT.')
            print(eventids)
            self.cor.animatedMap(eventids, 'both', '', plane_zenith=None,plane_az=None,hilbert=False, max_method=None,center_dir='E',save=False,dpi=300)

        times, averaged_waveforms = tct.averageAlignedSignalsPerChannel( eventids, align_method=0, template_eventid=eventids[0], plot=plot_aligned_wf,event_type=None)
        
        resampled_averaged_waveforms_original_length = numpy.zeros((8,len(_reader.t())))
        for channel in range(8):
            resampled_averaged_waveforms_original_length[channel] = scipy.interpolate.interp1d(times,averaged_waveforms[channel],kind='cubic',bounds_error=False,fill_value=0)(reader.t())

        if False:
            for channel in range(8):
                plt.figure()
                plt.title(str(channel))
                for eventid in eventids:
                    tct.setEntry(eventid)
                    plt.plot(tct.t(),tct.wf(channel),label=str(eventid),alpha=0.8)
                plt.legend()
                plt.xlabel('t (ns)')
                plt.ylabel('adu')


        if save_template == True:
            filename_index = 0 
            filename = './generated_event_template_%i.csv'%filename_index
            existing_files = numpy.array(glob.glob('./*.csv'))

            while numpy.isin(filename,existing_files):
                filename_index += 1
                filename = './generated_event_template_%i.csv'%filename_index
            numpy.savetxt(filename,resampled_averaged_waveforms_original_length, delimiter=",")
            print('Genreated template saved as:\n%s'%filename)



        tdc.calculateMultipleTimeDelays(eventids, align_method=8,hilbert=False,plot=True, colors=numpy.array(coords)[:,0])

        return resampled_averaged_waveforms_original_length

    def disconnect(self):
        self.lasso.disconnect_events()
        self.canvas.draw_idle()


class Skipper(object):
    '''
    This uses the code from https://matplotlib.org/3.1.1/gallery/widgets/buttons.html.
    The goal is to give buttons to skip back and forth in time to avoid using the drag
    function which is very slow over xlaunch applications.  I will likely do skip forward
    be +60 s, skip back is - 15 s.  Window width of 60s.  
    '''
    def __init__(self, ax, next_skip_s=60, prev_skip_s=15):
        self.ax = ax
        self.next_skip_s = next_skip_s
        self.prev_skip_s = prev_skip_s
        self.axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(self.axnext, 'Next')
        self.bnext.on_clicked(self.next)
        self.bprev = Button(self.axprev, 'Previous')
        self.bprev.on_clicked(self.prev)

    def next(self,event):
        xlim = self.ax.get_xlim()
        self.ax.set_xlim(xlim[0] + self.next_skip_s, xlim[1] + self.next_skip_s )
        plt.draw()

    def prev(self,event):
        xlim = self.ax.get_xlim()
        self.ax.set_xlim(xlim[0] - self.prev_skip_s, xlim[1] - self.prev_skip_s )
        plt.draw()



if __name__ == '__main__':
    plt.close('all')
    if len(sys.argv) > 1:
        run = int(sys.argv[1])
        
    else:
        run = 1701
    #runs = numpy.arange(1700,1701)#numpy.array([1728])#numpy.arange(1770,1800) #No RF triggers before 1642, but 1642,43,44,45 don't have pointing?
    runs = [run]
    colormap_mode = 7
    add_plane_tracks = True
    #0 : Color Corresponds to Peak Frequency In Band (MHz)
    #1 : Uniqueness (1 = Unique, 0 = Common)
    #2 : Number of Similar Events in Run
    #3 : hpol_max_corr_dir_ENU_zenith (deg)
    #4 : Color Denotes Fraction of Events in Run That Resemble This Event
    #5 : Similarity Count
    #6 : Impulsivity
    #7 : Max Correlation Value (All Hpol Baselines)
    #8 : Max Correlation Background Templates
    #9 : Similarity count from hist(sum(delays,axis=1))

    #################################
    corr_length = 262144#32768
    align_method = 0

    # Align method can be one of a few:
    # 0: argmax of corrs (default)
    # 1: argmax of hilbert of corrs
    # 2: Average of argmin and argmax
    # 3: argmin of corrs
    # 4: Pick the largest max peak preceding the max of the hilbert of corrs
    # 5: Pick the average indices of values > 95% peak height in corrs
    # 6: Pick the average indices of values > 98% peak height in hilbert of corrs
    # 7: Gets argmax of abs(corrs) and then finds highest positive peak before this value
    # 8: Apply cfd to waveforms to get first pass guess at delays, then pick the best correlation near that. 

    # filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_%i-align_%i'%(corr_length,align_method)
    

    #filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_262144-align_0'
    filter_string = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_0-shorten_signals-1-shorten_thresh-0.70-shorten_delay-10.00-shorten_length-90.00'
    

    #filter_string = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_%i-align_0'%corr_length
    #filter_string = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-corlen_%i-align_0'%corr_length
    #filter_string = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_%i-align_4'%corr_length
    #################################
    # CUT PARAMETERS
    #################################
    cut_on_77MHz_delays = False #This will apply cuts based on the 77MHz time delays stored in background_timedelays folder.
    matching_delays = 10 #Number of required time delays to match within ns_threshold to be ignored
    ns_threshold = 1.0
    cut77MHz_filter_string = filter_string #These will be the time delays that will compare to the previously generated template times.

    apply_hpol_corr_cut = False
    hpol_corr_threshold = 0.5 #This minimum value of corr vals to pass threshold. 
    hpol_minimum_correlating_baselines = 4 # Atleast this many of the 6 baselines must pass the above threshold for the event to count.

    apply_vpol_corr_cut = False
    vpol_corr_threshold = 0.6 #This minimum value of corr vals to pass threshold. 
    vpol_minimum_correlating_baselines = 4 # Atleast this many of the 6 baselines must pass the above threshold for the event to count.

    apply_rough_dir_cut = False
    cut_angle = 100 #Angles below this are ignored
    map_filter_string = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_0-Hilb_0-upsample_32768-maxmethod_0' #The settings used you want to cut dir on. 

    cut_48MHz_signals = False
    cut_43MHz_signals = False
    cut_36MHz_signals = False #36.1 ish
    cut_33MHz_signals = False

    apply_template_cut = False

    apply_similarity_count_cut = True
    similarity_count_cut_limit = 3000

    apply_hpol_impuslivity_cut = False
    hpol_impulsivity_threshold = 0.5

    apply_vpol_impuslivity_cut = False
    vpol_impulsivity_threshold = 0.4


    apply_similarity_hist_cut = False #Here the time delays for a each event are summed to create one number.  These are then histogrammed.
    similarity_hist_nbins = 300 #Number of bins in histogram
    similarity_hist_threshold = 100 #Events in bins with less than this number of counts are plotted.  





    

    for run_index, run in enumerate(runs):
        try:
            print('\nProcessing Run %i'%run)
            reader = Reader(datapath,run)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.  
            if filename is not None:
                with h5py.File(filename, 'r') as file:
                    try:
                        dsets = list(file.keys()) #Existing datasets
                        print(dsets)
                        print(list(file['time_delays'].keys()))
                        n_events_total = len(file['eventids'][...])
                        pol = 'hpol'

                        ############################
                        # Applying cuts on data (Done in 2 parts)
                        ############################                        
                        # Part 1
                        ############################
                        rf_cut = file['trigger_type'][...] == 2 #This is RF triggers.

                        ############################
                        if cut_on_77MHz_delays == True:

                            print('Cutting based on time delays calculated for 77 MHz dominated signals.\nEvents with less than %i matching time delays (w/in %0.2f ns) will be plotted.'%(matching_delays,ns_threshold))

                            template_77MHz_time_delays = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'background_timedelays/time_delays_77MHz_type1.csv') #Time delays for background 77 MHz signal.  Within some window of time delays events should be ignored. Type 1 is a crosspol event with better timing than the type 0, which was mostly only hpol.  

                            time_delays_for_77MHz_cut = numpy.vstack((file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(0,1)][...],file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(0,2)][...],file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(0,3)][...],file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(1,2)][...],file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(1,3)][...],file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(2,3)][...])).T
                            cut77MHz_delays = numpy.sum(time_delays_for_77MHz_cut - template_77MHz_time_delays[None,0:6] < ns_threshold,axis=1) < matching_delays #matching_delays of the time delays within ns_threshold ns of the template direction.
                            if False:
                                for pair_index, pair in enumerate([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]):
                                    plt.figure()
                                    plt.title(str(pair))
                                    plt.hist(file['time_delays'][cut77MHz_filter_string]['hpol_t_%isubtract%i'%(pair[0],pair[1])][rf_cut],bins=1000,range=(-150,150),log=True)
                                    plt.axvline(template_77MHz_time_delays[pair_index],c='r',linestyle='--',alpha=0.5)
                                    plt.minorticks_on()
                                    plt.grid(b=True, which='major', color='k', linestyle='-')
                                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                                    plt.ylabel('Counts')
                                    plt.xlabel('Time Delay (ns)')
                        else:
                            cut77MHz_delays = numpy.ones(n_events_total)

                        ############################
                        if apply_rough_dir_cut == True:
                            print('Applying cut on direction as calculated and stored in file.\n  Only showing events with zenith < %0.2f'%cut_angle)
                            rough_dir_cut = file['map_direction'][map_filter_string]['hpol_ENU_zenith'][...] < cut_angle #CURRENTLY ALL ANGLES, BASICALLY DISABLING #Above horizen.  Don't trust calibration enough for this to be perfect.
                        else:
                            rough_dir_cut = numpy.ones(n_events_total)

                        ############################
                        inband_cut = numpy.ones(n_events_total)

                        if cut_48MHz_signals == True:
                            print('Applying cut on signals with a dominant spectral Frequency of ~ 48 MHz.  These will not be shown. ')
                            inband_cut = numpy.logical_and(inband_cut,~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...] < 49,axis=1), numpy.any(file['inband_peak_freq_MHz'][...] > 48,axis=1))) #Cutting out known CW

                        ############################
                        if cut_43MHz_signals == True:
                            print('Applying cut on signals with a dominant spectral Frequency of ~ 43.4 MHz.  These will not be shown. ')
                            inband_cut = numpy.logical_and(inband_cut,~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...] < 45,axis=1), numpy.any(file['inband_peak_freq_MHz'][...] > 43,axis=1))) #Cutting out known CW

                        ############################
                        if cut_33MHz_signals == True:
                            print('Applying cut on signals with a dominant spectral Frequency of ~ 33 MHz.  These will not be shown. ')
                            inband_cut = numpy.logical_and(inband_cut,~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...] < 33.8,axis=1), numpy.any(file['inband_peak_freq_MHz'][...] > 33.0,axis=1))) #Cutting out known CW

                        ############################
                        if cut_36MHz_signals == True:
                            print('Applying cut on signals with a dominant spectral Frequency of ~ 36.13 MHz.  These will not be shown. ')
                            inband_cut = numpy.logical_and(inband_cut,~numpy.logical_and(numpy.any(file['inband_peak_freq_MHz'][...] < 36.2,axis=1), numpy.any(file['inband_peak_freq_MHz'][...] > 36.0,axis=1))) #Cutting out known CW
                        ############################


                        ############################
                        # Depending on the alignment method these could result in multiple output values.
                        print(list(file['time_delays'].keys()))
                        output_shape = numpy.shape(file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,1)][...])
                        ############################

                        if apply_hpol_corr_cut == True:
                            print('Applying cut on signals based on their maximum correlation values in hpol.')
                            all_hpol_corr_vals = numpy.vstack((file['time_delays'][filter_string]['hpol_max_corr_0subtract1'][...],file['time_delays'][filter_string]['hpol_max_corr_0subtract2'][...],file['time_delays'][filter_string]['hpol_max_corr_0subtract3'][...],file['time_delays'][filter_string]['hpol_max_corr_1subtract2'][...],file['time_delays'][filter_string]['hpol_max_corr_1subtract3'][...],file['time_delays'][filter_string]['hpol_max_corr_2subtract3'][...])).T
                            all_hpol_corr_cut = numpy.sum(all_hpol_corr_vals > hpol_corr_threshold,axis=1) >= hpol_minimum_correlating_baselines
                        else:
                            all_hpol_corr_cut = numpy.ones(n_events_total)

                        ############################
                        if apply_vpol_corr_cut == True:
                            print('Applying cut on signals based on their maximum correlation values in vpol.')
                            all_vpol_corr_vals = numpy.vstack((file['time_delays'][filter_string]['vpol_max_corr_0subtract1'][...],file['time_delays'][filter_string]['vpol_max_corr_0subtract2'][...],file['time_delays'][filter_string]['vpol_max_corr_0subtract3'][...],file['time_delays'][filter_string]['vpol_max_corr_1subtract2'][...],file['time_delays'][filter_string]['vpol_max_corr_1subtract3'][...],file['time_delays'][filter_string]['vpol_max_corr_2subtract3'][...])).T
                            all_vpol_corr_cut = numpy.sum(all_vpol_corr_vals > vpol_corr_threshold,axis=1) >= vpol_minimum_correlating_baselines
                        else:
                            all_vpol_corr_cut = numpy.ones(n_events_total)

                        ############################
                        # Combining cuts
                        ############################
                        total_loading_cut = numpy.all(numpy.vstack((rf_cut,inband_cut,rough_dir_cut,cut77MHz_delays,all_hpol_corr_cut,all_vpol_corr_cut)),axis=0)

                        # print('\n\nNOTE FOR TESTING PURPOSES ONLY THE FIRST 100 EVENTS ARE BEING SHOWN, DELETE THIS LINE WHEN DONE.\n\n')
                        # total_loading_cut = numpy.multiply(total_loading_cut,numpy.cumsum(total_loading_cut) < 100)
                        ############################
                        # Loading Values with Cut
                        ############################

                        eventids = file['eventids'][total_loading_cut]
                        calibrated_trigtime = file['calibrated_trigtime'][total_loading_cut]
                        if len(output_shape) == 2:
                            total_loading_cut_indices = numpy.where(total_loading_cut)[0]
                            hpol_delays = numpy.transpose(numpy.array([file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,1)][total_loading_cut_indices,:],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,2)][total_loading_cut_indices,:],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,3)][total_loading_cut_indices,:],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,2)][total_loading_cut_indices,:],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,3)][total_loading_cut_indices,:],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(2,3)][total_loading_cut_indices,:]]),axes=(1,0,2))
                            vpol_delays = numpy.transpose(numpy.array([file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,1)][total_loading_cut_indices,:],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,2)][total_loading_cut_indices,:],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,3)][total_loading_cut_indices,:],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,2)][total_loading_cut_indices,:],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,3)][total_loading_cut_indices,:],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(2,3)][total_loading_cut_indices,:]]),axes=(1,0,2))
                        else:
                            hpol_delays = numpy.vstack((file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,1)][total_loading_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,2)][total_loading_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(0,3)][total_loading_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,2)][total_loading_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(1,3)][total_loading_cut],file['time_delays'][filter_string]['hpol_t_%isubtract%i'%(2,3)][total_loading_cut])).T
                            vpol_delays = numpy.vstack((file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,1)][total_loading_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,2)][total_loading_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(0,3)][total_loading_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,2)][total_loading_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(1,3)][total_loading_cut],file['time_delays'][filter_string]['vpol_t_%isubtract%i'%(2,3)][total_loading_cut])).T

                        ############################
                        # Applying cuts on data (Done in 2 parts)
                        ############################                        
                        # Part 2
                        ############################
                        if numpy.logical_or(colormap_mode == 9,apply_similarity_hist_cut):
                            try:
                                vals = numpy.sum(hpol_delays,axis=1) #Works with higher number of entries from align_method_13
                                n, bins = numpy.histogram(vals,bins=similarity_hist_nbins)
                                
                                bin_numbers = numpy.zeros(numpy.shape(vals),dtype=int)
                                for i in range(len(bins)-1):
                                    bin_numbers[numpy.logical_and(vals >= bins[i], vals < bins[i+1])] = i

                                similarity_hist_counts = n[bin_numbers]

                                if apply_similarity_hist_cut:
                                    print('Applying cut based on histogram made of summed time delays values.')
                                    similarity_hist_cut = similarity_hist_counts < similarity_hist_threshold
                                else:
                                    similarity_hist_cut = numpy.ones(sum(total_loading_cut))
                            except Exception as e:
                                print(e)
                                print('This is not designed to work for align_method_13')
                        else:
                            similarity_hist_cut = numpy.ones(sum(total_loading_cut))

                        ############################
                        if numpy.logical_or(apply_template_cut == True, colormap_mode in [8]):
                            background_template_files = numpy.array(glob.glob(os.environ['BEACON_ANALYSIS_DIR']+'templates/*.csv'))
                            background_template_roots = numpy.array([f.split('/')[-1].replace('.csv','') for f in background_template_files])

                            for root_index, root in enumerate(background_template_roots):
                                if root_index == 0:
                                    max_template_vals = numpy.max(file['template_correlations'][root][...][total_loading_cut,:],axis=1)
                                else:
                                    max_template_vals = numpy.max(numpy.vstack((max_template_vals,numpy.max(file['template_correlations'][root][...][total_loading_cut,:],axis=1))),axis=0)
                            
                            if apply_template_cut == True:
                                template_cut_threshold = 1.0
                                print('Appling a cut on events based on maximum correlation values with previously generated templates.\n Only events with max correlation less than %0.2f are shown.'%template_cut_threshold)
                                template_cut = max_template_vals < template_cut_threshold
                            else:
                                template_cut = numpy.ones(sum(total_loading_cut))
                        else:
                            template_cut = numpy.ones(sum(total_loading_cut))

                        
                        ############################

                        if numpy.logical_or(colormap_mode in [1,2,4,5],apply_similarity_count_cut == True):
                            #similarity_count needed for colormap at least (if not also cut)
                            try:
                                print('KEYS FOR SIMILARAITY_COUNT:')
                                print(list(file['similarity_count'].keys()))
                                file['similarity_count'][filter_string]['%s_count'%(pol)][total_loading_cut]
                                similarity_count = file['similarity_count'][filter_string]['%s_count'%(pol)][total_loading_cut]

                                if False:
                                    plt.figure()
                                    plt.hist(file['similarity_count'][filter_string]['%s_count'%('hpol')][total_loading_cut],label='hpol similarity count',bins=1000)
                                    plt.hist(file['similarity_count'][filter_string]['%s_count'%('vpol')][total_loading_cut],label='vpol similarity count',bins=1000)
                            except Exception as e:
                                print(run)
                                print(e)
                                exc_type, exc_obj, exc_tb = sys.exc_info()
                                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                print(exc_type, fname, exc_tb.tb_lineno)
                                print('Similarity count not previously performed, or is corrupt.  Attempting to perform now.')
                                similarity_count = countSimilar(hpol_delays)

                            if apply_similarity_count_cut == True:
                                similarity_cut = similarity_count < similarity_count_cut_limit #Less than this number of similar events in a run to show up.
                                print('Applying cut on events based on similarity_count.  Events with a similarity_count < %i will be shown.'%similarity_count_cut_limit)
                            else:
                                similarity_cut = numpy.ones(sum(total_loading_cut))
                        else:
                            similarity_cut = numpy.ones(sum(total_loading_cut))

                        ############################

                        impulsivity_cut = numpy.ones(sum(total_loading_cut))
                        if numpy.any([apply_hpol_impuslivity_cut == True, apply_vpol_impuslivity_cut == True, colormap_mode in [6]]):
                            impulsivity_hpol = file['impulsivity'][filter_string]['hpol'][...][total_loading_cut]
                            impulsivity_vpol = file['impulsivity'][filter_string]['vpol'][...][total_loading_cut]

                            if apply_hpol_impuslivity_cut:
                                print('Applying cut on impulsivity of hpol signals.  Events with a impulsivity > %0.2f will be shown.'%hpol_impulsivity_threshold)
                                impulsivity_cut = numpy.logical_and(impulsivity_cut, impulsivity_hpol > hpol_impulsivity_threshold)

                            if apply_vpol_impuslivity_cut:
                                print('Applying cut on impulsivity of vpol signals.  Events with a impulsivity > %0.2f will be shown.'%vpol_impulsivity_threshold)
                                impulsivity_cut = numpy.logical_and(impulsivity_cut, impulsivity_vpol > vpol_impulsivity_threshold)
                                if False:
                                    try:
                                        plt.figure()
                                        plt.subplot(2,1,1)
                                        plt.hist(impulsivity_hpol[similarity_count > 1000.0],bins=100,label='similarity_count > 10.0')
                                        plt.hist(impulsivity_hpol[similarity_count < 1000.0],bins=100,label='similarity_count < 10.0')
                                        plt.legend()
                                        plt.xlabel('impulsivity_hpol')
                                        plt.subplot(2,1,2)
                                        plt.hist(impulsivity_vpol[similarity_count > 1000.0],bins=100,label='similarity_count > 10.0')
                                        plt.hist(impulsivity_vpol[similarity_count < 1000.0],bins=100,label='similarity_count < 10.0')
                                        plt.legend()
                                        plt.xlabel('impulsivity_vpol')
                                    except Exception as e:
                                        file.close()
                                        print('Error in plotting.')
                                        print(e)
                                        exc_type, exc_obj, exc_tb = sys.exc_info()
                                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                        print(exc_type, fname, exc_tb.tb_lineno)

                        ############################
                        ignore_unreal = numpy.any(abs(hpol_delays) > 300,axis=1) #might be too strict for ones that don't have 4 visible pulses.
                        if len(output_shape) == 2:
                            ignore_unreal = numpy.any(ignore_unreal,axis=1) 
                        

                        ############################
                        # Combining cuts
                        ############################
                        cut = numpy.all(numpy.vstack((~ignore_unreal,similarity_cut,template_cut,impulsivity_cut,similarity_hist_cut)),axis=0)

                        if colormap_mode == 0:
                            c = file['inband_peak_freq_MHz'][...][total_loading_cut,0][cut]
                            suptitle = 'Color Corresponds to Peak Frequency In Band (MHz)'
                            norm = None
                        elif colormap_mode == 1:
                            uniqueness = 1 - similarity_count/max(similarity_count)
                            c = uniqueness[cut]
                            suptitle = 'Uniqueness (1 = Unique, 0 = Common)'
                            norm = None
                        elif colormap_mode == 2:
                            c = similarity_count[cut]
                            suptitle = 'Number of Similar Events in Run'
                            if max(similarity_count[cut]) >= 100:
                                norm = LogNorm()
                            else:
                                norm = None
                        elif colormap_mode == 3:
                            c = file['map_direction'][map_filter_string]['hpol_ENU_zenith'][...][total_loading_cut][cut]
                            suptitle = 'hpol_max_corr_dir_ENU_zenith (deg)'
                            norm = None
                        elif colormap_mode == 4:
                            c = similarity_count[cut]/n_events_total
                            suptitle = 'Color Denotes Fraction of Events in Run That Resemble This Event'
                            norm = LogNorm()
                        elif colormap_mode == 5:
                            c = similarity_count[cut]
                            suptitle = 'Similarity Count'
                            norm = None
                        elif colormap_mode == 6:
                            c = numpy.max(numpy.vstack((impulsivity_hpol[cut],impulsivity_vpol[cut])),axis=0)
                            suptitle = 'Impulsivity'
                            norm = None
                        elif colormap_mode == 7:
                            if len(output_shape) == 2:
                                all_hpol_corr_vals = numpy.transpose(numpy.array([file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,1)][total_loading_cut_indices,:][cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,2)][total_loading_cut_indices,:][cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(0,3)][total_loading_cut_indices,:][cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(1,2)][total_loading_cut_indices,:][cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(1,3)][total_loading_cut_indices,:][cut],file['time_delays'][filter_string]['hpol_max_corr_%isubtract%i'%(2,3)][total_loading_cut_indices,:][cut]]),axes=(1,0,2))
                            else:
                                all_hpol_corr_vals = numpy.vstack((file['time_delays'][filter_string]['hpol_max_corr_0subtract1'][...][total_loading_cut][cut],file['time_delays'][filter_string]['hpol_max_corr_0subtract2'][...][total_loading_cut][cut],file['time_delays'][filter_string]['hpol_max_corr_0subtract3'][...][total_loading_cut][cut],file['time_delays'][filter_string]['hpol_max_corr_1subtract2'][...][total_loading_cut][cut],file['time_delays'][filter_string]['hpol_max_corr_1subtract3'][...][total_loading_cut][cut],file['time_delays'][filter_string]['hpol_max_corr_2subtract3'][...][total_loading_cut][cut])).T
                            c = numpy.max(all_hpol_corr_vals, axis=1)
                            #something to do with correlation values, max for all baselines?
                            #file['hpol_max_corr_1subtract2'][...][total_loading_cut][cut]
                            suptitle = 'Max Correlation Value (All Hpol Baselines)'
                            norm = None
                        elif colormap_mode == 8:
                            c = max_template_vals[cut]
                            suptitle = 'Max Correlation Background Templates'
                            norm = None
                        elif colormap_mode == 9:
                            c = similarity_hist_counts[cut]
                            suptitle = 'Similarity Count from Hist of Summed Time Delays'
                            norm = LogNorm()
                        else:
                            c = numpy.ones_like(cut)
                            suptitle = ''
                            norm = None

                        if run_index == 0:
                            times = calibrated_trigtime[cut]
                            total_hpol_delays = hpol_delays[cut,:]
                            total_vpol_delays = vpol_delays[cut,:]
                            if numpy.logical_or(colormap_mode in [1,2,4,5],apply_similarity_count_cut == True):
                                total_similarity_count = similarity_count[cut]
                                similarity_percent = similarity_count[cut]/n_events_total
                            total_colors = c
                            total_eventids = eventids[cut]
                            total_runnum = numpy.ones_like(eventids[cut])*run

                        else:
                            total_hpol_delays = numpy.vstack((total_hpol_delays,hpol_delays[cut,:]))
                            total_vpol_delays = numpy.vstack((total_vpol_delays,vpol_delays[cut,:]))
                            times =  numpy.append(times,calibrated_trigtime[cut])
                            if len(c.shape) == 2:
                                total_colors = numpy.append(total_colors,c,axis=0)
                            else:
                                total_colors = numpy.append(total_colors,c)
                            if numpy.logical_or(colormap_mode in [1,2,4,5],apply_similarity_count_cut == True):
                                total_similarity_count = numpy.append(total_similarity_count,similarity_count[cut])
                                similarity_percent = numpy.append(similarity_percent,similarity_count[cut]/n_events_total)

                            total_eventids = numpy.append(total_eventids,eventids[cut])
                            total_runnum = numpy.append(total_runnum,numpy.ones_like(eventids[cut])*run) 

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

    if len(output_shape) == 2:
        times = numpy.tile(times,(output_shape[1],1)).T
    #times = calibrated_trigtime[cut]
    #total_hpol_delays = delays[cut,:]
    #similarity_percent = similarity_count/numpy.shape(delays)[0]
    try:
        ###############################################
        # Finding planes that are potentially visible #
        ###############################################
        if add_plane_tracks:
            flight_data_location_hdf5 = '/project2/avieregg/beacon/flight_backup_jan2020/data/altered/' #The files will be hdf5.
            
            #SHOULD GET START AND STOP FROM THE FILES
            start = numpy.min(times)
            stop = numpy.max(times)
            min_approach_cut_km = 50 #km
            #unique_flights,all_vals = getTracks(start,stop,min_approach_cut_km,hour_window = 12)
            flight_tracks_ENU, all_vals = pt.getENUTrackDict(start,stop,min_approach_cut_km,hour_window = 0,flights_of_interest=[])

            def addPlaneTracks(_ax, antenna_pair_list, _flight_tracks_ENU, add_label=True):
                min_y = 1000
                max_y = -1000
                for ant_i, ant_j in antenna_pair_list:
                    #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                    #NS Pairs = (0,2), (1,3)
                    #EW Pairs = (0,1), (2,3)
                    #Plot tracks
                    existing_locations_A = numpy.array([])
                    existing_locations_B = numpy.array([])

                    for flight in list(_flight_tracks_ENU.keys()):
                        track = _flight_tracks_ENU[flight]
                        tof, dof, dt = pt.getTimeDelaysFromTrack(track)
                        distance = numpy.sqrt(numpy.sum(track[:,0:3]**2,axis=1))/1000 #km
                        plot_distance_cut_limit = None
                        if plot_distance_cut_limit is not None:
                            plot_distance_cut = distance <= plot_distance_cut_limit
                        else:
                            plot_distance_cut = numpy.ones_like(distance,dtype=bool)

                        x = (track[plot_distance_cut,3] - numpy.min(times))/60.0
                        y = dt['expected_time_differences_hpol'][(ant_i, ant_j)][plot_distance_cut]
                        _ax.plot(x,y,linestyle = '--',alpha=0.5)
                        text_color = _ax.lines[-1].get_color()
                        #_ax.scatter(x,y,cmap=cm,norm=norm)

                        #Attempt at avoiding overlapping text.
                        text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
                        if existing_locations_A.size != 0:
                            if len(numpy.shape(existing_locations_A)) == 1:
                                dist = numpy.sqrt((text_loc[0]-existing_locations_A[0])**2 + (text_loc[1]-existing_locations_A[1])**2) #weird units but works
                                while dist < 15:
                                    text_loc[1] -= 1
                                    dist = numpy.sqrt((text_loc[0]-existing_locations_A[0])**2 + (text_loc[1]-existing_locations_A[1])**2) #weird units but works
                            else:
                                dist = min(numpy.sqrt((existing_locations_A[:,0] - text_loc[0])**2 + (existing_locations_A[:,1] - text_loc[1])**2))
                                while dist < 15:
                                    text_loc[1] -= 1
                                    dist = min(numpy.sqrt((existing_locations_A[:,0] - text_loc[0])**2 + (existing_locations_A[:,1] - text_loc[1])**2)) #weird units but works
                            existing_locations_A = numpy.vstack((existing_locations_A,text_loc))
                        else:
                            existing_locations_A = text_loc           
                        if add_label == True:
                            _ax.text(text_loc[0],text_loc[1],flight,color=text_color,withdash=True)
                        _max_y = max(y)
                        _min_y = min(y)
                        if _max_y > max_y:
                            max_y = _max_y
                        if _min_y < min_y:
                            min_y = _min_y
                _ax.set_ylim(min(1.1*min_y,0.9*min_y),max(1.1*max_y,0.9*max_y))
                return _ax

        if False:
            remove_top_n_bins = 0
            n_bins = 500
            remove_indices = numpy.array([])


            plt.figure()
            bins = numpy.linspace(numpy.min(total_hpol_delays[:,0:3]),numpy.max(total_hpol_delays[:,0:3]),n_bins+1)
            plt.subplot(3,1,1)
            n, outbins, patches = plt.hist(total_hpol_delays[:,0],bins=bins,label = 't0 - t1')
            plt.xlabel('t0 - t1 (ns)')
            plt.ylabel('Counts')
            plt.yscale('log', nonposy='clip')
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            top_left_edges = numpy.argsort(n)[::-1][0:remove_top_n_bins]
            for left_bin_edge_index in top_left_edges:
                if left_bin_edge_index == len(n):
                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(total_hpol_delays[:,0] >= outbins[left_bin_edge_index],total_hpol_delays[:,0] <= outbins[left_bin_edge_index + 1]) )[0]) #Different bin boundary condition on right edge. 
                else:
                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(total_hpol_delays[:,0] >= outbins[left_bin_edge_index],total_hpol_delays[:,0] < outbins[left_bin_edge_index + 1]) )[0])



            plt.subplot(3,1,2)
            n, outbins, patches = plt.hist(total_hpol_delays[:,1],bins=bins,label = 't0 - t2')
            plt.xlabel('t0 - t2 (ns)')
            plt.ylabel('Counts')
            plt.yscale('log', nonposy='clip')
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            top_left_edges = numpy.argsort(n)[::-1][0:remove_top_n_bins]
            for left_bin_edge_index in top_left_edges:
                if left_bin_edge_index == len(n):
                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(total_hpol_delays[:,1] >= outbins[left_bin_edge_index],total_hpol_delays[:,1] <= outbins[left_bin_edge_index + 1]) )[0]) #Different bin boundary condition on right edge. 
                else:
                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(total_hpol_delays[:,1] >= outbins[left_bin_edge_index],total_hpol_delays[:,1] < outbins[left_bin_edge_index + 1]) )[0])


            plt.subplot(3,1,3)
            n, outbins, patches = plt.hist(total_hpol_delays[:,2],bins=bins,label = 't0 - t3')
            plt.xlabel('t0 - t3 (ns)')
            plt.ylabel('Counts')
            plt.yscale('log', nonposy='clip')
            plt.minorticks_on()
            plt.grid(b=True, which='major', color='k', linestyle='-')
            plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            top_left_edges = numpy.argsort(n)[::-1][0:remove_top_n_bins]
            for left_bin_edge_index in top_left_edges:
                if left_bin_edge_index == len(n):
                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(total_hpol_delays[:,2] >= outbins[left_bin_edge_index],total_hpol_delays[:,2] <= outbins[left_bin_edge_index + 1]) )[0]) #Different bin boundary condition on right edge. 
                else:
                    remove_indices = numpy.append(remove_indices,numpy.where( numpy.logical_and(total_hpol_delays[:,2] >= outbins[left_bin_edge_index],total_hpol_delays[:,2] < outbins[left_bin_edge_index + 1]) )[0])

            #remove_indices cuts from the time total_hpol_delays before [cut] is taken.  
            remove_indices = numpy.sort(numpy.unique(remove_indices))

        if False:
            fig = plt.figure()
            scatter = plt.scatter(total_hpol_delays[:,0],total_hpol_delays[:,1],c=(times - min(times))/60,cmap=cm)
            cbar = fig.colorbar(scatter)
            plt.xlabel('t0 - t1')
            plt.ylabel('t0 - t2')

            fig = plt.figure()
            scatter = plt.scatter(total_hpol_delays[:,0],total_hpol_delays[:,2],c=(times - min(times))/60,cmap=cm)
            cbar = fig.colorbar(scatter)
            cbar.ax.set_ylabel('Time from Start of Run (min)', rotation=270)
            plt.xlabel('t0 - t1')
            plt.ylabel('t0 - t3')

        ax1_exists = False
        if True:

            c = total_colors
            hist_cut = numpy.ones(len(times)).astype(bool)
            cut_eventids = total_eventids
            cut_runnum = total_runnum

            if True:
                #NORTH SOUTH SENSITIVE PLOT
                #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                #NS Pairs = (0,2), (1,3)
                #EW Pairs = (0,1), (2,3)

                fig_A = plt.figure()
                plt.suptitle('North South ' + suptitle)
                ax1 = plt.subplot(2,1,1)
                ax1_exists = True
                scatter1 = plt.scatter((times[hist_cut] - numpy.min(times))/60.0, total_hpol_delays[hist_cut,1],label = 't0 - t2',c=c,cmap=cm,norm=norm)
                
                if add_plane_tracks:
                    ax1 = addPlaneTracks(ax1, [(0,2)], flight_tracks_ENU)
                #cbar = ax1.colorbar(scatter)
                plt.xlabel('Calibrated trigtime From Start of Run (min)')
                plt.ylabel('t0 - t2 (ns)')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                
                divider = make_axes_locatable(ax1)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig_A.colorbar(scatter1, cax=cax, orientation='vertical')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                ax2 = plt.subplot(2,1,2,sharex=ax1)
                
                scatter2 = plt.scatter((times[hist_cut] - numpy.min(times))/60.0, total_hpol_delays[hist_cut,4],label = 't1 - t3',c=c,cmap=cm,norm=norm)
                if add_plane_tracks:
                    ax2 = addPlaneTracks(ax2, [(1,3)], flight_tracks_ENU)
                #cbar = ax2.colorbar(scatter)
                plt.xlabel('Calibrated trigtime From Start of Run (min)')
                plt.ylabel('t1 - t3 (ns)')
                plt.minorticks_on()
                plt.legend()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                divider = make_axes_locatable(ax2)
                cax2 = divider.append_axes('right', size='5%', pad=0.05)
                fig_A.colorbar(scatter2, cax=cax2, orientation='vertical')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                selector1 = SelectFromCollection(ax1, scatter1,cut_eventids,cut_runnum,total_hpol_delays[hist_cut],total_vpol_delays[hist_cut])
                selector2 = SelectFromCollection(ax2, scatter2,cut_eventids,cut_runnum,total_hpol_delays[hist_cut],total_vpol_delays[hist_cut])

                skipperA = Skipper(ax2) #On the lower axis.  

            if True:
                #EAST WEST SENSITIVE PLOT
                #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                #NS Pairs = (0,2), (1,3)
                #EW Pairs = (0,1), (2,3)

                
                fig_B = plt.figure()
                plt.suptitle('East West ' + suptitle)
                if ax1_exists:
                    ax3 = plt.subplot(2,1,1,sharex=ax1)
                else:
                    ax3 = plt.subplot(2,1,1)
                
                scatter3 = plt.scatter((times[hist_cut] - numpy.min(times))/60.0, total_hpol_delays[hist_cut,0],label = 't0 - t1',c=c,cmap=cm,norm=norm)
                if add_plane_tracks:
                    ax3 = addPlaneTracks(ax3, [(0,1)], flight_tracks_ENU)
                #cbar = ax3.colorbar(scatter)
                plt.xlabel('Calibrated trigtime From Start of Run (min)')
                plt.ylabel('t0 - t1 (ns)')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                
                divider = make_axes_locatable(ax3)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig_B.colorbar(scatter3, cax=cax, orientation='vertical')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)



                ax4 = plt.subplot(2,1,2,sharex=ax3)

                if ax1_exists:
                    ax4 = plt.subplot(2,1,2,sharex=ax1)
                else:
                    ax4 = plt.subplot(2,1,2,sharex=ax3)
                
                scatter4 = plt.scatter((times[hist_cut] - numpy.min(times))/60.0, total_hpol_delays[hist_cut,5],label = 't2 - t3',c=c,cmap=cm,norm=norm)
                if add_plane_tracks:
                    ax4 = addPlaneTracks(ax4, [(2,3)], flight_tracks_ENU)
                #cbar = ax4.colorbar(scatter)
                plt.xlabel('Calibrated trigtime From Start of Run (min)')
                plt.ylabel('t2 - t3 (ns)')
                plt.minorticks_on()
                plt.legend()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                divider = make_axes_locatable(ax4)
                cax4 = divider.append_axes('right', size='5%', pad=0.05)
                fig_B.colorbar(scatter4, cax=cax4, orientation='vertical')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                selector3 = SelectFromCollection(ax3, scatter3,cut_eventids,cut_runnum,total_hpol_delays[hist_cut],total_vpol_delays[hist_cut])
                selector4 = SelectFromCollection(ax4, scatter4,cut_eventids,cut_runnum,total_hpol_delays[hist_cut],total_vpol_delays[hist_cut])

                skipperB = Skipper(ax4) #On the lower axis.
                if ~ax1_exists:
                    ax3.set_xlim(0,60)
            if True:
                #Remaining PLOT
                #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
                #NS Pairs = (0,2), (1,3)
                #EW Pairs = (0,1), (2,3)
                #Remaining Pairs = (0,3), (1,2)

                
                fig_C = plt.figure()
                plt.suptitle('East West ' + suptitle)
                if ax1_exists:
                    ax5 = plt.subplot(2,1,1,sharex=ax1)
                else:
                    ax6 = plt.subplot(2,1,1)
                
                scatter5 = plt.scatter((times[hist_cut] - numpy.min(times))/60.0, total_hpol_delays[hist_cut,2],label = 't0 - t3',c=c,cmap=cm,norm=norm)
                if add_plane_tracks:
                    ax5 = addPlaneTracks(ax5, [(0,3)], flight_tracks_ENU)
                #cbar = ax5.colorbar(scatter)
                plt.xlabel('Calibrated trigtime From Start of Run (min)')
                plt.ylabel('t0 - t3 (ns)')
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                
                divider = make_axes_locatable(ax5)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                fig_C.colorbar(scatter5, cax=cax, orientation='vertical')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)



                if ax1_exists:
                    ax6 = plt.subplot(2,1,2,sharex=ax1)
                else:
                    ax6 = plt.subplot(2,1,2,sharex=ax5)
                
                scatter6 = plt.scatter((times[hist_cut] - numpy.min(times))/60.0, total_hpol_delays[hist_cut,3],label = 't1 - t2',c=c,cmap=cm,norm=norm)
                if add_plane_tracks:
                    ax6 = addPlaneTracks(ax6, [(1,2)], flight_tracks_ENU)
                #cbar = ax6.colorbar(scatter)
                plt.xlabel('Calibrated trigtime From Start of Run (min)')
                plt.ylabel('t1 - t2 (ns)')
                plt.minorticks_on()
                plt.legend()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                divider = make_axes_locatable(ax6)
                cax6 = divider.append_axes('right', size='5%', pad=0.05)
                fig_C.colorbar(scatter6, cax=cax6, orientation='vertical')
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


                selector5 = SelectFromCollection(ax5, scatter5,cut_eventids,cut_runnum,total_hpol_delays[hist_cut],total_vpol_delays[hist_cut])
                selector6 = SelectFromCollection(ax6, scatter6,cut_eventids,cut_runnum,total_hpol_delays[hist_cut],total_vpol_delays[hist_cut])


                skipperC = Skipper(ax6) #On the lower axis.
                if ~ax1_exists:
                    ax5.set_xlim(0,60)
            if ax1_exists:
                ax1.set_xlim(0,60)
            ############################
            '''
            c = total_colors
            cut = numpy.ones_like(c).astype(bool)

            fig = plt.figure()
            ax = plt.subplot(1,3,1)
            
            scatter1 = plt.scatter((times[cut] - numpy.min(times))/3600.0, total_hpol_delays[cut,0],label = 't0 - t1',c=c,cmap=cm,norm=norm)
            #cbar = ax.colorbar(scatter)
            plt.xlabel('Calibrated trigtime From Start of Run (h)')
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



            ax2 = plt.subplot(1,3,2,sharex=ax)
            
            scatter2 = plt.scatter((times[cut] - numpy.min(times))/3600.0, total_hpol_delays[cut,1],label = 't0 - t2',c=c,cmap=cm,norm=norm)
            #cbar = ax.colorbar(scatter)
            plt.xlabel('Calibrated trigtime From Start of Run (h)')
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

            ax3 = plt.subplot(1,3,3,sharex=ax)
            
            scatter3 = plt.scatter((times[cut] - numpy.min(times))/3600.0, total_hpol_delays[cut,2],label = 't0 - t3',c=c,cmap=cm,norm=norm)
            #cbar = ax.colorbar(scatter)
            plt.xlabel('Calibrated trigtime From Start of Run (h)')
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
            '''
    
    except Exception as e:
        file.close()
        print('Error in plotting.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    

