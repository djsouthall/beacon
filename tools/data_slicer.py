#!/usr/bin/env python3
'''
This contains the dataSlicer class that allows for plotting measured/saved analysis metrics against eachother in
mainly 2dhistogram format (with ROI highlighting support).  

This tool will hopefully serve to characterize and cut out particular sources of noise, or to probe patterns in
expected distroubtions of parameters. 
'''

import sys
import os
import inspect
import h5py
import copy
from collections import OrderedDict

import numpy
import scipy
import scipy.signal
import scipy.signal

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.info as info
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.get_sun_coords_from_timestamp import getSunAzEl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
import matplotlib
import matplotlib.pyplot as plt
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    plt.rcParams['toolbar'] = 'toolmanager'
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.backend_tools import ToolBase, ToolToggleBase, ToolQuit
from matplotlib.widgets import TextBox
plt.ion()

import astropy.units as apu
from astropy.coordinates import SkyCoord
import astropy
import astropy.time
import time
from datetime import datetime
from pytz import timezone,utc
import matplotlib.dates as mdates


raw_datapath = os.environ['BEACON_DATA']
processed_datapath = os.environ['BEACON_PROCESSED_DATA']

class dataSlicerSingleRun():
    '''
    Given a run/reader, and stored analysis file, this can produce 2d histogram plots for each of the known measureable 
    quantities, with contour functionality for highlighting regions of interest (ROI) as created by the user.  To see
    a list of the currently accepted known parameter keys use the printKnownParamKeys function.

    FOr ROI support, supply a dictionary to the addROI function to apply a cut on any (up to all) of the known parameters
    using this dictionary.  Cuts on each parameter are simple min/max window cuts, but by combining many parameters can
    be used to create higher dimensional parameter box cuts.
    
    With any ROI's defined, call the plotROI2dHist function, selecting 2 parameters you want histogrammed (2DHist).  
    This histogram will be plotted for all events passing early cuts like trigger_types and included_antennas (defined
    in the calling of the class).  Then the ROI's will be plotted on top of this underlying plot - shown as Contours.  
    These contours effectively circle where events passing a particular ROI's cuts show up as a population in the 
    plotted 2d histogram plot.  This can be used to further constrain the ROI or to simply understand it's behaviour
    better.   

    Note
    ----
    This operates on a per run basis, but a daughter class could be created to incorporate support for multiple runs.
    As most calculations require running through a single runs analysis file, I think this makes sense.  The histograms
    can just be added accross runs and hopefully the plotting functions could work with that. 

    Parameters
    ----------
    reader : beaconroot.examples.reader
        The event reader that serves as an ROOT interface between the stored run data and python.
    impulsivity_dset_key : str
        This is a string that must correspond to a specific stored and precalculated impulsivity dataset stored in the
        analysis h5py files.  This will be used whenever attempting to plot or cut on impulsivity values.
    time_delays_dset_key : str
        This is a string that must correspond to a specific stored and precalculated time delay dataset stored in the
        analysis h5py files.  This will be used whenever attempting to plot or cut on time delay values.
    map_dset_key : str
        This is a string that must correspond to a specific stored and precalculated map/alt-az dataset stored in the
        analysis h5py files.  This will be used whenever attempting to plot or cut on map/alt-az values.  Note that
        this will attempt to use this key to map datasets for BOTH maps using hilbert envelopes AND ones not using
        hilbert enevelopes.  It does so by changing the 0/1 in the name that refers to hilbert being enabled.
        If it succeeds in mapping both (i.e. both exist), then 'theta_best_h','theta_best_v','elevation_best_h',
        'elevation_best_v','phi_best_h','phi_best_v' will refer to the non-hilbert versions, and 'hilbert_theta_best_h',
        'hilbert_theta_best_v','hilbert_elevation_best_h','hilbert_elevation_best_v','hilbert_phi_best_h',
        'hilbert_phi_best_v' will all refer to the hilbert envelope versions.  If only the given dataset exists
        REGARDLESS OF WHETHER IT IS HILBERT OR NOT, it will just use the default keys, though titles will appropriately
        reflect if it is hilbert or not.  
    skip_common_setup : bool
        Should always be false unless running as a smaller portion of a whole, like in dataSlicer.  In taht case
        dataSlicer will calcualte the bins and store them in a non-redundent was as class attributes.
    curve_choice : int
        Which curve/cr template you wish to use when loading the template search results from the cr template search.
        Currently the only option is 0, which corresponds to a simple bi-polar delta function convolved with the known
        BEACON responses. 
    trigger_types : list of ints
        The trigger types you want to include in each plot.  This may be later done on a function by function basis, but
        for now is called at the beginning. 
    included_antennas : list of ints
        List should contain list of antennas trusted for this particular run.  This will be used in certain cuts which
        may look at the max or average values of a certain polarization.  If the antenna is not included on this list
        then it will not be included in those calculations.  This does not apply to any precalculated values that already
        washed out antenna specific information.  This can also be used to investigate particular antenna pairings by
        leaving only them remaining.  
    cr_template_n_bins_h : int
        The number of bins in the x dimension of the cr template search plot.
    cr_template_n_bins_v : int
        The number of bins in the y dimension of the cr template search plot.
    impulsivity_hv_n_bins : int
        The number of bins in the impulsivity_hv plot.
    std_n_bins_h : int
        The number of bins for plotting std of the h antennas.
    std_n_bins_v : int
        The number of bins for plotting std of the v antennas.
    max_std_val : float
        The max bin edge value for plotting std on both antennas.  Default is None.  If None is given then this will be
        automatically calculated (though likely too high).
    p2p_n_bins_h : int
        The number of bins for plotting p2p of the h antennas.
    p2p_n_bins_v : int
        The number of bins for plotting p2p of the v antennas.
    max_p2p_val : float
        The max bin edge value for plotting p2p on both antennas.  Default is 128.  If None is given then this will be
        automatically calculated.
    snr_n_bins_h : int
        The number of bins for plotting snr of the h antennas.
    snr_n_bins_v : int
        The number of bins for plotting snr of the v antennas.
    max_snr_val : float
        The max bin edge value for plotting snr on both antennas.  Default is None.  If None is given then this will be
        automatically calculated (though likely too high).
    n_phi : int
        The number of azimuthal angles to probe in the specified range.
    range_phi_deg : tuple of floats with len = 2 
        The specified range of azimuthal angles to probe.
    n_theta : int 
        The number of zenith angles to probe in the specified range.
    range_theta_deg : tuple of floats with len = 2  
        The specified range of zenith angles to probe.
    include_test_roi : bool
        This will include test regions of interest that are more for testing the class itself. 
    max_possible_map_value_n_bins_h : int
        The number of bins to use when plotting the max possible map value for each event.  
        Specifically for the hpol signals/map.
    max_possible_map_value_n_bins_v : int
        The number of bins to use when plotting the max possible map value for each event.  
        Specifically for the hpol signals/map.
    max_peak_to_sidelobe_val : float
        The maximum value for the range chosen to display the peak to sidelobe value value for each event.
    peak_to_sidelobe_n_bins_h : int
        The number of bins to use when plotting the peak to sidelobe param for hpol.
    peak_to_sidelobe_n_bins_v : int
        The number of bins to use when plotting the peak to sidelobe param for vpol.
    '''
    known_param_keys = [    'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v','min_snr_h','min_snr_v','snr_gap_h','snr_gap_v',\
                            'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
                            'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',\
                            'mean_max_corr_h', 'max_max_corr_h','mean_max_corr_v', 'max_max_corr_v','similarity_count_h','similarity_count_v','similarity_fraction_h','similarity_fraction_v',\
                            'max_corr_0subtract1_h','max_corr_0subtract2_h','max_corr_0subtract3_h','max_corr_1subtract2_h','max_corr_1subtract3_h','max_corr_2subtract3_h',\
                            'max_corr_0subtract1_v','max_corr_0subtract2_v','max_corr_0subtract3_v','max_corr_1subtract2_v','max_corr_1subtract3_v','max_corr_2subtract3_v',\
                            'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','elevation_best_all','elevation_best_choice','phi_best_h','phi_best_v','phi_best_all','phi_best_choice',\
                            'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','all_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value','all_max_possible_map_value','hpol_max_map_value','vpol_max_map_value','all_max_map_value',\
                            'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
                            'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
                            'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
                            'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v',\
                            'map_max_time_delay_0subtract1_all','map_max_time_delay_0subtract2_all','map_max_time_delay_0subtract3_all',\
                            'map_max_time_delay_1subtract2_all','map_max_time_delay_1subtract3_all','map_max_time_delay_2subtract3_all',\
                            'sun_az','sun_el','coincidence_method_1_h','coincidence_method_1_v','coincidence_method_2_h','coincidence_method_2_v']

    def __init__(self,  reader, impulsivity_dset_key, time_delays_dset_key, map_dset_key, skip_common_setup=False, analysis_data_dir=None, \
                        curve_choice=0, trigger_types=[1,2,3],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-200,max_time_delays_val=200,\
                        max_corr_n_bins=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=None,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=128,\
                        n_phi=181, range_phi_deg=(-180,180), n_theta=361, range_theta_deg=(0,180),\
                        max_possible_map_value_n_bins_h=100, max_possible_map_value_n_bins_v=100, max_possible_map_value_n_bins_all=100,\
                        max_map_value_n_bins_h=100, max_map_value_n_bins_v=100, max_map_value_n_bins_all=100,\
                        max_peak_to_sidelobe_val=5,peak_to_sidelobe_n_bins_h=100,peak_to_sidelobe_n_bins_v=100,peak_to_sidelobe_n_bins_all=100):
        try:
            self.updateReader(reader,analysis_data_dir=analysis_data_dir)
            self.math_keywords = ['SLICERSUBTRACT', 'SLICERADD', 'SLICERDIVIDE', 'SLICERMULTIPLY', 'SLICERMAX', 'SLICERMIN', 'SLICERMEAN'] #Meta words that will relate 2 known variables and produce a plot with their arithmatic combination. 

            if self.reader.failed_setup == False:
                if skip_common_setup == False:
                    #Angular ranges are handled such that their bin centers are the same as the values sampled by the corrolator class given the same min, max, and n.  
                    self.n_phi = n_phi
                    self.range_phi_deg = numpy.asarray(range_phi_deg)
                    dphi = (max(self.range_phi_deg) - min(self.range_phi_deg)) / (self.n_phi - 1)
                    self.phi_edges = numpy.arange(min(self.range_phi_deg),max(self.range_phi_deg) + 2*dphi, dphi) - dphi/2.0
                    self.phi_centers = 0.5*(self.phi_edges[1:]+self.phi_edges[:-1])
                    # self.phi_edges = numpy.arange(min(self.range_phi_deg) - numpy.diff(numpy.linspace(min(self.range_phi_deg),max(self.range_phi_deg),n_phi))[0]/2, max(self.range_phi_deg) + numpy.diff(numpy.linspace(min(self.range_phi_deg),max(self.range_phi_deg),n_phi))[0]/2, numpy.diff(numpy.linspace(min(self.range_phi_deg),max(self.range_phi_deg),n_phi))[0])

                    self.n_theta = n_theta
                    self.range_theta_deg = numpy.asarray(range_theta_deg)
                    dtheta = (max(self.range_theta_deg) - min(self.range_theta_deg)) / (self.n_theta - 1)
                    self.theta_edges = numpy.arange(min(self.range_theta_deg),max(self.range_theta_deg) + 2*dtheta, dtheta) - dtheta/2.0
                    self.theta_centers = 0.5*(self.theta_edges[1:]+self.theta_edges[:-1])
                    # self.theta_edges = numpy.arange(min(self.range_theta_deg) - numpy.diff(numpy.linspace(min(self.range_theta_deg),max(self.range_theta_deg),n_theta))[0]/2, max(self.range_theta_deg) + numpy.diff(numpy.linspace(min(self.range_theta_deg),max(self.range_theta_deg),n_theta))[0]/2, numpy.diff(numpy.linspace(min(self.range_theta_deg),max(self.range_theta_deg),n_theta))[0])

                    self.elevation_edges = 90 - self.theta_edges
                    self.elevation_centers = 0.5*(self.elevation_edges[1:]+self.elevation_edges[:-1])
                
                self.tct = None #This will be defined when necessary by functions below. 

                self.included_antennas = included_antennas#[0,1,2,3,4,5,6,7]
                self.included_hpol_antennas = numpy.array([0,2,4,6])[numpy.isin([0,2,4,6],self.included_antennas)]
                self.included_vpol_antennas = numpy.array([1,3,5,7])[numpy.isin([1,3,5,7],self.included_antennas)]

                #Parameters:
                #General Params:

                #Histogram preparations
                #These will be used for plotting each parameter against eachother. 

                #2dhist Params:
                #plot_2dhists = True
                self.cr_template_curve_choice = curve_choice #Which curve to select from correlation data.
                self.cr_template_n_bins_h = cr_template_n_bins_h
                self.cr_template_n_bins_v = cr_template_n_bins_v
                
                #Impulsivity Plot Params:
                self.impulsivity_dset_key = impulsivity_dset_key
                self.impulsivity_n_bins_h = impulsivity_n_bins_h
                self.impulsivity_n_bins_v = impulsivity_n_bins_v

                #Time Delay Plot Params:
                self.time_delays_dset_key = time_delays_dset_key
                self.time_delays_n_bins_h = time_delays_n_bins_h
                self.time_delays_n_bins_v = time_delays_n_bins_v
                self.min_time_delays_val = min_time_delays_val
                self.max_time_delays_val = max_time_delays_val

                #Cross correlation max values
                self.min_max_corr_val = 0.0
                self.max_max_corr_val = 1.0
                self.max_corr_n_bins = max_corr_n_bins

                #std Plot Params:
                self.min_std_val = 1.0 #To be rewritten, setting a reasonable lower bound for when max_std_val is given. 
                self.max_std_val = max_std_val
                self.std_n_bins_h = std_n_bins_h
                self.std_n_bins_v = std_n_bins_v

                #p2p Plot Params:
                self.max_p2p_val = max_p2p_val
                self.p2p_n_bins_h = p2p_n_bins_h
                self.p2p_n_bins_v = p2p_n_bins_v

                #snr Plot Params:
                self.max_snr_val = max_snr_val
                self.snr_n_bins_h = snr_n_bins_h
                self.snr_n_bins_v = snr_n_bins_v

                #max_possible_map_value Plot Params:
                self.max_max_possible_map_value_val = 1 #Too many maxes :(
                self.max_possible_map_value_n_bins_h = max_possible_map_value_n_bins_h
                self.max_possible_map_value_n_bins_v = max_possible_map_value_n_bins_v
                self.max_possible_map_value_n_bins_all = max_possible_map_value_n_bins_all

                #max_map_value Plot Params:
                self.max_max_map_value_val = 1 #Too many maxes :(
                self.max_map_value_n_bins_h = max_map_value_n_bins_h
                self.max_map_value_n_bins_v = max_map_value_n_bins_v
                self.max_map_value_n_bins_all = max_map_value_n_bins_all

                #peak_to_sidelobe Plot Params:
                self.max_peak_to_sidelobe_val = max_peak_to_sidelobe_val
                self.peak_to_sidelobe_n_bins_h = peak_to_sidelobe_n_bins_h
                self.peak_to_sidelobe_n_bins_v = peak_to_sidelobe_n_bins_v
                self.peak_to_sidelobe_n_bins_all = peak_to_sidelobe_n_bins_all
                self.peak_to_sidelobe_max_val = 5.0


                #Unknown, will be calculated as needed.
                self.max_beam_power = None
                self.max_beam_number = None

                #Map Direction Params:
                self.map_dset_key = map_dset_key
                if len(self.map_dset_key.split('deploy_calibration_')) > 1:
                    self.map_deploy_index = str(self.map_dset_key.split('deploy_calibration_')[-1].split('-n_phi')[0])
                    #self.map_deploy_index = int(self.map_dset_key.split('deploy_calibration_')[-1].split('-')[0])
                else:
                    self.map_deploy_index = None #Will use default

                self.checkForRateDatasets() #Will append to known param key based on which repeated rate signal datasets are available
                self.checkForComplementaryBothMapDatasets() #Will append to known param key and prepare for if hilbert used or not.
                

                self.trigger_types = trigger_types
                
                #In progress.
                #ROI  List
                #This should be a list with coords x1,x2, y1, y2 for each row.  These will be used to produce plots pertaining to 
                #these specific regions of interest.  x and y should be correlation values, and ordered.
                self.roi = {}
                if include_test_roi:
                    sample_ROI = self.printSampleROI(verbose=False)
                    self.roi['test'] = sample_ROI

                self.checkDataAvailability()

                self.masked_fallback_mode = 1

                #self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, numpy.shape(roi)[0])]
            else:
                print('Reader failed setup in dataSlicerSingleRun')
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def openSuccess(self):
        '''
        Will attempt to read the file.  Will return False is a problem arises.  
        '''
        try:
            with h5py.File(self.analysis_filename, 'r') as file:
                success = True
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print('dsets_present set to False')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            success = False
        return success

    def checkDataAvailability(self,verbose=False):
        '''
        This will check whether the given datasets are actually available for the given run.
        '''
        try:
            with h5py.File(self.analysis_filename, 'r') as file:
                if numpy.all( numpy.isin(['map_direction','map_direction','time_delays'] ,list(file.keys()))):
                    map_present = self.map_dset_key in list(file['map_direction'].keys())
                    if verbose:
                        print('map_present', map_present)
                        print(list(file['map_direction'].keys()))
                    impulsivity_present = self.impulsivity_dset_key in list(file['impulsivity'].keys())
                    if verbose:
                        print('impulsivity_present', impulsivity_present)
                        print(list(file['impulsivity'].keys()))
                    time_delays_present = self.time_delays_dset_key in list(file['time_delays'].keys())
                    if verbose:
                        print('time_delays_present', time_delays_present)
                        print(list(file['time_delays'].keys()))

                    self.dsets_present = numpy.all([map_present,impulsivity_present,time_delays_present])
                    if False:
                        print(self.map_dset_key in list(file['map_direction'].keys()))
                        print(self.map_dset_key)
                        print(list(file['map_direction'].keys()))
                        print(self.reader.run)
                        print([map_present,impulsivity_present,time_delays_present])
                else:
                    self.dsets_present = False


        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print('dsets_present set to False')
            self.dsets_present = False
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def printKnownParamKeys(self):
        '''
        This will return a list of the currently supported parameter keys for making 2d histogram plots.
        '''
        print(self.known_param_keys)
        return self.known_param_keys

    def checkForRateDatasets(self, verbose=True):
        '''
        This will look for the datasets corresponding to analyze_event_rate_frequency.py and add them to known_param_keys.

        If the gaussian fit of the randomized data (as generated in analyze_event_rate_frequency.py) is available then
        an additional dataset option will be made available for which the TS of each data will instead be presented
        as a number of standard deviations it is from the equivalent random dataset.  High values of this are highly
        likely to be associated with events of repeating sources like 60 Hz events.  Similar to the calculation below: 
        
        # gaus_fit_popt = [0,0,0]
        # popt[0] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_scale']
        # popt[1] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_mean']
        # popt[2] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_sigma']
        # normalize_by_window_index = file['event_rate_testing'][rate_string][time_window_string].attrs['normalize_by_window_index']

        # metric_true = file['event_rate_testing'][rate_string][time_window_string][...][loaded_eventids]

        # sigma = (metric_true - popt[1])/popt[2]
        '''
        try:
            added_param_keys = []
            self.event_rate_gaus_param = {}
            with h5py.File(self.analysis_filename, 'r') as file:
                try:
                    event_rate_testing_dsets = list(file['event_rate_testing'].keys())
                    for rate_string in event_rate_testing_dsets:
                        self.event_rate_gaus_param[rate_string] = {}
                        event_rate_testing_subsets = list(file['event_rate_testing'][rate_string].keys())
                        for time_window_string in event_rate_testing_subsets:
                            param_key = 'event_rate_ts_%s_%s'%(rate_string,time_window_string)
                            self.known_param_keys.append(param_key)
                            added_param_keys.append(param_key)
                            if numpy.all(numpy.isin(numpy.array(['random_fit_scale','random_fit_mean','random_fit_sigma']) , file['event_rate_testing'][rate_string][time_window_string].attrs)):
                                self.event_rate_gaus_param[rate_string][time_window_string] = {}
                                self.event_rate_gaus_param[rate_string][time_window_string]['scale'] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_scale']
                                self.event_rate_gaus_param[rate_string][time_window_string]['mean'] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_mean']
                                self.event_rate_gaus_param[rate_string][time_window_string]['sigma'] = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_sigma']
                                param_key2 = 'event_rate_sigma_%s_%s'%(rate_string,time_window_string)
                                self.known_param_keys.append(param_key2)
                                added_param_keys.append(param_key2)
                    file.close()
                except Exception as e:
                    file.close()
                    print('\nError in %s'%inspect.stack()[0][3])
                    print('Run: ',self.reader.run)
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

            if verbose:
                print('Event rate datasets in file and available for slicing:')
                for param_key in added_param_keys: 
                    print('\t' + param_key)

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def checkForComplementaryBothMapDatasets(self, verbose=True):
        '''
        This will use the given dset key for map data, and determine whether it is using hilbert envelopes or not.
        It will then check for the presence of the counterpart dataset (no hilbert enbelopes), and make the appropriate 
        preperations to allow both datasets to be available.

        Similarly for each, it will attempt to determine the "scope" of the dataset (abovehorizon, belowhorizon, 
        allsky), and then identify if the other options are available. 
        '''


        # 'hilbert_theta_best_h','hilbert_theta_best_v','hilbert_elevation_best_h','hilbert_elevation_best_v','hilbert_phi_best_h','hilbert_phi_best_v'
        #'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1'

        try:
            self.hilbert_map = bool(self.map_dset_key.lower().split('hilb')[1][1]) #split by hilbert, choosing string to right, the first digit after the underscore should be the bool
            self.normal_map = not self.hilbert_map #Default before checking if both dsets exist is that only the one given exists.

            self.map_dset_key_hilbert = self.map_dset_key  #Fallback

            #map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_30-n_phi_1440-min_phi_neg180-max_phi_180-n_theta_720-min_theta_0-max_theta_180-scope_allsky'

            self.allsky_normal_map = self.normal_map and 'scope_allsky' in self.map_dset_key
            self.belowhorizon_normal_map = self.normal_map and 'scope_belowhorizon' in self.map_dset_key
            self.abovehorizon_normal_map = self.normal_map and 'scope_abovehorizon' in self.map_dset_key

            self.allsky_hilbert_map = self.hilbert_map and 'scope_allsky' in self.map_dset_key
            self.belowhorizon_hilbert_map = self.hilbert_map and 'scope_belowhorizon' in self.map_dset_key
            self.abovehorizon_hilbert_map = self.hilbert_map and 'scope_abovehorizon' in self.map_dset_key

            if 'Hilb_' in self.map_dset_key:
                initial_hilbert_value = int(bool(self.map_dset_key.lower().split('hilb')[1][1]))
                initial_hilbert = 'Hilb_%i'%initial_hilbert_value
            else:
                initial_hilbert = None


            if self.allsky_normal_map or self.allsky_hilbert_map:
                initial_scope = 'scope_allsky'
            elif self.belowhorizon_normal_map or self.belowhorizon_hilbert_map:
                initial_scope = 'scope_belowhorizon'
            elif self.abovehorizon_normal_map or self.abovehorizon_hilbert_map:
                initial_scope = 'scope_abovehorizon'
            else:
                initial_scope = None

            with h5py.File(self.analysis_filename, 'r') as file:
                map_direction_keys = [key for key in file['map_direction'].keys()]#list(file['map_direction'].keys())

                #Check for map counterpart not assuming any premade scope.  This will be self.hilbert_map and self.normal_map
                
                if numpy.logical_and(self.map_dset_key in map_direction_keys,self.map_dset_key.replace('Hilb_%i'%(self.hilbert_map),'Hilb_%i'%(not self.hilbert_map)) in map_direction_keys):
                    if self.hilbert_map == True:
                        self.map_dset_key_hilbert = self.map_dset_key
                        self.map_dset_key = self.map_dset_key.replace('Hilb_%i'%(self.hilbert_map),'Hilb_%i'%(not self.hilbert_map))
                    else:
                        self.map_dset_key_hilbert = self.map_dset_key.replace('Hilb_%i'%(self.hilbert_map),'Hilb_%i'%(not self.hilbert_map))
                    self.hilbert_map = True #Since both datasets exist, both maps are enabled.                    
                    self.normal_map = True #Since both datasets exist, both maps are enabled.

                    added_keys = ['hilbert_theta_best_h','hilbert_theta_best_v','hilbert_theta_best_all','hilbert_elevation_best_h','hilbert_elevation_best_v','hilbert_elevation_best_all','hilbert_phi_best_h','hilbert_phi_best_v','hilbert_phi_best_all','hilbert_hpol_peak_to_sidelobe','hilbert_vpol_peak_to_sidelobe','hilbert_all_peak_to_sidelobe','hilbert_hpol_max_possible_map_value','hilbert_vpol_max_possible_map_value','hilbert_all_max_possible_map_value','hilbert_hpol_max_map_value','hilbert_vpol_max_map_value','hilbert_all_max_map_value','hilbert_map_max_time_delay_0subtract1_h','hilbert_map_max_time_delay_0subtract2_h','hilbert_map_max_time_delay_0subtract3_h','hilbert_map_max_time_delay_1subtract2_h','hilbert_map_max_time_delay_1subtract3_h','hilbert_map_max_time_delay_2subtract3_h','hilbert_map_max_time_delay_0subtract1_v','hilbert_map_max_time_delay_0subtract2_v','hilbert_map_max_time_delay_0subtract3_v','hilbert_map_max_time_delay_1subtract2_v','hilbert_map_max_time_delay_1subtract3_v','hilbert_map_max_time_delay_2subtract3_v','hilbert_map_max_time_delay_0subtract1_all','hilbert_map_max_time_delay_0subtract2_all','hilbert_map_max_time_delay_0subtract3_all','hilbert_map_max_time_delay_1subtract2_all','hilbert_map_max_time_delay_1subtract3_all','hilbert_map_max_time_delay_2subtract3_all']

                    for k in added_keys:
                        self.known_param_keys.append(k)

                    if verbose:
                        print('NOTE: Both version of map data (normal v.s. Hilbert) are available in this run for the given map dataset key.')
                        print('Use the same map keys prepended with "hilbert_" to use them.  For instance "hilbert_elevation_best_h"')


                if initial_hilbert is not None and initial_scope is not None:
                    #check for specific predefined horizon-based cut versions of the map
                    _map_dset_key = self.map_dset_key.replace(initial_hilbert, 'Hilb_0')

                    #Checking allsky
                    _map_dset_key = _map_dset_key.replace(initial_scope,'scope_allsky')
                    if _map_dset_key in map_direction_keys:
                        self.allsky_normal_map = True
                        self.map_dset_key_normal_allsky = copy.copy(_map_dset_key)
                        added_keys = ['theta_best_h_allsky','theta_best_v_allsky','theta_best_all_allsky','elevation_best_h_allsky','elevation_best_v_allsky','elevation_best_all_allsky','phi_best_h_allsky','phi_best_v_allsky','phi_best_all_allsky','hpol_peak_to_sidelobe_allsky','vpol_peak_to_sidelobe_allsky','all_peak_to_sidelobe_allsky','hpol_max_possible_map_value_allsky','vpol_max_possible_map_value_allsky','all_max_possible_map_value_allsky','hpol_max_map_value_allsky','vpol_max_map_value_allsky','all_max_map_value_allsky','map_max_time_delay_0subtract1_h_allsky','map_max_time_delay_0subtract2_h_allsky','map_max_time_delay_0subtract3_h_allsky','map_max_time_delay_1subtract2_h_allsky','map_max_time_delay_1subtract3_h_allsky','map_max_time_delay_2subtract3_h_allsky','map_max_time_delay_0subtract1_v_allsky','map_max_time_delay_0subtract2_v_allsky','map_max_time_delay_0subtract3_v_allsky','map_max_time_delay_1subtract2_v_allsky','map_max_time_delay_1subtract3_v_allsky','map_max_time_delay_2subtract3_v_allsky','map_max_time_delay_0subtract1_all_allsky','map_max_time_delay_0subtract2_all_allsky','map_max_time_delay_0subtract3_all_allsky','map_max_time_delay_1subtract2_all_allsky','map_max_time_delay_1subtract3_all_allsky','map_max_time_delay_2subtract3_all_allsky']
                        for k in added_keys:
                            self.known_param_keys.append(k)
                    
                    #Checking abovehorizon
                    _map_dset_key = _map_dset_key.replace('scope_allsky','scope_abovehorizon')
                    if _map_dset_key in map_direction_keys:
                        self.abovehorizon_normal_map = True
                        self.map_dset_key_normal_abovehorizon = copy.copy(_map_dset_key)
                        added_keys = ['theta_best_h_abovehorizon','theta_best_v_abovehorizon','theta_best_all_abovehorizon','elevation_best_h_abovehorizon','elevation_best_v_abovehorizon','elevation_best_all_abovehorizon','phi_best_h_abovehorizon','phi_best_v_abovehorizon','phi_best_all_abovehorizon','hpol_peak_to_sidelobe_abovehorizon','vpol_peak_to_sidelobe_abovehorizon','all_peak_to_sidelobe_abovehorizon','hpol_max_possible_map_value_abovehorizon','vpol_max_possible_map_value_abovehorizon','all_max_possible_map_value_abovehorizon','hpol_max_map_value_abovehorizon','vpol_max_map_value_abovehorizon','all_max_map_value_abovehorizon','map_max_time_delay_0subtract1_h_abovehorizon','map_max_time_delay_0subtract2_h_abovehorizon','map_max_time_delay_0subtract3_h_abovehorizon','map_max_time_delay_1subtract2_h_abovehorizon','map_max_time_delay_1subtract3_h_abovehorizon','map_max_time_delay_2subtract3_h_abovehorizon','map_max_time_delay_0subtract1_v_abovehorizon','map_max_time_delay_0subtract2_v_abovehorizon','map_max_time_delay_0subtract3_v_abovehorizon','map_max_time_delay_1subtract2_v_abovehorizon','map_max_time_delay_1subtract3_v_abovehorizon','map_max_time_delay_2subtract3_v_abovehorizon','map_max_time_delay_0subtract1_all_abovehorizon','map_max_time_delay_0subtract2_all_abovehorizon','map_max_time_delay_0subtract3_all_abovehorizon','map_max_time_delay_1subtract2_all_abovehorizon','map_max_time_delay_1subtract3_all_abovehorizon','map_max_time_delay_2subtract3_all_abovehorizon']
                        for k in added_keys:
                            self.known_param_keys.append(k)
                    
                    #Checking belowhorizon
                    _map_dset_key = _map_dset_key.replace('scope_abovehorizon','scope_belowhorizon')
                    if _map_dset_key in map_direction_keys:
                        self.belowhorizon_normal_map = True
                        self.map_dset_key_normal_belowhorizon = copy.copy(_map_dset_key)
                        added_keys = ['theta_best_h_belowhorizon','theta_best_v_belowhorizon','theta_best_all_belowhorizon','elevation_best_h_belowhorizon','elevation_best_v_belowhorizon','elevation_best_all_belowhorizon','phi_best_h_belowhorizon','phi_best_v_belowhorizon','phi_best_all_belowhorizon','hpol_peak_to_sidelobe_belowhorizon','vpol_peak_to_sidelobe_belowhorizon','all_peak_to_sidelobe_belowhorizon','hpol_max_possible_map_value_belowhorizon','vpol_max_possible_map_value_belowhorizon','all_max_possible_map_value_belowhorizon','hpol_max_map_value_belowhorizon','vpol_max_map_value_belowhorizon','all_max_map_value_belowhorizon','map_max_time_delay_0subtract1_h_belowhorizon','map_max_time_delay_0subtract2_h_belowhorizon','map_max_time_delay_0subtract3_h_belowhorizon','map_max_time_delay_1subtract2_h_belowhorizon','map_max_time_delay_1subtract3_h_belowhorizon','map_max_time_delay_2subtract3_h_belowhorizon','map_max_time_delay_0subtract1_v_belowhorizon','map_max_time_delay_0subtract2_v_belowhorizon','map_max_time_delay_0subtract3_v_belowhorizon','map_max_time_delay_1subtract2_v_belowhorizon','map_max_time_delay_1subtract3_v_belowhorizon','map_max_time_delay_2subtract3_v_belowhorizon','map_max_time_delay_0subtract1_all_belowhorizon','map_max_time_delay_0subtract2_all_belowhorizon','map_max_time_delay_0subtract3_all_belowhorizon','map_max_time_delay_1subtract2_all_belowhorizon','map_max_time_delay_1subtract3_all_belowhorizon','map_max_time_delay_2subtract3_all_belowhorizon']
                        for k in added_keys:
                            self.known_param_keys.append(k)

                    #Checking best from p2p calculation
                    _map_dset_key = _map_dset_key.replace('scope_belowhorizon','scope_best')
                    if _map_dset_key in map_direction_keys:
                        self.calculated_best_normal_map = True
                        self.map_dset_calculated_best = copy.copy(_map_dset_key)

                    #check for specific predefined horizon-based cut versions of the map
                    _map_dset_key = self.map_dset_key.replace(initial_hilbert, 'Hilb_1')

                    #Checking allsky
                    _map_dset_key = _map_dset_key.replace(initial_scope,'scope_allsky')
                    if _map_dset_key in map_direction_keys:
                        self.allsky_hilbert_map = True
                        self.map_dset_key_hilbert_allsky = copy.copy(_map_dset_key)

                        added_keys = ['hilbert_theta_best_h_allsky','hilbert_theta_best_v_allsky','hilbert_theta_best_all_allsky','hilbert_elevation_best_h_allsky','hilbert_elevation_best_v_allsky','hilbert_elevation_best_all_allsky','hilbert_phi_best_h_allsky','hilbert_phi_best_v_allsky','hilbert_phi_best_all_allsky','hilbert_hpol_peak_to_sidelobe_allsky','hilbert_vpol_peak_to_sidelobe_allsky','hilbert_all_peak_to_sidelobe_allsky','hilbert_hpol_max_possible_map_value_allsky','hilbert_vpol_max_possible_map_value_allsky','hilbert_all_max_possible_map_value_allsky','hilbert_hpol_max_map_value_allsky','hilbert_vpol_max_map_value_allsky','hilbert_all_max_map_value_allsky','hilbert_map_max_time_delay_0subtract1_h_allsky','hilbert_map_max_time_delay_0subtract2_h_allsky','hilbert_map_max_time_delay_0subtract3_h_allsky','hilbert_map_max_time_delay_1subtract2_h_allsky','hilbert_map_max_time_delay_1subtract3_h_allsky','hilbert_map_max_time_delay_2subtract3_h_allsky','hilbert_map_max_time_delay_0subtract1_v_allsky','hilbert_map_max_time_delay_0subtract2_v_allsky','hilbert_map_max_time_delay_0subtract3_v_allsky','hilbert_map_max_time_delay_1subtract2_v_allsky','hilbert_map_max_time_delay_1subtract3_v_allsky','hilbert_map_max_time_delay_2subtract3_v_allsky','hilbert_map_max_time_delay_0subtract1_all_allsky','hilbert_map_max_time_delay_0subtract2_all_allsky','hilbert_map_max_time_delay_0subtract3_all_allsky','hilbert_map_max_time_delay_1subtract2_all_allsky','hilbert_map_max_time_delay_1subtract3_all_allsky','hilbert_map_max_time_delay_2subtract3_all_allsky']
                        for k in added_keys:
                            self.known_param_keys.append(k)
                    
                    #Checking abovehorizon
                    _map_dset_key = _map_dset_key.replace('scope_allsky','scope_abovehorizon')
                    if _map_dset_key in map_direction_keys:
                        self.abovehorizon_hilbert_map = True
                        self.map_dset_key_hilbert_abovehorizon = copy.copy(_map_dset_key)
                        added_keys = ['hilbert_theta_best_h_abovehorizon','hilbert_theta_best_v_abovehorizon','hilbert_theta_best_all_abovehorizon','hilbert_elevation_best_h_abovehorizon','hilbert_elevation_best_v_abovehorizon','hilbert_elevation_best_all_abovehorizon','hilbert_phi_best_h_abovehorizon','hilbert_phi_best_v_abovehorizon','hilbert_phi_best_all_abovehorizon','hilbert_hpol_peak_to_sidelobe_abovehorizon','hilbert_vpol_peak_to_sidelobe_abovehorizon','hilbert_all_peak_to_sidelobe_abovehorizon','hilbert_hpol_max_possible_map_value_abovehorizon','hilbert_vpol_max_possible_map_value_abovehorizon','hilbert_all_max_possible_map_value_abovehorizon','hilbert_hpol_max_map_value_abovehorizon','hilbert_vpol_max_map_value_abovehorizon','hilbert_all_max_map_value_abovehorizon','hilbert_map_max_time_delay_0subtract1_h_abovehorizon','hilbert_map_max_time_delay_0subtract2_h_abovehorizon','hilbert_map_max_time_delay_0subtract3_h_abovehorizon','hilbert_map_max_time_delay_1subtract2_h_abovehorizon','hilbert_map_max_time_delay_1subtract3_h_abovehorizon','hilbert_map_max_time_delay_2subtract3_h_abovehorizon','hilbert_map_max_time_delay_0subtract1_v_abovehorizon','hilbert_map_max_time_delay_0subtract2_v_abovehorizon','hilbert_map_max_time_delay_0subtract3_v_abovehorizon','hilbert_map_max_time_delay_1subtract2_v_abovehorizon','hilbert_map_max_time_delay_1subtract3_v_abovehorizon','hilbert_map_max_time_delay_2subtract3_v_abovehorizon','hilbert_map_max_time_delay_0subtract1_all_abovehorizon','hilbert_map_max_time_delay_0subtract2_all_abovehorizon','hilbert_map_max_time_delay_0subtract3_all_abovehorizon','hilbert_map_max_time_delay_1subtract2_all_abovehorizon','hilbert_map_max_time_delay_1subtract3_all_abovehorizon','hilbert_map_max_time_delay_2subtract3_all_abovehorizon']
                        for k in added_keys:
                            self.known_param_keys.append(k)
                    
                    #Checking belowhorizon
                    _map_dset_key = _map_dset_key.replace('scope_abovehorizon','scope_belowhorizon')
                    if _map_dset_key in map_direction_keys:
                        self.belowhorizon_hilbert_map = True
                        self.map_dset_key_hilbert_below = copy.copy(_map_dset_key)
                        added_keys = ['hilbert_theta_best_h_belowhorizon','hilbert_theta_best_v_belowhorizon','hilbert_theta_best_all_belowhorizon','hilbert_elevation_best_h_belowhorizon','hilbert_elevation_best_v_belowhorizon','hilbert_elevation_best_all_belowhorizon','hilbert_phi_best_h_belowhorizon','hilbert_phi_best_v_belowhorizon','hilbert_phi_best_all_belowhorizon','hilbert_hpol_peak_to_sidelobe_belowhorizon','hilbert_vpol_peak_to_sidelobe_belowhorizon','hilbert_all_peak_to_sidelobe_belowhorizon','hilbert_hpol_max_possible_map_value_belowhorizon','hilbert_vpol_max_possible_map_value_belowhorizon','hilbert_all_max_possible_map_value_belowhorizon','hilbert_hpol_max_map_value_belowhorizon','hilbert_vpol_max_map_value_belowhorizon','hilbert_all_max_map_value_belowhorizon','hilbert_map_max_time_delay_0subtract1_h_belowhorizon','hilbert_map_max_time_delay_0subtract2_h_belowhorizon','hilbert_map_max_time_delay_0subtract3_h_belowhorizon','hilbert_map_max_time_delay_1subtract2_h_belowhorizon','hilbert_map_max_time_delay_1subtract3_h_belowhorizon','hilbert_map_max_time_delay_2subtract3_h_belowhorizon','hilbert_map_max_time_delay_0subtract1_v_belowhorizon','hilbert_map_max_time_delay_0subtract2_v_belowhorizon','hilbert_map_max_time_delay_0subtract3_v_belowhorizon','hilbert_map_max_time_delay_1subtract2_v_belowhorizon','hilbert_map_max_time_delay_1subtract3_v_belowhorizon','hilbert_map_max_time_delay_2subtract3_v_belowhorizon','hilbert_map_max_time_delay_0subtract1_all_belowhorizon','hilbert_map_max_time_delay_0subtract2_all_belowhorizon','hilbert_map_max_time_delay_0subtract3_all_belowhorizon','hilbert_map_max_time_delay_1subtract2_all_belowhorizon','hilbert_map_max_time_delay_1subtract3_all_belowhorizon','hilbert_map_max_time_delay_2subtract3_all_belowhorizon']
                        for k in added_keys:
                            self.known_param_keys.append(k)

                    #Checking best from p2p calculation # I don't really plan for this to be used at all with hilbert.
                    _map_dset_key = _map_dset_key.replace('scope_belowhorizon','scope_best')
                    if _map_dset_key in map_direction_keys:
                        self.calculated_best_hilbert_map = True
                        self.map_dset_calculated_best = copy.copy(_map_dset_key)


                if verbose:
                    print('Map Direction datasets in file:')
                    for d in map_direction_keys: 
                        print('\t' + d)


                file.close()
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def printDatasets(self):
        '''
        This will print the names of the first level datasets (not built to be recursive yet.)
        '''
        try:
            with h5py.File(self.analysis_filename, 'r') as file:
                print('Datasets in file:')
                print(list(file.keys()))
                print('Time Delay datasets in file:')
                for d in list(file['time_delays'].keys()): 
                    print('\t' + d)
                    for dd in list(file['time_delays'][d].keys()):
                        print('\t\t' + dd)
                print('Impulsivity datasets in file:')
                for d in list(file['impulsivity'].keys()): 
                    print('\t' + d)

                print('Map Direction datasets in file:')
                for d in list(file['map_direction'].keys()): 
                    print('\t' + d)


                file.close()
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def updateReader(self, reader, analysis_data_dir=None):
        '''
        This will let the user update the reader file if they are looping over multiple runs.
        '''
        try:
            self.reader = reader
            self.analysis_filename = createFile(reader,analysis_data_dir=analysis_data_dir) #Creates an analysis file if one does not exist.  Returns filename to load file.
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print('Run: ',self.reader.run)
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def printSampleROI(self,verbose=True):
        '''
        This will print and return an example ROI dictionary.  This is to provide examples of what a dictionary input
        to self.addROI might look like. 
        '''
        try:
            sample_ROI = {  'corr A':{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]},\
                            'high v imp':{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]},\
                            'small h.4 v.4 imp':{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]}}
            if verbose == True:
                print('Sample ROI dict:')
                print(sample_ROI)
            return sample_ROI
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addROI(self, roi_key, roi_dict):
        '''
        Let's you adjust the regions of interest to be plotted on the curves.  
        Each added roi should come as a key (label) and dictionary.  The dictionary should contain keys with the
        corresponding box cuts specified as lists of 4 coordinates (x1,x2,y1,y2) as the relate to that params specific
        plot.

        For example:

        roi_key = "roi1"
        roi_dict = {"impulsivity_hv":[roi1_h_lower, roi1_h_upper, roi1_v_lower, roi1_v_upper]}


        I am moving towards ROI selection for any set of parameters.  They will be looped over, where each key in an
        ROI must be in the know param list.  Then a window cut will be applied based on the upper and lower bounds of 
        that specific parameter.  This will be done for each of the listed parameters, and the resulting cut will be
        used as the events passing the ROI definition.  Below is an example of a 3 parameter ROI. 
        roi_dict = {    'param_a_string':[roi1_a_lower, roi1_a_upper],\
                        'param_b_string':[roi1_b_lower, roi1_b_upper],\
                        'param_c_string':[roi1_c_lower, roi1_c_upper]}
        '''
        try:
            passed = []
            for param_key in list(roi_dict.keys()):
                kw_cut = numpy.where([kw in param_key for kw in self.math_keywords])[0]

                if len(kw_cut) == 1:
                    math_keyword = self.math_keywords[kw_cut[0]]
                    if len(param_key.split(math_keyword)) == 2:
                        param_key_a = param_key.split(math_keyword)[0]
                        param_key_b = param_key.split(math_keyword)[1]
                        
                        if numpy.isin(param_key_a, self.known_param_keys) and numpy.isin(param_key_b, self.known_param_keys):
                            passed.append(True)
                        else:
                            passed.append(False)
                    else:
                        passed.append(False)
                elif numpy.isin(param_key, self.known_param_keys):
                    passed.append(True)
                else:
                    passed.append(False)

            if numpy.all(passed):
                self.roi[roi_key] = roi_dict
                self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, len(list(self.roi.keys())))]
            else:
                print('WARNING!!!')
                import pdb; pdb.set_trace()
                for key in list(roi_dict.keys())[~numpy.isin(list(roi_dict.keys()), self.known_param_keys)]:
                    print('The given roi parameter [%s] not in the list of known parameters\n%s:'%(key,str(self.known_param_keys)))
                return
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def resetAllROI(self):
        '''
        This will delete the ROI dict.
        '''
        try:
            self.roi = {}
            self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, len(list(self.roi.keys())))]

            #Could add more checks here to ensure roi_dict is good.
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getEventidsFromTriggerType(self, trigger_types=None):
        '''
        This will get the eventids that match one of the specified trigger types.  If trigger_types is None then
        this will load using the trigger types specified in self.trigger_types, this is the default behaviour. 
        '''
        try:
            with h5py.File(self.analysis_filename, 'r') as file:
                #Initial cut based on trigger type.
                if trigger_types is None:
                    eventids = numpy.where(numpy.isin(file['trigger_type'][...],self.trigger_types))[0]
                else:
                    eventids = numpy.where(numpy.isin(file['trigger_type'][...],trigger_types))[0]
                file.close()
            return eventids
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getModifiedDataFromParam(self, eventids, param_key, verbose=True):
        '''
        This will attempt to return param data similar to getDataFromParam, but will do so assuming that the given
        param_key contains one of the the known self.math_keywords.  It will then delineate the two parameters to be
        combined, perform the desired operation, and return the values.  This does not nest multiple operations yet.

        The parameters to be combined should be given such that the only additional text is a keyword.  For example
        std_hSLICERSUBTRACTstd_v will be interpreted as std_h - std_v.  So no additional hyphens, spaces, etc. should
        be included.  This is also case sensitive. 
        '''
        try:
            kw_cut = numpy.where([kw in param_key for kw in self.math_keywords])[0]
            #param_key_cut = numpy.where([key in param_key for key in self.known_param_keys])[0]
            if len(kw_cut) == 1:
                math_keyword = self.math_keywords[kw_cut[0]]
                if len(param_key.split(math_keyword)) == 2:
                    param_key_a = param_key.split(math_keyword)[0]
                    param_key_b = param_key.split(math_keyword)[1]
                    param_a = self.getDataFromParam(eventids, param_key_a)
                    param_b = self.getDataFromParam(eventids, param_key_b)
                    if math_keyword == 'SLICERADD':
                        param = param_a + param_b
                    elif math_keyword == 'SLICERSUBTRACT':
                        param = param_a - param_b
                    elif math_keyword == 'SLICERDIVIDE':
                        param = numpy.divide(param_a, param_b)
                    elif math_keyword == 'SLICERMULTIPLY':
                        param = numpy.multiply(param_a, param_b)
                    elif math_keyword == 'SLICERMAX':
                        param = numpy.max(numpy.vstack((param_a, param_b)),axis=0)
                    elif math_keyword == 'SLICERMIN':
                        param = numpy.min(numpy.vstack((param_a, param_b)),axis=0)
                    elif math_keyword == 'SLICERMEAN':
                        param = numpy.mean(numpy.vstack((param_a, param_b)),axis=0)
                    else:
                        print('WARNING: GIVEN param_key IS NOT ACCOUNTED FOR IN getModifiedDataFromParam, RETURNING EMPTY ARRAY')
                        param = numpy.array([])
                else:
                    print('WARNING: MATH KEYWORD GIVEN IN getModifiedDataFromParam BUT INSUFFICIENT NORMAL param_key VARIABLES ARE IN THE STRING AS WELL, RETURNING EMPTY ARRAY')
                    param = numpy.array([])

            elif len(param_key_cut) == 1:
                if verbose:
                    print('getModifiedDataFromParam called with no math_keyword and only one param_key.  Returning result from getDataFromParam.')
                param = self.getDataFromParam(eventids, param_key)
            else:
                if verbose:
                    print('Error in getModifiedDataFromParam, no known math_keyword is in the given param_key, and this function should not have been called.')
                param = numpy.array([])
            return param

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getDataFromParam(self, eventids, param_key):
        '''
        Given eventids, this will load and return array for parameters associated with string param.
        '''
        try:
            if len(numpy.shape(eventids)) > 1:
                print('WARNING!!! eventids is in the incorrect format.')
            
            if numpy.any(numpy.diff(eventids) < 1):
                print('WARNING!!! eventids is not sorted or has duplicate entries.  May mess up output of getDataFromParam.')

            len_eventids = len(eventids)
            if len(eventids) > 0:
                if len(numpy.where([kw in param_key for kw in self.math_keywords])[0]) == 1:
                    param = self.getModifiedDataFromParam(eventids, param_key, verbose=False)
                elif param_key in self.known_param_keys:
                    with h5py.File(self.analysis_filename, 'r') as file:
                        if param_key == 'impulsivity_h':
                            if len_eventids < 500:
                                param = file['impulsivity'][self.impulsivity_dset_key]['hpol'][eventids]
                            else:
                                param = file['impulsivity'][self.impulsivity_dset_key]['hpol'][...][eventids]
                        elif param_key == 'impulsivity_v':
                            if len_eventids < 500:
                                param = file['impulsivity'][self.impulsivity_dset_key]['vpol'][eventids]
                            else:
                                param = file['impulsivity'][self.impulsivity_dset_key]['vpol'][...][eventids]
                        elif param_key == 'cr_template_search_h':
                            this_dset = 'bi-delta-curve-choice-%i'%self.cr_template_curve_choice
                            if len_eventids < 500:
                                output_correlation_values = file['cr_template_search'][this_dset][eventids]
                            else:
                                output_correlation_values = file['cr_template_search'][this_dset][...][eventids]
                            param = numpy.max(output_correlation_values[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD 
                        elif param_key == 'cr_template_search_v':
                            this_dset = 'bi-delta-curve-choice-%i'%self.cr_template_curve_choice
                            if len_eventids < 500:
                                output_correlation_values = file['cr_template_search'][this_dset][eventids]
                            else:
                                output_correlation_values = file['cr_template_search'][this_dset][...][eventids]
                            param = numpy.max(output_correlation_values[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                        elif param_key == 'std_h':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param = numpy.mean(std[:,self.included_hpol_antennas],axis=1) 
                        elif param_key == 'std_v':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param = numpy.mean(std[:,self.included_vpol_antennas],axis=1) 
                        elif param_key == 'p2p_h': 
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param = numpy.max(p2p[:,self.included_hpol_antennas],axis=1) 
                        elif param_key == 'p2p_v': 
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param = numpy.max(p2p[:,self.included_vpol_antennas],axis=1) 
                        elif param_key == 'snr_h':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param_1 = numpy.mean(std[:,self.included_hpol_antennas],axis=1) 
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param_2 = numpy.max(p2p[:,self.included_hpol_antennas],axis=1) 
                            param = numpy.divide(param_2, param_1)
                        elif param_key == 'min_snr_h':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param_1 = std[:,self.included_hpol_antennas]
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param_2 = p2p[:,self.included_hpol_antennas]
                            param = numpy.min(numpy.divide(param_2, param_1),axis=1)
                        elif param_key == 'snr_gap_h':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param_1 = std[:,self.included_hpol_antennas]
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param_2 = p2p[:,self.included_hpol_antennas]
                            param = numpy.max(numpy.divide(param_2, param_1),axis=1) - numpy.min(numpy.divide(param_2, param_1),axis=1)
                        elif param_key == 'snr_v':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param_1 = numpy.mean(std[:,self.included_vpol_antennas],axis=1) 
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param_2 = numpy.max(p2p[:,self.included_vpol_antennas],axis=1) 
                            param = numpy.divide(param_2, param_1)
                        elif param_key == 'min_snr_v':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param_1 = std[:,self.included_vpol_antennas]
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param_2 = p2p[:,self.included_vpol_antennas]
                            param = numpy.min(numpy.divide(param_2, param_1),axis=1)
                        elif param_key == 'snr_gap_v':
                            if len_eventids < 500:
                                std = file['std'][eventids]
                            else:
                                std = file['std'][...][eventids]
                            param_1 = std[:,self.included_vpol_antennas]
                            if len_eventids < 500:
                                p2p = file['p2p'][eventids]
                            else:
                                p2p = file['p2p'][...][eventids]
                            param_2 = p2p[:,self.included_vpol_antennas]
                            param = numpy.max(numpy.divide(param_2, param_1),axis=1) - numpy.min(numpy.divide(param_2, param_1),axis=1)
                        elif 'time_delay_' in param_key and 'map' not in param_key:
                            split_param_key = param_key.split('_')
                            dset = '%spol_t_%ssubtract%s'%(split_param_key[3],split_param_key[2].split('subtract')[0],split_param_key[2].split('subtract')[1]) #Rewriting internal key name to time delay formatting.
                            if len_eventids < 500:
                                param = file['time_delays'][self.time_delays_dset_key][dset][eventids]
                            else:
                                param = file['time_delays'][self.time_delays_dset_key][dset][...][eventids]
                        elif 'max_corr_' in param_key and 'map' not in param_key:
                            if param_key == 'mean_max_corr_h':
                                param = numpy.mean(numpy.vstack((self.getDataFromParam(eventids,'max_corr_0subtract1_h'),self.getDataFromParam(eventids,'max_corr_0subtract2_h'),self.getDataFromParam(eventids,'max_corr_0subtract3_h'),self.getDataFromParam(eventids,'max_corr_1subtract2_h'),self.getDataFromParam(eventids,'max_corr_1subtract3_h'),self.getDataFromParam(eventids,'max_corr_2subtract3_h'))),axis=0)
                            elif param_key == 'max_max_corr_h':
                                param = numpy.max(numpy.vstack((self.getDataFromParam(eventids,'max_corr_0subtract1_h'),self.getDataFromParam(eventids,'max_corr_0subtract2_h'),self.getDataFromParam(eventids,'max_corr_0subtract3_h'),self.getDataFromParam(eventids,'max_corr_1subtract2_h'),self.getDataFromParam(eventids,'max_corr_1subtract3_h'),self.getDataFromParam(eventids,'max_corr_2subtract3_h'))),axis=0)
                            elif param_key == 'mean_max_corr_v':
                                param = numpy.mean(numpy.vstack((self.getDataFromParam(eventids,'max_corr_0subtract1_v'),self.getDataFromParam(eventids,'max_corr_0subtract2_v'),self.getDataFromParam(eventids,'max_corr_0subtract3_v'),self.getDataFromParam(eventids,'max_corr_1subtract2_v'),self.getDataFromParam(eventids,'max_corr_1subtract3_v'),self.getDataFromParam(eventids,'max_corr_2subtract3_v'))),axis=0)
                            elif param_key == 'max_max_corr_v':
                                param = numpy.max(numpy.vstack((self.getDataFromParam(eventids,'max_corr_0subtract1_v'),self.getDataFromParam(eventids,'max_corr_0subtract2_v'),self.getDataFromParam(eventids,'max_corr_0subtract3_v'),self.getDataFromParam(eventids,'max_corr_1subtract2_v'),self.getDataFromParam(eventids,'max_corr_1subtract3_v'),self.getDataFromParam(eventids,'max_corr_2subtract3_v'))),axis=0)
                            else:
                                split_param_key = param_key.split('_')
                                dset = '%spol_max_corr_%ssubtract%s'%(split_param_key[3],split_param_key[2].split('subtract')[0],split_param_key[2].split('subtract')[1]) #Rewriting internal key name to time delay formatting.
                                if len_eventids < 500:
                                    param = file['time_delays'][self.time_delays_dset_key][dset][eventids]
                                else:
                                    param = file['time_delays'][self.time_delays_dset_key][dset][...][eventids]

                        elif 'cw_present' == param_key:
                            if len_eventids < 500:
                                param = file['cw']['has_cw'][eventids].astype(int)
                            else:
                                param = file['cw']['has_cw'][...][eventids].astype(int)
                        elif 'cw_freq_Mhz' == param_key:
                            if len_eventids < 500:
                                param = file['cw']['freq_hz'][eventids]/1e6 #MHz
                            else:
                                param = file['cw']['freq_hz'][...][eventids]/1e6 #MHz
                        elif 'cw_linear_magnitude' == param_key:
                            if len_eventids < 500:
                                param = file['cw']['linear_magnitude'][eventids]
                            else:
                                param = file['cw']['linear_magnitude'][...][eventids]
                        elif 'cw_dbish' == param_key:
                            cw_dsets = list(file['cw'].keys())
                            if not numpy.isin('dbish',cw_dsets):
                                print('No stored dbish data from cw dataset, attempting to calculate from linear magnitude.')
                                if not hasattr(self, 'cw_prep'):
                                    print('Creating FFTPrepper class to prepare CW bins.')
                                    self.cw_prep = FFTPrepper(self.reader, final_corr_length=int(file['cw'].attrs['final_corr_length']), crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), waveform_index_range=(None,None), plot_filters=False)
                                    self.cw_prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)

                                if len_eventids < 500:
                                    linear_magnitude = file['cw']['linear_magnitude'][eventids]
                                else:
                                    linear_magnitude = file['cw']['linear_magnitude'][...][eventids]
                                param = 10.0*numpy.log10( linear_magnitude**2 / len(self.cw_prep.t()) )
                            else:
                                # if len_eventids < 500:
                                    #param = file['cw']['dbish'][eventids] #Should work, but I messed it up.  Bodging it for now.
                                # else:
                                    #param = file['cw']['dbish'][...][eventids] #Should work, but I messed it up.  Bodging it for now.
                                #print('dbish not correctly setup in flag_cw.py.  Converting linear to dbish now.')
                                if len_eventids < 500:
                                    linear_magnitude = file['cw']['linear_magnitude'][eventids]
                                else:
                                    linear_magnitude = file['cw']['linear_magnitude'][...][eventids]
                                param = 10.0*numpy.log10( linear_magnitude**2 / len(self.cw_prep.t()) )
                        
                        elif param_key == 'theta_best_choice':
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_calculated_best]['best_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_calculated_best]['best_ENU_zenith'][...][eventids]
                        elif param_key == 'elevation_best_choice':
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_calculated_best]['best_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_calculated_best]['best_ENU_zenith'][...][eventids]
                            param = 90.0 - param
                        elif param_key == 'phi_best_choice':
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_calculated_best]['best_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_calculated_best]['best_ENU_azimuth'][...][eventids]

                        elif 'hilbert' not in param_key and 'theta_best_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_allsky]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_allsky]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_allsky]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_allsky]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_allsky]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_allsky]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_allsky]['all_ENU_azimuth'][...][eventids]

                        elif 'hilbert' not in param_key and 'theta_best_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_abovehorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_abovehorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_abovehorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_abovehorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_abovehorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_abovehorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_abovehorizon]['all_ENU_azimuth'][...][eventids]

                        elif 'hilbert' not in param_key and 'theta_best_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_belowhorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_belowhorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_belowhorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_belowhorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_belowhorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_normal_belowhorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_normal_belowhorizon]['all_ENU_azimuth'][...][eventids]

                        #Must be last in it's category so more specific scope calls are hit first in elif case
                        elif 'hilbert' not in param_key and 'theta_best_h' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_v' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'theta_best_all' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_h' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_v' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'elevation_best_all' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_h' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_v' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' not in param_key and 'phi_best_all' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key]['all_ENU_azimuth'][...][eventids]

                        elif 'hilbert' in param_key and 'theta_best_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_allsky]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_allsky]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_allsky]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_allsky]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_allsky]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_allsky]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_allsky]['all_ENU_azimuth'][...][eventids]

                        elif 'hilbert' in param_key and 'theta_best_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_abovehorizon]['all_ENU_azimuth'][...][eventids]

                        elif 'hilbert' in param_key and 'theta_best_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert_belowhorizon]['all_ENU_azimuth'][...][eventids]

                        #Must be last in it's category so more specific scope calls are hit first in elif case
                        elif 'hilbert' in param_key and 'theta_best_h' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert]['hpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_v' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert]['vpol_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'theta_best_all' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert]['all_ENU_zenith'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_h' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert]['hpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert]['hpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_v' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert]['vpol_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert]['vpol_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'elevation_best_all' in param_key:
                            if len_eventids < 500:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert]['all_ENU_zenith'][eventids]
                            else:
                                param = 90.0 - file['map_direction'][self.map_dset_key_hilbert]['all_ENU_zenith'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_h' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert]['hpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert]['hpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_v' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert]['vpol_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert]['vpol_ENU_azimuth'][...][eventids]
                        elif 'hilbert' in param_key and 'phi_best_all' in param_key:
                            if len_eventids < 500:
                                param = file['map_direction'][self.map_dset_key_hilbert]['all_ENU_azimuth'][eventids]
                            else:
                                param = file['map_direction'][self.map_dset_key_hilbert]['all_ENU_azimuth'][...][eventids]
                        elif 'calibrated_trigtime' == param_key:
                            if len_eventids < 500:
                                param = file['calibrated_trigtime'][eventids]
                            else:
                                param = file['calibrated_trigtime'][...][eventids]
                        elif 'triggered_beams' == param_key:
                            param = self.reader.returnTriggerInfo()[0][eventids]
                        elif 'beam_power' == param_key:
                            param = self.reader.returnTriggerInfo()[1][eventids]



                        elif 'coincidence_method_1_h' == param_key:
                            if len_eventids < 500:
                                param = file['coincidence_count']['method_1']['hpol'][eventids]
                            else:
                                param = file['coincidence_count']['method_1']['hpol'][...][eventids]
                        elif 'coincidence_method_1_v' == param_key:
                            if len_eventids < 500:
                                param = file['coincidence_count']['method_1']['vpol'][eventids]
                            else:
                                param = file['coincidence_count']['method_1']['vpol'][...][eventids]
                        elif 'coincidence_method_2_h' == param_key:
                            if len_eventids < 500:
                                param = numpy.mean(file['coincidence_count']['method_2'][eventids,0:8:2],axis=1)
                            else:
                                param = numpy.mean(file['coincidence_count']['method_2'][...][eventids,0:8:2],axis=1)

                        elif 'coincidence_method_2_v' == param_key:
                            if len_eventids < 500:
                                param = numpy.mean(file['coincidence_count']['method_2'][eventids,1:8:2],axis=1)
                            else:
                                param = numpy.mean(file['coincidence_count']['method_2'][...][eventids,1:8:2],axis=1)


                        elif 'hilbert' not in param_key and 'hpol_peak_to_sidelobe_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_peak_to_sidelobe_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_peak_to_sidelobe_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_peak_to_sidelobe_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_peak_to_sidelobe_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_peak_to_sidelobe_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_peak_to_sidelobe' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['hpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_peak_to_sidelobe' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['hpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['hpol_peak_to_sidelobe'][...][eventids]

                        elif 'hilbert' not in param_key and 'vpol_peak_to_sidelobe_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_peak_to_sidelobe_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_peak_to_sidelobe_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_peak_to_sidelobe_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_peak_to_sidelobe_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_peak_to_sidelobe_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_peak_to_sidelobe' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['vpol_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_peak_to_sidelobe' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['vpol_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['vpol_peak_to_sidelobe'][...][eventids]


                        elif 'hilbert' not in param_key and 'all_peak_to_sidelobe_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_peak_to_sidelobe_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_peak_to_sidelobe_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'all_peak_to_sidelobe_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'all_peak_to_sidelobe_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'all_peak_to_sidelobe_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_peak_to_sidelobe' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['all_peak_to_sidelobe'][...][eventids]
                        elif 'hilbert' in param_key and 'all_peak_to_sidelobe' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['all_peak_to_sidelobe'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['all_peak_to_sidelobe'][...][eventids]




                        elif 'hilbert' not in param_key and 'hpol_max_possible_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_max_possible_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_max_possible_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_possible_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_possible_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_possible_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_max_possible_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['hpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_possible_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['hpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['hpol_max_possible_map_value'][...][eventids]

                        elif 'hilbert' not in param_key and 'hpol_max_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_max_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_max_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'hpol_max_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['hpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'hpol_max_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['hpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['hpol_max_map_value'][...][eventids]



                        elif 'hilbert' not in param_key and 'vpol_max_possible_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_max_possible_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_max_possible_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_possible_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_possible_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_possible_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_max_possible_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['vpol_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_possible_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['vpol_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['vpol_max_possible_map_value'][...][eventids]

                        elif 'hilbert' not in param_key and 'vpol_max_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_max_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_max_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'vpol_max_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['vpol_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'vpol_max_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['vpol_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['vpol_max_map_value'][...][eventids]



                        elif 'hilbert' not in param_key and 'all_max_possible_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_max_possible_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_max_possible_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_possible_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_possible_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_possible_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_max_possible_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['all_max_possible_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_possible_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['all_max_possible_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['all_max_possible_map_value'][...][eventids]

                        elif 'hilbert' not in param_key and 'all_max_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_allsky]['all_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_max_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_abovehorizon]['all_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_max_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_normal_belowhorizon]['all_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_map_value_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_allsky]['all_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_map_value_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_abovehorizon]['all_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_map_value_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert_belowhorizon]['all_max_map_value'][...][eventids]
                        elif 'hilbert' not in param_key and 'all_max_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key]['all_max_map_value'][...][eventids]
                        elif 'hilbert' in param_key and 'all_max_map_value' in param_key:
                            #Must be last in it's category so more specific scope calls are hit first in elif case
                            if len_eventids < 500:
                                param = file['map_properties'][self.map_dset_key_hilbert]['all_max_map_value'][eventids]
                            else:
                                param = file['map_properties'][self.map_dset_key_hilbert]['all_max_map_value'][...][eventids]




                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_0subtract1'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_0subtract1'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_0subtract1'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_0subtract1'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_0subtract1'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_0subtract1'][...][eventids]


                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_0subtract1'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_0subtract1'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract1_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_0subtract1'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract1_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_0subtract1'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_0subtract1'][...][eventids]




                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_0subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_0subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_0subtract2'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_0subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_0subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_0subtract2'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_0subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_0subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract2_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_0subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract2_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_0subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_0subtract2'][...][eventids]



                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_0subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_0subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_0subtract3'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_0subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_0subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_0subtract3'][...][eventids]



                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_0subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_0subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_0subtract3_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_0subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_0subtract3_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_0subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_0subtract3'][...][eventids]



                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_1subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_1subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_1subtract2'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_1subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_1subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_1subtract2'][...][eventids]


                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_1subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_1subtract2'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract2_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_1subtract2'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract2_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_1subtract2'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_1subtract2'][...][eventids]



                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_1subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_1subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_1subtract3'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_1subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_1subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_1subtract3'][...][eventids]


                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_1subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_1subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_1subtract3_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_1subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_1subtract3_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_1subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_1subtract3'][...][eventids]



                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['hpol_2subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['hpol_2subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['hpol_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_h_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['hpol_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_h_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['hpol_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_h_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['hpol_2subtract3'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['vpol_2subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['vpol_2subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['vpol_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_v_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['vpol_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_v_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['vpol_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_v_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['vpol_2subtract3'][...][eventids]

                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_allsky]['all_2subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_abovehorizon]['all_2subtract3'][...][eventids]
                        elif 'hilbert' not in param_key and 'map_max_time_delay_2subtract3_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_normal_belowhorizon]['all_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_all_allsky' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_allsky]['all_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_all_abovehorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_abovehorizon]['all_2subtract3'][...][eventids]
                        elif 'hilbert' in param_key and 'map_max_time_delay_2subtract3_all_belowhorizon' in param_key:
                            if len_eventids < 500:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_2subtract3'][eventids]
                            else:
                                param = file['map_times'][self.map_dset_key_hilbert_belowhorizon]['all_2subtract3'][...][eventids]

                        elif 'event_rate_ts_' in param_key:
                            rate_string, time_window_string = param_key.replace('event_rate_ts_','').split('_')
                            if len_eventids < 500:
                                param = file['event_rate_testing'][rate_string][time_window_string][eventids]
                            else:
                                param = file['event_rate_testing'][rate_string][time_window_string][...][eventids]
                        elif 'event_rate_sigma_' in param_key:
                            rate_string, time_window_string = param_key.replace('event_rate_sigma_','').split('_')
                            mean = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_mean']
                            sigma = file['event_rate_testing'][rate_string][time_window_string].attrs['random_fit_sigma']
                            
                            if len_eventids < 500:
                                param = (file['event_rate_testing'][rate_string][time_window_string][eventids] - mean)/sigma
                            else:
                                param = (file['event_rate_testing'][rate_string][time_window_string][...][eventids] - mean)/sigma
                        elif 'similarity_count_' in param_key:
                            if len_eventids < 500:
                                param = file['similarity_count'][self.time_delays_dset_key]['%spol_count'%(param_key.split('_')[-1])][eventids]
                            else:
                                param = file['similarity_count'][self.time_delays_dset_key]['%spol_count'%(param_key.split('_')[-1])][...][eventids]
                        elif 'similarity_fraction_' in param_key:
                            if len_eventids < 500:
                                param = file['similarity_count'][self.time_delays_dset_key]['%spol_fraction'%(param_key.split('_')[-1])][eventids]
                            else:
                                param = file['similarity_count'][self.time_delays_dset_key]['%spol_fraction'%(param_key.split('_')[-1])][...][eventids]


                        elif param_key == 'sun_az':
                            param = getSunAzEl(self.getDataFromParam(eventids, 'calibrated_trigtime'), interp=True, interp_step_s=5*60)[0]

                        elif param_key == 'sun_el':
                            param = getSunAzEl(self.getDataFromParam(eventids, 'calibrated_trigtime'), interp=True, interp_step_s=5*60)[1]                            

                        file.close()
                else:
                    print('\nWARNING!!!\nOther parameters have not been accounted for yet.\n%s'%(param_key))
                return param
            else:
                return numpy.array([])
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def getCutsFromROI(self,roi_key,eventids=None,load=False,save=False, return_successive_cut_counts=False, return_total_cut_counts=False):
        '''
        This will determine the eventids that match the cuts listed in the dictionary corresponding to roi_key, and will
        return them.  

        Parameters
        ----------
        roi_key : str
            The key of the ROI you want to calculate.
        load : bool
            If True then this will bypass calculation of the cut, and will instead attempt to load it from the file.  If
            this is done succesfully then save is set to False if True, and the eventids are returned.  If a cut matching
            the param is not present in the file then it will instead calculate the cut, and save if save == True.
        save : bool
            If True then the resulting ROI cut will be stored to the analysis file in the ROI group.  Note that any ROI
            saved with the same name will be automatically overwritten.
        '''
        try:
            if return_successive_cut_counts == True:
                successive_cut_counts = OrderedDict()
            if return_total_cut_counts == True:
                total_cut_counts = OrderedDict()

            if load == True:
                print('WARNING LOAD CUT IS DEPRECATED')
                with h5py.File(self.analysis_filename, 'r') as file:
                    dsets = list(file.keys()) #Existing datasets
                    if not numpy.isin('ROI',dsets):
                        load_failed = True
                    else:
                        ROI_dsets = list(file['cr_template_search'].keys())
                        if numpy.isin(roi_key,ROI_dsets):
                            save = False
                            eventids = numpy.where(file['ROI'][roi_key][...])[0] #List of eventids. 
                            load_failed = False
                        else:
                            load_failed = True
                    file.close()
            else:
                load_failed = True

            if load_failed == True:
                #Really this is the default behaviour
                if roi_key in list(self.roi.keys()):
                    if eventids is None:
                        try:
                            eventids = numpy.asarray(self.getEventidsFromTriggerType())
                        except Exception as e:
                            import pdb; pdb.set_trace()

                    if return_successive_cut_counts == True:
                        successive_cut_counts['initial'] = len(eventids)
                    if return_total_cut_counts == True:
                        total_cut_counts['initial'] = len(eventids)
                        master_cut = numpy.ones_like(eventids,dtype=bool)

                    if len(eventids) > 0:
                        for param_key in list(self.roi[roi_key].keys()):
                            param = self.getDataFromParam(eventids, param_key)

                            cut = numpy.logical_and(param >= self.roi[roi_key][param_key][0], param < self.roi[roi_key][param_key][1])

                            if return_total_cut_counts == True:
                                #Can't do shortening of events because I want the stats at every level
                                master_cut = numpy.logical_and(master_cut,cut)
                                total_cut_counts[param_key] = sum(cut)
                                if return_successive_cut_counts == True:
                                    successive_cut_counts[param_key] = sum(master_cut)
                            else:
                                #Can't shorten eventids each time.
                                #Reduce eventids by box cut
                                eventids = eventids[cut]  #Should get smaller/faster with every param cut.
                                if return_successive_cut_counts == True:
                                    successive_cut_counts[param_key] = len(eventids)
                        if return_total_cut_counts == True:
                            eventids = eventids[master_cut]
                else:
                    print('WARNING!!!')
                    print('Requested ROI [%s] is not specified in self.roi list:\n%s'%(roi_key,str(list(self.roi.keys()))))

            if save == True:
                '''
                Here I not only want to save bool data for events in the cut, but I also want to store the meta information
                required to interpret the cut.  So make first check to make sure that a group is made for ROI.  Then a dataset
                within this group for each ROI containing the bool data.  That dataset then also has associated attrs dictionary
                containing information such as allowed trigger types, allowed antennas, and the ROI dictionary that defines
                the n-d box cut. I have not done this specific thing before so hopefully it isn't terrible.  
                '''
                with h5py.File(self.analysis_filename, 'a') as file:
                    try:
                        dsets = list(file.keys()) #Existing datasets

                        if not numpy.isin('ROI',dsets):
                            file.create_group('ROI')

                        ROI_dsets = list(file['ROI'].keys())
                        
                        if numpy.isin(roi_key,ROI_dsets):
                            print('ROI["%s"] group already exists in file %s, it will be deleted and remade.'%(roi_key,self.analysis_filename))
                            del file['ROI'][roi_key]
                        
                        file['ROI'].create_dataset(roi_key, (file.attrs['N'],), dtype=bool, compression='gzip', compression_opts=4, shuffle=True)
                        
                        #Store all required information needed to reproduce cut here as attributes.
                        file['ROI'][roi_key].attrs['dict'] = str(self.roi[roi_key]) #Dict must be stored as a str.  To interpret it use ast.literal_eval(the_string)
                        file['ROI'][roi_key].attrs['included_antennas'] = self.included_antennas
                        file['ROI'][roi_key].attrs['cr_template_curve_choice'] = self.cr_template_curve_choice
                        file['ROI'][roi_key].attrs['trigger_types'] = self.trigger_types

                        #Store dataset.
                        file['ROI'][roi_key][...] = numpy.isin(numpy.arange(file.attrs['N']),eventids) #True for events in the cut list.  
                    except Exception as e:
                        print('\nError in %s'%inspect.stack()[0][3])
                        print('Run: ',self.reader.run)
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                    file.close() #want file to close where exception met or not.  
                print('Saving the cuts to the analysis file.')

            if return_successive_cut_counts and return_total_cut_counts:
                return eventids, successive_cut_counts, total_cut_counts
            elif return_successive_cut_counts:
                return eventids, successive_cut_counts
            elif return_total_cut_counts:
                return eventids, total_cut_counts
            else:
                return eventids
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def get2dHistCounts(self, main_param_key_x, main_param_key_y, eventids, set_bins=True, load=False,mask_top_N_bins=0,fill_value=0):
        '''
        Given the set of requested eventids, this will calculate the counts for the plot corresponding to 
        main_param_key, and will return the counts matrix. 

        Parameters
        ----------
        main_param_key_x : str
            The parameter for which you want the calculation to be performed (x-axis binning).
        main_param_key_y : str
            The parameter for which you want the calculation to be performed (y-axis binning).
        eventids : numpy.ndarray of ints
            The eventids you want the calculation performed/loaded for.  This is used to loop over events, and thus
            should match up with self.reader (i.e. should not have eventids higher than are in that run).
        load : bool
            If load == True then this will attempt to load the bin data from the analysis file.  Otherwise (or if load fails)
            this will calculate the data and return the counts. 
        set_bins : bool
            If True this will set the current loaded bins to the set appropriate for this data.  This should only be
            called the first time this data is accessed, otherwise unecessary computation is performed.  The common
            occurance of this is calling it when plotting a 2d histogram, but NOT when plotting contours overtop of
            said histogram. 
        '''
        try:
            if load == True:
                try:
                    bin_indices_raw_x, bin_edges_x, label_x = self.loadHistogramData(main_param_key_x,eventids)
                    bin_indices_x = self.cleanHistogramData(bin_indices_raw_x, bin_edges_x)

                    bin_indices_raw_y, bin_edges_y, label_y = self.loadHistogramData(main_param_key_y,eventids)
                    bin_indices_y = self.cleanHistogramData(bin_indices_raw_y, bin_edges_y)

                    if set_bins == True:
                        print('Setting current bin edges to loaded bins.')
                        self.current_bin_edges_x = bin_edges_x
                        self.current_label_x = label_x
                        self.current_bin_edges_y = bin_edges_y
                        self.current_label_y = label_y
                        self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y = numpy.meshgrid((self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2, (self.current_bin_edges_y[:-1] + self.current_bin_edges_y[1:]) / 2) # Use for contours
                        self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y = numpy.meshgrid(self.current_bin_edges_x , self.current_bin_edges_y)#use for pcolormesh

                    counts = numpy.zeros(self.current_bin_centers_mesh_x.shape, dtype=int)
                    #Add counts based on stored indices.
                    numpy.add.at(counts,(bin_indices_y,bin_indices_x),1) #Need to double check this works as I expect, and is not transposed.  
                except Exception as e:
                    load = False
                    print('\nError in %s'%inspect.stack()[0][3])
                    print('Run: ',self.reader.run)
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)

            if load == False:
                if set_bins == True:
                    self.setCurrentPlotBins(main_param_key_x,main_param_key_y,eventids)
                print('\tLoading data for %s'%main_param_key_x)
                param_x = self.getDataFromParam(eventids, main_param_key_x)
                print('\tLoading data for %s'%main_param_key_y)
                param_y = self.getDataFromParam(eventids, main_param_key_y)
                print('\tGetting counts from 2dhist')

                counts = numpy.histogram2d(param_x, param_y, bins = [self.current_bin_edges_x,self.current_bin_edges_y])[0].T #Outside of file being open 
            if mask_top_N_bins > 0:
                counts = self.returnMaskedArray(counts,mask_top_N_bins,fill_value=fill_value)    
            return counts

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getSingleParamPlotBins(self, param_key, eventids, verbose=False):
        '''
        This will determine the current bins to be used for param_key based on the internally defined parameters and
        measured quantitites.  This will be called by setCurrentPlotBins for each param key, but can also be called for
        an individual parameter key when attempting to save bin data.
        
        Parameters
        ----------
        param_key : str
            The parameter you want the bins calculated for.
        eventids : numpy.ndarray of ints
            The eventids you want to be included for any calculations of bin bounds.

        Returns
        -------
        current_bin_edges : numpy.ndarray of floats
            The edge of each bin for the chosen param key.
        label : str
            The x/y label for plotting. 
        '''
        try:
            calculate_bins_from_min_max = True #By default will be calculated at bottom of conditional list, unless a specific condition overrides.
            
            if len(eventids) == 0:
                if verbose:
                    print('No eventids given to getSingleParamPlotBins, using all eventids corresponding to selected trigger types.')
                eventids = self.getEventidsFromTriggerType()

            if len(numpy.where([kw in param_key for kw in self.math_keywords])[0]) == 1:
                if verbose:
                    print('\tPreparing to get counts for %s'%param_key)
                kw_cut = numpy.where([kw in param_key for kw in self.math_keywords])[0]

                math_keyword = self.math_keywords[kw_cut[0]]
                if len(param_key.split(math_keyword)) == 2:
                    param_key_a = param_key.split(math_keyword)[0]
                    param_key_b = param_key.split(math_keyword)[1]
                    
                    current_bin_edges_a, label_a = self.getSingleParamPlotBins(param_key_a, eventids, verbose=False)
                    current_bin_edges_b, label_b = self.getSingleParamPlotBins(param_key_b, eventids, verbose=False)
                    
                    if math_keyword == 'SLICERADD':
                        label = '%s\n%s\n%s'%(label_a, ' + ' , label_b)
                    elif math_keyword == 'SLICERSUBTRACT':
                        label = '%s\n%s\n%s'%(label_a, ' - ' , label_b)
                    elif math_keyword == 'SLICERDIVIDE':
                        label = '%s\n%s\n%s'%(label_a, ' / ' , label_b)
                    elif math_keyword == 'SLICERMULTIPLY':
                        label = '%s\n%s\n%s'%(label_a, ' x ' , label_b)
                    elif math_keyword == 'SLICERMAX':
                        label = 'Max of \n%s and %s'%(label_a, label_b)
                    elif math_keyword == 'SLICERMIN':
                        label = 'Min of \n%s and %s'%(label_a, label_b)
                    elif math_keyword == 'SLICERMEAN':
                        label = 'Mean of \n%s and %s'%(label_a, label_b)
                    else:
                        label = '%s\n%s\n%s'%(label_a, math_keyword , label_b)

                else:
                    label = 'Failed math operation parsing'
                
                try:
                    if math_keyword == 'SLICERADD':
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        x_min_val = min(current_bin_edges_a) + min(current_bin_edges_b)
                        x_max_val = max(current_bin_edges_a) + max(current_bin_edges_b)
                    elif math_keyword == 'SLICERSUBTRACT':
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        x_min_val = min(current_bin_edges_a) - max(current_bin_edges_b)
                        x_max_val = max(current_bin_edges_a) - min(current_bin_edges_b)

                    elif math_keyword == 'SLICERMULTIPLY':
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        vals_of_interest = [min(current_bin_edges_a) * min(current_bin_edges_b) , min(current_bin_edges_a) * max(current_bin_edges_b), max(current_bin_edges_a) * min(current_bin_edges_b), max(current_bin_edges_a) * max(current_bin_edges_b)]
                        x_min_val = min(vals_of_interest)
                        x_max_val = max(vals_of_interest)
                    elif math_keyword == 'SLICERDIVIDE':
                        #Probably doesn't handle div by 0 well
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        vals_of_interest = [min(current_bin_edges_a) / min(current_bin_edges_b) , min(current_bin_edges_a) / max(current_bin_edges_b), max(current_bin_edges_a) / min(current_bin_edges_b), max(current_bin_edges_a) / max(current_bin_edges_b)]
                        x_min_val = min(vals_of_interest)
                        x_max_val = max(vals_of_interest)
                        #import pdb; pdb.set_trace()
                    elif math_keyword == 'SLICERMAX':
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        x_min_val = min(min(current_bin_edges_a), min(current_bin_edges_b))
                        x_max_val = max(max(current_bin_edges_a), max(current_bin_edges_b))
                    elif math_keyword == 'SLICERMIN':
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        x_min_val = min(min(current_bin_edges_a), min(current_bin_edges_b))
                        x_max_val = max(max(current_bin_edges_a), max(current_bin_edges_b))
                    elif math_keyword == 'SLICERMEAN':
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        x_min_val = min(min(current_bin_edges_a), min(current_bin_edges_b))
                        x_max_val = max(max(current_bin_edges_a), max(current_bin_edges_b))
                    else:
                        param = self.getModifiedDataFromParam(eventids, param_key, verbose=False)
                        x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                        x_min_val = min(param)
                        x_max_val = max(param)
                except Exception as e:
                    param = self.getModifiedDataFromParam(eventids, param_key, verbose=False)
                    x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                    x_min_val = min(param)
                    x_max_val = max(param)
                if numpy.any((numpy.isinf(x_min_val), numpy.isinf(x_max_val),numpy.isnan(x_min_val), numpy.isnan(x_max_val))):
                    param = self.getModifiedDataFromParam(eventids, param_key, verbose=False)
                    x_n_bins = max((len(current_bin_edges_a) , len(current_bin_edges_b)))
                    x_min_val = min(param)
                    x_max_val = max(param)

            else:

                if param_key in self.known_param_keys:
                    #Append snr to the lists below only if max value isn't present.
                    if verbose:
                        print('\tPreparing to get counts for %s'%param_key)
                    if self.max_snr_val is None:
                        std_requiring_params = ['std_h','std_v','snr_h','snr_v','min_snr_h','min_snr_v','snr_gap_h','snr_gap_v'] #List all params that might require max snr to be calculated if hard limit not given.
                        p2p_requiring_params = ['p2p_h','p2p_v','snr_h','snr_v','min_snr_h','min_snr_v','snr_gap_h','snr_gap_v'] #List all params that might require max p2p to be calculated if hard limit not given.
                    else:
                        std_requiring_params = ['std_h','std_v'] #List all params that might require max snr to be calculated if hard limit not given.
                        p2p_requiring_params = ['p2p_h','p2p_v'] #List all params that might require max p2p to be calculated if hard limit not given.

                    if numpy.isin(param_key,std_requiring_params):
                        if self.max_std_val is None:
                            with h5py.File(self.analysis_filename, 'r') as file:
                                print('Calculating max_std')
                                self.max_std_val = numpy.max(file['std'][...][eventids,:]) #Should only have to be done once on first call. 
                                self.min_std_val = numpy.min(file['std'][...][eventids,:])
                                file.close()
                    if numpy.isin(param_key,p2p_requiring_params):
                        if self.max_p2p_val is None:
                            with h5py.File(self.analysis_filename, 'r') as file:
                                print('Calculating max_p2p')
                                self.max_p2p_val = numpy.max(file['p2p'][...][eventids,:]) #Should only have to be done once on first call. 
                                file.close()

                    if self.max_snr_val is None:
                        self.max_snr_val = self.max_p2p_val/self.min_std_val

                    #Append the max beam power or max beam not present and precalculated if not present
                    if numpy.isin(param_key,['triggered_beams','beam_power']):
                        if numpy.logical_or(self.max_beam_power is None, self.max_beam_number is None):
                            triggered_beams, beam_power, unused_eventids = self.reader.returnTriggerInfo()
                            self.max_beam_power = numpy.max(beam_power)
                            self.max_beam_number = numpy.max(triggered_beams)
                        
                else:
                    print('WARNING!!!')
                    print('Given key [%s] is not listed in known_param_keys:\n%s'%(param_key,str(self.known_param_keys)))
                    import pdb; pdb.set_trace()
                    return
                


                if param_key == 'impulsivity_h':
                    label = 'Impulsivity (hpol)'
                    x_n_bins = self.impulsivity_n_bins_h
                    x_max_val = 1
                    x_min_val = -0.25
                elif param_key == 'impulsivity_v':
                    label = 'Impulsivity (vpol)'
                    x_n_bins = self.impulsivity_n_bins_v
                    x_max_val = 1
                    x_min_val = -0.25
                elif param_key == 'cr_template_search_h':
                    label = 'HPol Correlation Values with CR Template'
                    x_n_bins = self.cr_template_n_bins_h
                    x_max_val = 1
                    x_min_val = 0
                elif param_key == 'cr_template_search_v':
                    label = 'VPol Correlation Values with CR Template'
                    x_n_bins = self.cr_template_n_bins_v
                    x_max_val = 1
                    x_min_val = 0
                elif param_key == 'std_h':
                    label = 'Mean Time Domain STD (hpol)'
                    x_n_bins = self.std_n_bins_h
                    x_max_val = self.max_std_val
                    x_min_val = 0
                elif param_key == 'std_v':
                    label = 'Mean Time Domain STD (vpol)'
                    x_n_bins = self.std_n_bins_v
                    x_max_val = self.max_std_val
                    x_min_val = 0
                elif param_key == 'p2p_h': 
                    label = 'Mean P2P (hpol)'
                    x_n_bins = self.p2p_n_bins_h
                    x_max_val = self.max_p2p_val
                    x_min_val = 0
                elif param_key == 'p2p_v': 
                    label = 'Mean P2P (vpol)'
                    x_n_bins = self.p2p_n_bins_v
                    x_max_val = self.max_p2p_val
                    x_min_val = 0
                elif param_key == 'snr_h':
                    label = 'SNR (hpol)\n max(P2P)/min(STD)'
                    x_n_bins = self.snr_n_bins_h
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif param_key == 'min_snr_h':
                    label = 'Min SNR (hpol)\n min(P2P/STD)'
                    x_n_bins = self.snr_n_bins_h
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif param_key == 'snr_gap_h':
                    label = 'Max - Min SNR (hpol)'
                    x_n_bins = self.snr_n_bins_h
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif param_key == 'snr_v':
                    label = 'SNR (vpol)\n max(P2P)/min(STD)'
                    x_n_bins = self.snr_n_bins_v
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif param_key == 'min_snr_v':
                    label = 'Min SNR (vpol)\n min(P2P/STD)'
                    x_n_bins = self.snr_n_bins_v
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif param_key == 'snr_gap_v':
                    label = 'Max - Min SNR (vpol)'
                    x_n_bins = self.snr_n_bins_v
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif 'time_delay_' in param_key:
                    if 'map_max_' in param_key:
                        #Time delays derived from the maps max value
                        split_param_key = param_key.split('_')
                        if split_param_key[6] == 'allsky':
                            title_scope = 'All Sky'
                        elif split_param_key[6] == 'belowhorizon':
                            title_scope = 'Below Horizon'
                        elif split_param_key[6] == 'abovehorizon':
                            title_scope = 'Above Horizon'
                        else:
                            title_scope = ''

                        label = '%s-pol Time Delay from %s Max Map Location\n Ant %s - Ant %s'%(split_param_key[5].title(), title_scope, split_param_key[4].split('subtract')[0],split_param_key[4].split('subtract')[1])
                        if split_param_key[3] == 'h':
                            x_n_bins = self.time_delays_n_bins_h
                        else:
                            x_n_bins = self.time_delays_n_bins_v
                        
                        if numpy.logical_or(self.min_time_delays_val is None, self.max_time_delays_val is None):
                            time_delays = self.getDataFromParam(eventids, param_key)
                        
                        if self.min_time_delays_val is None:
                            x_min_val = min(time_delays) - 1
                        else:
                            x_min_val = self.min_time_delays_val

                        if self.max_time_delays_val is None:
                            x_max_val = max(time_delays) + 1
                        else:
                            x_max_val = self.max_time_delays_val

                    else:
                        # Time delays calculated from cross correlations
                        split_param_key = param_key.split('_')
                        label = '%s-pol Time Delay From XCorr\n Ant %s - Ant %s'%(split_param_key[3].title(),split_param_key[2].split('subtract')[0],split_param_key[2].split('subtract')[1])
                        if split_param_key[3] == 'h':
                            x_n_bins = self.time_delays_n_bins_h
                        else:
                            x_n_bins = self.time_delays_n_bins_v
                        
                        if numpy.logical_or(self.min_time_delays_val is None, self.max_time_delays_val is None):
                            time_delays = self.getDataFromParam(eventids, param_key)
                        
                        if self.min_time_delays_val is None:
                            x_min_val = min(time_delays) - 1
                        else:
                            x_min_val = self.min_time_delays_val

                        if self.max_time_delays_val is None:
                            x_max_val = max(time_delays) + 1
                        else:
                            x_max_val = self.max_time_delays_val
                elif 'max_corr_' in param_key and 'map' not in param_key:
                    split_param_key = param_key.split('_')
                    if 'mean_max_corr' in param_key:
                        label = '%s-pol All Baselines Mean Value From XCorr'%(split_param_key[3].title())
                    elif param_key == 'max_max_corr':
                        label = '%s-pol All Baselines Max Value From XCorr'%(split_param_key[3].title())
                    else:
                        split_param_key = param_key.split('_')
                        label = '%s-pol Max Value From XCorr\n Ant %s - Ant %s'%(split_param_key[3].title(),split_param_key[2].split('subtract')[0],split_param_key[2].split('subtract')[1])

                    x_n_bins = self.max_corr_n_bins
                    x_min_val = self.min_max_corr_val
                    x_max_val = self.max_max_corr_val

                elif 'cw_present' == param_key:
                    label = 'CW Detected Removed (1) or Not (0)'
                    calculate_bins_from_min_max = False
                    current_bin_edges = numpy.array([0.        , 0.33333333, 0.66666667, 1.        ])#
                    # x_n_bins = 3
                    # x_max_val = 1
                    # x_min_val = 0
                elif 'cw_freq_Mhz' == param_key:
                    with h5py.File(self.analysis_filename, 'r') as file:
                        x_max_val = 1000*float(file['cw'].attrs['sine_subtract_min_freq_GHz'])
                        x_min_val = 1000*float(file['cw'].attrs['sine_subtract_max_freq_GHz'])
                        cw_dsets = list(file['cw'].keys())
                        if not hasattr(self, 'cw_prep'):
                            if verbose:
                                print('Creating FFTPrepper class to prepare CW bins.')
                            self.cw_prep = FFTPrepper(self.reader, final_corr_length=int(file['cw'].attrs['final_corr_length']), crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), waveform_index_range=(None,None), plot_filters=False)
                            self.cw_prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)

                    label = 'Identified CW Freq (MHz)'
                    
                    raw_freqs = self.cw_prep.rfftWrapper(self.cw_prep.t(), numpy.ones_like(self.cw_prep.t()))[0]
                    df = raw_freqs[1] - raw_freqs[0]
                    current_bin_edges = (numpy.append(raw_freqs,raw_freqs[-1]+df) - df/2)/1e6 #MHz
                    current_bin_edges = current_bin_edges[current_bin_edges<120]
                    x_n_bins = len(current_bin_edges)
                    calculate_bins_from_min_max = False #override the default behaviour below.
                elif 'cw_linear_magnitude' == param_key:
                    label = 'abs(linear magnitude) of\nmaximum identified CW Peak'
                    x_n_bins = 1000
                    x_max_val = 10000 # A Guess, Try it out and adjust.
                    x_min_val = 0
                elif 'cw_dbish' == param_key:
                    label = 'dBish Magnitude of\nmaximum identified CW Peak'
                    x_n_bins = 120
                    x_max_val = 60
                    x_min_val = 0


                elif 'theta_best_choice' in param_key:
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Zenith (Deg)\nCalculated from Best P2P*Max of All Maps'
                    else:
                        label = 'Best Reconstructed Zenith (Deg)\nCalculated from Best P2P*Max of All Maps'
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False

                elif 'elevation_best_choice' in param_key:
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Elevation (Deg)\nCalculated from Best P2P*Max of All Maps'
                    else:
                        label = 'Best Reconstructed Elevation (Deg)\nCalculated from Best P2P*Max of All Maps'
                    current_bin_edges = self.elevation_edges
                    calculate_bins_from_min_max = False

                elif 'phi_best_choice' in param_key:
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Azimuth (Deg)\nCalculated from Best P2P*Max of All Maps'
                    else:
                        label = 'Best Reconstructed Azimuth (Deg)\nCalculated from Best P2P*Max of All Maps'
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False




                elif 'hilbert_' not in param_key and 'theta_best_h' in param_key:
                    scope = param_key.replace('theta_best_h','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Zenith (Deg)\nHpol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Zenith (Deg)\nHpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' not in param_key and 'theta_best_v' in param_key:
                    scope = param_key.replace('theta_best_v','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Zenith (Deg)\nVpol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Zenith (Deg)\nVpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False

                elif 'hilbert_' not in param_key and 'theta_best_all' in param_key:
                    scope = param_key.replace('theta_best_all','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Zenith (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Zenith (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False

                elif 'hilbert_' not in param_key and 'elevation_best_h' in param_key:
                    scope = param_key.replace('elevation_best_h','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Elevation (Deg)\nHpol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Elevation (Deg)\nHpol Antennas Only' + ' ' + scope
                        current_bin_edges = self.elevation_edges
                        calculate_bins_from_min_max = False

                elif 'hilbert_' not in param_key and 'elevation_best_v' in param_key:
                    scope = param_key.replace('elevation_best_v','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Elevation (Deg)\nVpol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Elevation (Deg)\nVpol Antennas Only' + ' ' + scope
                        current_bin_edges = self.elevation_edges
                        calculate_bins_from_min_max = False

                elif 'hilbert_' not in param_key and 'elevation_best_all' in param_key:
                    scope = param_key.replace('elevation_best_all','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Elevation (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Elevation (Deg)\nAll-pol Antennas Only' + ' ' + scope
                        current_bin_edges = self.elevation_edges
                        calculate_bins_from_min_max = False

                elif 'hilbert_' not in param_key and 'phi_best_h' in param_key:
                    scope = param_key.replace('phi_best_h','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Azimuth (Deg)\nHpol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Azimuth (Deg)\nHpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False

                elif 'hilbert_' not in param_key and 'phi_best_v' in param_key:
                    scope = param_key.replace('phi_best_v','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Azimuth (Deg)\nVpol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Azimuth (Deg)\nVpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' not in param_key and 'phi_best_all' in param_key:
                    scope = param_key.replace('phi_best_all','')
                    if numpy.logical_and(self.hilbert_map == True, self.normal_map == False):
                        label = 'Best Reconstructed Hilbert Azimuth (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    else:
                        label = 'Best Reconstructed Azimuth (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False

                elif 'hilbert_' in param_key and 'theta_best_h' in param_key:
                    scope = param_key.replace('hilbert_theta_best_h','')
                    label = 'Best Reconstructed Hilbert Zenith (Deg)\nHpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'theta_best_v' in param_key:
                    scope = param_key.replace('hilbert_theta_best_v','')
                    label = 'Best Reconstructed Hilbert Zenith (Deg)\nVpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'theta_best_all' in param_key:
                    scope = param_key.replace('hilbert_theta_best_all','')
                    label = 'Best Reconstructed Hilbert Zenith (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    current_bin_edges = self.theta_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'elevation_best_h' in param_key:
                    scope = param_key.replace('hilbert_elevation_best_h','')
                    label = 'Best Reconstructed Hilbert Elevation (Deg)\nHpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.elevation_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'elevation_best_v' in param_key:
                    scope = param_key.replace('hilbert_elevation_best_v','')
                    label = 'Best Reconstructed Hilbert Elevation (Deg)\nVpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.elevation_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'elevation_best_all' in param_key:
                    scope = param_key.replace('hilbert_elevation_best_all','')
                    label = 'Best Reconstructed Hilbert Elevation (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    current_bin_edges = self.elevation_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'phi_best_h' in param_key:
                    scope = param_key.replace('hilbert_phi_best_h','')
                    label = 'Best Reconstructed Hilbert Azimuth (Deg)\nHpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'phi_best_v' in param_key:
                    scope = param_key.replace('hilbert_phi_best_v','')
                    label = 'Best Reconstructed Hilbert Azimuth (Deg)\nVpol Antennas Only' + ' ' + scope
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False
                elif 'hilbert_' in param_key and 'phi_best_all' in param_key:
                    scope = param_key.replace('hilbert_phi_best_all','')
                    label = 'Best Reconstructed Hilbert Azimuth (Deg)\nAll-pol Antennas Only' + ' ' + scope
                    current_bin_edges = self.phi_edges
                    calculate_bins_from_min_max = False
                elif 'calibrated_trigtime' == param_key:
                    label = 'Calibrated Trigger Time (s)'
                    with h5py.File(self.analysis_filename, 'r') as file:
                        x_min_val = file['calibrated_trigtime'][...][0]
                        x_max_val = file['calibrated_trigtime'][...][-1]

                    x_n_bins = numpy.ceil((x_max_val - x_min_val)/60).astype(int) #bin into roughly 1 min chunks.
                elif 'triggered_beams' == param_key:
                    x_min_val = -0.5
                    x_max_val = self.max_beam_number + 0.5 #This is crude but at least it avoids hard coding the number of beams.
                    x_n_bins = int(x_max_val - x_min_val)
                    label = 'Triggered Beam'
                elif 'beam_power' == param_key:
                    label = 'Beam Power Sum (arb)'
                    x_min_val = 0
                    x_max_val = self.max_beam_power
                    x_n_bins = 2*128 #Unsure what is reasonable for this number.  

                elif 'peak_to_sidelobe' in param_key:
                    if 'allsky' in param_key:
                        label_scope = 'All Sky'
                    elif 'belowhorizon' in param_key:
                        label_scope = 'Below Horizon'
                    elif 'abovehorizon' in param_key:
                        label_scope = 'Above Horizon'
                    else:
                        label_scope = ''


                    if 'hilbert_' in param_key:
                        label = '%s %s Peak to Sidelobe Ratio (Hilbert Envelope Applied)'%(param_key.split('_')[0].title(), label_scope)
                    else:
                        label = '%s %s Peak to Sidelobe Ratio'%(param_key.split('_')[0].title(), label_scope)
                    if 'hpol' in param_key:
                        x_n_bins = self.peak_to_sidelobe_n_bins_h
                    elif 'vpol' in param_key:
                        x_n_bins = self.peak_to_sidelobe_n_bins_v
                    elif 'all_' in param_key:
                        x_n_bins = self.peak_to_sidelobe_n_bins_all
                    else:
                        x_n_bins = self.peak_to_sidelobe_n_bins_h

                    x_max_val = self.peak_to_sidelobe_max_val
                    x_min_val = 1.0


                elif 'coincidence_method_' in param_key:
                    label = 'Coincident Event Count Method %s'%(param_key.replace('coincidence_method_','').replace('_',' '))
                    current_bin_edges = numpy.arange(-0.5,11,1)
                    calculate_bins_from_min_max = False


                elif 'max_possible_map_value' in param_key:
                    if 'hilbert_' in param_key:
                        label = 'Maximum Possible Map Value Per Event (Hilbert Envelope Applied)'
                        x_max_val = 10.0 #Set high because I don't know if hilbert will max out differently
                    else:
                        label = 'Maximum Possible Map Value Per Event'
                        x_max_val = self.max_max_possible_map_value_val

                    x_min_val = 0.0
                    if 'hpol' in param_key:
                        x_n_bins = self.max_possible_map_value_n_bins_h
                    elif 'vpol' in param_key:
                        x_n_bins = self.max_possible_map_value_n_bins_v
                    elif 'all_' in param_key:
                        x_n_bins = self.max_possible_map_value_n_bins_all

                elif 'max_map_value' in param_key:
                    pol = 'HPol '*('hpol' in param_key) + 'VPol '*('vpol' in param_key)
                    scope = 'Above Horizon '*('abovehorizon' in param_key) + 'Below Horizon '*('belowhorizon' in param_key) + 'All Sky '*('allsky' in param_key) 
                    if 'hilbert_' in param_key:
                        label = pol + scope + 'Maximum Map Value Per Event (Hilbert Envelope Applied)'
                        x_max_val = 10.0 #Set high because I don't know if hilbert will max out differently
                    else:
                        label = pol + scope + 'Maximum Map Value Per Event'
                        x_max_val = self.max_max_map_value_val

                    x_min_val = 0.0
                    if 'hpol' in param_key:
                        x_n_bins = self.max_map_value_n_bins_h
                    elif 'vpol' in param_key:
                        x_n_bins = self.max_map_value_n_bins_v
                    elif 'all_' in param_key:
                        x_n_bins = self.max_map_value_n_bins_all
                elif 'similarity_' in param_key:
                    param = self.getDataFromParam(eventids, param_key)
                    label = '%spol Time Delay Similarity %s'%(param_key.split('_')[-1].title(), param_key.split('_')[1].title())
                    x_n_bins = 1000
                    x_max_val = max(param)
                    x_min_val = min(param)
                elif 'event_rate_ts_' in param_key:
                    rate_string, time_window_string = param_key.replace('event_rate_ts_','').split('_')
                    label = 'Test Statistic for Expected Event Rate\nof%s Using %s Window'%(rate_string, time_window_string)
                    x_n_bins = 100
                    x_max_val = self.event_rate_gaus_param[rate_string][time_window_string]['mean'] + 20*self.event_rate_gaus_param[rate_string][time_window_string]['sigma']
                    x_min_val = self.event_rate_gaus_param[rate_string][time_window_string]['mean'] - 10*self.event_rate_gaus_param[rate_string][time_window_string]['sigma']
                elif 'event_rate_sigma_' in param_key:
                    rate_string, time_window_string = param_key.replace('event_rate_sigma_','').split('_')
                    label = 'Test Statistic for Expected Event Rate\nof%s Using %s Window'%(rate_string, time_window_string)
                    x_n_bins = 100
                    x_max_val = 20
                    x_min_val = -10

                elif param_key == 'sun_az':
                    label = 'Azimuth Direction of the Sun (degrees, E=0, N=90)'
                    x_n_bins = 360
                    x_max_val = -180
                    x_min_val = 180

                elif param_key == 'sun_el':
                    label = 'Elevation Direction of the Sun (degrees)'
                    x_n_bins = 360
                    x_max_val = 90
                    x_min_val = -90

            if calculate_bins_from_min_max:
                current_bin_edges = numpy.linspace(x_min_val,x_max_val,x_n_bins + 1) #These are bin edges

            return current_bin_edges, label
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setCurrentPlotBins(self, main_param_key_x, main_param_key_y, eventids):
        '''
        Sets the current bins to be used for main_param_key.  These can then be accessed easily by other functions for
        producing the hist count data.  This should be called by any functions starting a plot. 

        TODO: Will need to add a load feature later. 
        '''
        self.current_bin_edges_x, self.current_label_x = self.getSingleParamPlotBins(main_param_key_x, eventids)
        self.current_bin_edges_y, self.current_label_y = self.getSingleParamPlotBins(main_param_key_y, eventids)

        self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y = numpy.meshgrid((self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2, (self.current_bin_edges_y[:-1] + self.current_bin_edges_y[1:]) / 2) # Use for contours
        self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y = numpy.meshgrid(self.current_bin_edges_x , self.current_bin_edges_y)#use for pcolormesh
        

    def returnMaskedArray(self,data,mask_top_N_bins,fill_value=None,fallback_mode=None):
        '''
        The input number of bins with the highest counts will be masked, and a masked array of the same shape will be
        returned.  Because multiple bins may have the same content, more than mask_top_N_bins may actually be removed.

        If the given number of bins to mask is too much (more than the number of bins with meaningful content), then
        the code will execute based on fallback_mode.  fallback_mode == 0 will return the original data, with no mask,
        while fallback_mode == 1 will return the data masked for everything not equal to the minimum non-zero bin count.
        If None then it will use the default defined by the class.
        '''
        try:
            if fallback_mode is None:
                fallback_mode = self.masked_fallback_mode
            if fill_value is None:
                fill_value = numpy.ma.default_fill_value(data.flat[0])
            if mask_top_N_bins == 0:
                output = numpy.ma.masked_array(data,fill_value=fill_value)
            elif numpy.sum(data) == 0:
                print('DATA GIVEN IS 0. RETURNING.')
                output = numpy.ma.masked_array(data,fill_value=fill_value)
            elif mask_top_N_bins > numpy.size(data):
                print('WARNING!!! NUMBER OF BINS TO MASK [%i] GREATER THAN NUMBER OF BINS.'%mask_top_N_bins)
                print('RETURNING ORIGINAL DATA UNMASKED')
                output = numpy.ma.masked_array(data,fill_value=fill_value)

            elif mask_top_N_bins >= numpy.sum(data > 0):
                print('WARNING!!! NUMBER OF BINS TO MASK [%i] GREATER THAN OR EQUAL TO THE NUMBER OF BINS WITH CONTENT [%i].'%(mask_top_N_bins,numpy.sum(data > 0)))
                if fallback_mode == 0:
                    print('RETURNING ORIGINAL DATA UNMASKED')
                    output = numpy.ma.masked_array(data,fill_value=fill_value)
                else:
                    sorted_bin_contents = numpy.unique(numpy.sort(numpy.concatenate(data))[::-1])
                    maximum_accepted_bin_content = numpy.min(sorted_bin_contents[sorted_bin_contents > 0])
                    output = numpy.ma.masked_array(data,mask=data > maximum_accepted_bin_content,fill_value=fill_value) #fallback, no longer >=, just >
                    if numpy.sum(output) == 0:
                        print('RETURNING DATA WITHOUT MASK DUE TO EMPTY COUNT ATTEMPT')
                        output = numpy.ma.masked_array(data,fill_value=fill_value)   
                    else:
                        print('RETURNING DATA WITH MASK NOT INCLUDING MINIMUM NON-ZERO BINS')
            else:
                minimum_denied_bin_content = numpy.sort(numpy.concatenate(data))[::-1][mask_top_N_bins] #Anything greater than or equal to this will be masked.

                if minimum_denied_bin_content <= 1:
                    minimum_denied_bin_content = 1
                    print('WARNING!!! THE MINIMUM DENIED BIN CONTENT FOR MASK IS <= 1 (RESULTING IN NO COUNTS).')
                    #print('WARNING!!! NUMBER OF BINS TO MASK [%i] GREATER THAN OR EQUAL TO THE NUMBER OF BINS WITH CONTENT GREATER THAN OR EQUAL TO MINIMUM DENIED BIN CONTENT [%i]. , IGNORING.'%(mask_top_N_bins,numpy.sum(data >= minimum_denied_bin_content)))
                    if fallback_mode == 0:
                        print('RETURNING ORIGINAL DATA UNMASKED')
                        output = numpy.ma.masked_array(data,fill_value=fill_value)
                    else:
                        output = numpy.ma.masked_array(data,mask=data > minimum_denied_bin_content,fill_value=fill_value) #fallback, no longer >=, just >
                        if numpy.sum(output) == 0:
                            print('RETURNING DATA WITHOUT MASK DUE TO EMPTY COUNT ATTEMPT')
                            output = numpy.ma.masked_array(data,fill_value=fill_value)   
                        else:
                            print('RETURNING DATA WITH MASK NOT INCLUDING MINIMUM NON-ZERO BINS')
                else:
                    output = numpy.ma.masked_array(data,mask=data >= minimum_denied_bin_content,fill_value=fill_value) #default behaviour, should be >=
            return output
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def plot2dHist(self, main_param_key_x,  main_param_key_y, eventids, title=None,cmap='coolwarm', return_counts=False, load=False,lognorm=True, mask_top_N_bins=0, fill_value=None):
        '''
        This is meant to be a function the plot corresponding to the main parameter, and will plot the same quantity 
        (corresponding to main_param_key) with just events corresponding to the cut being used.  This subset will show
        up as a contour on the plot.  

        Parameters
        ----------
        mask_top_N_bins : int
            Given a value here, the map will be converted to a masked array, where the most populaated N bins will be
            masked.
        '''
        try:
            #Should make eventids a self.eventids so I don't need to call this every time.
            counts = self.get2dHistCounts(main_param_key_x,main_param_key_y,eventids,load=load,set_bins=True,mask_top_N_bins=mask_top_N_bins0, fill_value=fill_value) #set_bins should only be called on first call, not on contours.
            if mask_top_N_bins > 0:
                masked_bins = numpy.sum(counts.mask())


            _fig, _ax = plt.subplots()
            if title is None:
                title = '%s, Run = %i\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,int(self.reader.run),str(self.trigger_types))
                if mask_top_N_bins > 0:
                    title += ', masked_bins = %i'%masked_bins
            plt.title(title)

            if lognorm == True:
                _im = _ax.pcolormesh(self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y, counts,norm=colors.LogNorm(vmin=0.5, vmax=counts.max()),cmap=cmap)#cmap=plt.cm.coolwarm
            else:
                _im = _ax.pcolormesh(self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y, counts,cmap=cmap)
            if 'theta_best_' in main_param_key_y:
                _ax.invert_yaxis()
            if True:
                if numpy.logical_or(numpy.logical_and('phi_best_' in main_param_key_x,'phi_best_' in main_param_key_y),numpy.logical_or(numpy.logical_and('theta_best_' in main_param_key_x,'theta_best_' in main_param_key_y),numpy.logical_and('elevation_best_' in main_param_key_x,'elevation_best_' in main_param_key_y))):
                    plt.plot(self.current_bin_centers_mesh_y[:,0],self.current_bin_centers_mesh_y[:,0],linewidth=1,linestyle='--',color='tab:gray',alpha=0.5)
                if numpy.logical_and('phi_best_' in main_param_key_x,numpy.logical_or('theta_best_' in main_param_key_y,'elevation_best_' in main_param_key_y)):
                    #Make cor to plot the array plane.
                            
                    cor = Correlator(self.reader,  upsample=2**10, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False,apply_phase_response=False, tukey=False, sine_subtract=False, deploy_index=self.map_deploy_index) #only for array plane
                    if numpy.logical_and(main_param_key_x.split('_')[-1] == 'h', main_param_key_y.split('_')[-1] == 'h'):
                        plane_xy = cor.getPlaneZenithCurves(cor.n_hpol.copy(), 'hpol', 90.0, azimuth_offset_deg=0.0)
                    elif numpy.logical_and(main_param_key_x.split('_')[-1] == 'v', main_param_key_y.split('_')[-1] == 'v'):
                        plane_xy = cor.getPlaneZenithCurves(cor.n_vpol.copy(), 'vpol', 90.0, azimuth_offset_deg=0.0)
                    else:
                        plane_xy = None
                    if plane_xy is not None:
                        if 'elevation_best_' in main_param_key_y:
                            plane_xy[1] = 90.0 - plane_xy[1]

                        plt.plot(plane_xy[0], plane_xy[1],linestyle='-',linewidth=1,color='k')
                        plt.xlim([numpy.min(self.range_phi_deg),numpy.max(self.range_phi_deg)])
                        #plt.xlim([-180,180])
            
            plt.xlabel(self.current_label_x)
            plt.ylabel(self.current_label_y)

            plt.grid(which='both', axis='both')
            _ax.minorticks_on()
            _ax.grid(b=True, which='major', color='k', linestyle='-')
            _ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


            try:
                cbar = _fig.colorbar(_im)
                cbar.set_label('Counts')
            except Exception as e:
                print('Error in colorbar, often caused by no events.')
                print(e)

            if return_counts:
                return _fig, _ax, counts
            else:
                return _fig, _ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addContour(self, ax, main_param_key_x, main_param_key_y, contour_eventids, contour_color, load=False, n_contour=5, alpha=0.85, log_contour=True,mask=None,fill_value=None):
        '''
        Given the plot made from plot2dHist, this will add contours to it for the events specified by contour_eventids.
        This assumes that setCurrentPlotBins has already been called by plot2dHist. 

        Parameters
        ----------
        ax : matplotlib axes
            This should be the output axes from plot2dHist.
        main_param_key : str
            This should match the one used for plt2dHist, as it is used to determine the xy coords of the contour and
            what parameter should be calculated (or loaded) for each selected eventid.
        contour_eventids : numpy.ndarray of int
            This should be a list of eventids that meet the cut you wish to plot.
        contour_color : tuple
            The color which you want the contour to be plotted on top of the 
        n_contour : int
            The number of contours to plot.  This will create contours ranging from the min to max count value, with n+1
            lines.  The lowest line will NOT be plotted (thus plotting n total), because this is often 0 and clutters
            the plot.
        log_contour : bool
            If True then the Locator used in matplotlib.axes.Axes.contour will be matplotlib.ticker.LogLocator(), 
            rather than the default contour behaviour of matplotlib.ticker.MaxNLocator.

        Returns
        -------
        ax : matplotlib axes
            This is the updated axes that the contour was added to.
        cs : QuadContourSet
            This is the contour object, that can be used for adding labels to the legend for this contour.
        '''
        try:
            counts = self.get2dHistCounts(main_param_key_x, main_param_key_y, contour_eventids, load=load, set_bins=False)
            if mask is not None:
                #Want mask to be applied on top N bins of original data, not the countoured data.  So mask is given.
                if fill_value is not None:
                    fill_value = numpy.ma.default_fill_value(counts.flat[0])
                counts = numpy.ma.masked_array(counts, mask=mask, fill_value=fill_value)

            if log_contour:
                #Log?
                #locator = ticker.LogLocator()
                levels = numpy.ceil(numpy.logspace(0,numpy.log10(numpy.max(counts)),n_contour))[1:n_contour+1] #Not plotting bottom contour because it is often background and clutters plot.
            else:
                #Linear?
                #locator = ticker.MaxNLocator()
                levels = numpy.ceil(numpy.linspace(0,numpy.max(counts),n_contour))[1:n_contour+1] #Not plotting bottom contour because it is often background and clutters plot.

            levels = numpy.unique(levels) #Covers edge case of very low counts resulting in degenerate contour labels (Must be increasing)

            # The below code was intended to make outer contours more transparent, but it produced differing results from plotting contour once, and I don't know why, so I am just commenting it out :/
            # import pdb; pdb.set_trace()
            # print(levels)
            # for index, level in enumerate(levels[::-1]):
            #     # if index != len(levels)-1:
            #     #     continue
            #     _alpha = alpha*( 0.7*((index+1)/len(levels)) + 0.3) #Highest level contour should be specified alpha, others are dimmer to make it clear where high concentration is.  Doesn't let opacity go below 0.3.
            #     # print(_alpha)
            #     print(level)
            #     cs = ax.contour(self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y, counts, colors=[contour_color],levels=[level],alpha=_alpha, antialiased=True)
            # print(levels)
            cs = ax.contour(self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y, counts, colors=[contour_color],levels=levels,alpha=alpha, antialiased=True)

            return ax, cs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            

    def plotROI2dHist(self, main_param_key_x, main_param_key_y, eventids=None, cmap='coolwarm', include_roi=True, load=False, lognorm=True, mask_top_N_bins=0, fill_value=0, suppress_legend=False):
        '''
        This is the "do it all" function.  Given the parameter it will plot the 2dhist of the corresponding param by
        calling plot2dHist.  It will then plot the contours for each ROI on top.  It will do so assuming that each 
        ROI has a box cut in self.roi for EACH parameter.  I.e. it expects the # of listed entries in each ROI to be
        the same, and it will add a contour for the eventids that pass ALL cuts for that specific roi. 

        If eventids is given then then those events will be used to create the plot, and the trigger type cut will be ignored.
        '''
        try:
            if eventids is None:
                eventids = self.getEventidsFromTriggerType()
                #The given eventids don't necessarily match the default cut, and thus shouldn't be labeled as such in the title.
                title = '%s, Run = %i\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,int(self.reader.run),str(self.trigger_types))                
            else:
                title = '%s, Run = %i'%(main_param_key_x + ' vs ' + main_param_key_y,int(self.reader.run))
            fig, ax, counts = self.plot2dHist(main_param_key_x, main_param_key_y, eventids, title=title, cmap=cmap,lognorm=lognorm, return_counts=True, mask_top_N_bins=mask_top_N_bins, fill_value=fill_value) #prepares binning, must be called early (before addContour)

            #these few lines below this should be used for adding contours to the map. 
            if include_roi:
                legend_properties = []
                legend_labels = []

                for roi_index, roi_key in enumerate(list(self.roi.keys())):
                    contour_eventids = self.getCutsFromROI(roi_key, load=load)
                    contour_eventids = numpy.intersect1d(contour_eventids,eventids) #getCutsFromROI doesn't account for eventids, this makes it only those that are in ROI and given.
                    # print(len(contour_eventids))
                    #import pdb; pdb.set_trace()
                    ax, cs = self.addContour(ax, main_param_key_x, main_param_key_y, contour_eventids, self.roi_colors[roi_index], n_contour=6,mask=counts.mask, fill_value=fill_value)
                    legend_properties.append(cs.legend_elements()[0][0])
                    legend_labels.append('roi %i: %s'%(roi_index, roi_key))

                if suppress_legend == False:
                    plt.legend(legend_properties,legend_labels,loc='upper left')

            return fig, ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print('Run: ',self.reader.run)
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def plotTimeDelayHist(self,include_roi=True, load=False):
        '''
        This will plot 1d histograms for all time delays, plotted in a grid.  It will make a seperate plot
        for both hpol and vpol (total of 12 subplots).
        '''
        if load == True:
            print('Load feature not currently supported for plotTimeDelayHist')

        eventids = self.getEventidsFromTriggerType()
        
        if include_roi:
            roi_eventids = []
            for roi_index, roi_key in enumerate(list(self.roi.keys())):
                roi_eventids.append(self.getCutsFromROI(roi_key, load=load))

        for mode in ['h','v']:
            fig = plt.figure()
            fig.canvas.set_window_title('%spol Time Delays'%(mode.title()))
            for baseline_index, baseline in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                param_key = 'time_delay_%isubtract%i_%s'%(baseline[0],baseline[1],mode)
                
                self.setCurrentPlotBins(param_key, param_key, eventids)
                label = self.current_label_x
                param = self.getDataFromParam(eventids, param_key)
                ax = plt.subplot(3,2,baseline_index+1)
                if include_roi:
                    alpha = 0.5
                else:
                    alpha = 1.0

                plt.hist(param, bins = self.current_bin_edges_x,label=label,alpha=alpha)
                plt.minorticks_on()
                plt.yscale('log', nonposy='clip')
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                if baseline_index in [4,5]:
                    plt.xlabel('Time Delay (ns)')
                if baseline_index in [0,2,4]:
                    plt.ylabel('Counts')

                if include_roi:
                    for roi_index, roi_key in enumerate(list(self.roi.keys())):
                        param = self.getDataFromParam(roi_eventids[roi_index], param_key)
                        c = numpy.copy(self.roi_colors[roi_index])
                        c[3] = 0.7 #setting alpha for fill but not for edge.
                        plt.hist(param, bins = self.current_bin_edges_x,label = 'roi %i: %s'%(roi_index, roi_key), color = c, edgecolor='black', linewidth=1)
                plt.legend(fontsize=10)


    def saveHistogramData(self, param_key):
        '''
        This will determine the bin locations for a given parameter choice and store that value and the necessary meta
        data to the analysis file. 
        '''
        eventids = numpy.arange(self.reader.N())
        bin_edges, label = self.getSingleParamPlotBins(param_key, eventids)
        print('\tLoading data for %s'%param_key)
        param_data = self.getDataFromParam(eventids, param_key)
        bin_indices_raw = numpy.digitize(param_data,bins=bin_edges)

        print('\tWriting digitized bin data for %s'%param_key)
        with h5py.File(self.analysis_filename, 'a') as file:

            dsets = list(file.keys()) #Existing datasets
            if not numpy.isin('histogram',dsets):
                file.create_group('histogram')
            else:
                print('histogram group already exists in file %s'%self.analysis_filename)

            histogram_dsets = list(file['histogram'].keys())
            
            if numpy.isin(param_key,histogram_dsets):
                print('histogram["%s"] dset already exists in file %s, replacing'%(param_key,self.analysis_filename))
                del file['histogram'][param_key]
            file['histogram'].create_dataset(param_key, (file.attrs['N'],), dtype='i', compression='gzip', compression_opts=4, shuffle=True)


            file['histogram'][param_key][...] = bin_indices_raw
            file['histogram'][param_key].attrs['bin_edges'] = bin_edges
            file['histogram'][param_key].attrs['label'] = label
            file.close()

    def loadHistogramData(self, param_key, eventids):
        '''
        This will load the bin locations for a given parameter choice.  The output of this will then need to be converted
        into histogram counts using numpy.add.at and accounting for overflow and underflow bin indices.  

        Note that the whole saving and loading of histogram datasets was implemented when I thought calculating the 
        values/histogramming them was the rate limiting step.  I subsequently learned that the loading (how I was doing
        it) was the problem.  The benefits of saving and loading this data are negligable for most current parameters.

        Returns
        -------
        bin_indices_raw : numpy.ndarray of ints
            This will return the output of numpy.digitize, which roughly corresponds to the index of the bin that each
            events associated param value lies.  This is NOT directly the index however, as 0 indicates underflow (not bin 0),
            and len(bins_edges) indicates overflow, additionally each bin index appears to be base 1 to account for these
            special values.  Use self.cleanHistogramData to get this array as just actually bin indices. 
        bin_edges : numpy.ndarray of floats
            These are the bins that were used to calculate bin_indices_raw.
        label : str
            This is the label that this dataset uses when plotting. 
        eventids : numpy.ndarray of ints
            The event indices you want to load histogram data for.  
        '''
        print('\tLoading digitized bin data for %s'%param_key)

        with h5py.File(self.analysis_filename, 'r') as file:
            bin_indices_raw = file['histogram'][param_key][...][eventids]
            bin_edges = file['histogram'][param_key].attrs['bin_edges']
            label = file['histogram'][param_key].attrs['label']
            file.close()

        return bin_indices_raw, bin_edges, label

    def cleanHistogramData(self, bin_indices, bin_edges):
        '''
        Given bin_indices and bin_edges, this will return a masked array where the underflow and overflow bins are masked.
        This array of indices could then be used as indices with numpy.add.at to turn these indices into histograms.

        Note that the whole saving and loading of histogram datasets was implemented when I thought calculating the 
        values/histogramming them was the rate limiting step.  I subsequently learned that the loading (how I was doing
        it) was the problem.  The benefits of saving and loading this data are negligable for most current parameters.
        '''
        return numpy.ma.masked_array(bin_indices,mask=numpy.logical_or(bin_indices == 0, bin_indices == len(bin_edges))) - 1 #Mask to ignore underflow/overflow bins.  Indices are base 1 here it seems.  

    def plotROIWaveforms(self, roi_key=None, final_corr_length=2**13, crit_freq_low_pass_MHz=None, low_pass_filter_order=None, crit_freq_high_pass_MHz=None, high_pass_filter_order=None, waveform_index_range=(None,None), plot_filter=False, apply_phase_response=False, save=False, plot_saved_templates=False, sine_subtract=False):
        '''
        This will call the TemplateCompareTool and use it to plot averaged FFT and waveforms for events passing the
        specificied ROI key.  If the ROI key is not given then ALL ROI plots will be produced.  If the template
        compare tool has already been called then the parameters associated with it will be ignored and the interally
        defined template compare tool will be used. 
        '''
        if self.tct is None:
            self.tct = TemplateCompareTool(self.reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, waveform_index_range=waveform_index_range, plot_filters=False,apply_phase_response=apply_phase_response)
            if sine_subtract:
                self.tct.addSineSubtract(0.03, 0.09, 0.03, max_failed_iterations=3, verbose=False, plot=False)

        if roi_key is None:
            keys = list(self.roi.keys())
        else:
            keys = [roi_key]

        for roi_key in keys:
            eventids = self.getCutsFromROI(roi_key,load=False,save=False)
            if len(eventids) == 0:
                print('No eventids in ROI: ' + roi_key)
                continue
            times, averaged_waveforms = self.tct.averageAlignedSignalsPerChannel(eventids, template_eventid=eventids[-1], align_method=0, plot=False, sine_subtract=sine_subtract)
            times_ns = times/1e9
            freqs_MHz = numpy.fft.rfftfreq(len(times_ns),d=numpy.diff(times_ns)[0])/1e6
            #Plot averaged FFT per channel
            fft_fig = plt.figure()
            fft_fig.canvas.set_window_title('FFT %s'%(roi_key))
            for mode_index, mode in enumerate(['hpol','vpol']):
                plt.subplot(2,1,1+mode_index)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.ylabel('%s dBish'%mode)
                plt.xlabel('MHz')
                fft_ax = plt.gca()
                for channel, averaged_waveform in enumerate(averaged_waveforms):
                    if mode == 'hpol' and channel%2 == 1:
                        continue
                    elif mode == 'vpol' and channel%2 == 0:
                        continue

                    freqs, spec_dbish, spec = self.tct.rfftWrapper(times, averaged_waveform)
                    fft_ax.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel)#Dividing by 2 to match monutau.  Idk why I have to do this though normally this function has worked well...
                plt.xlim(10,110)
                plt.ylim(-20,30)


            #Plot averaged Waveform per channel
            wf_fig = plt.figure()
            wf_fig.canvas.set_window_title('WF %s'%(roi_key))
            for mode_index, mode in enumerate(['hpol','vpol']):
                plt.subplot(2,1,1+mode_index)
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                plt.ylabel('%s (adu)'%mode)
                plt.xlabel('Time (ns)')
                wf_ax = plt.gca()
                for channel, averaged_waveform in enumerate(averaged_waveforms):
                    if mode == 'hpol' and channel%2 == 1:
                        continue
                    elif mode == 'vpol' and channel%2 == 0:
                        continue

                    wf_ax.plot(times,averaged_waveform,label='Ch %i'%channel)#Dividing by 2 to match monutau.  Idk why I have to do this though normally this function has worked well...
                #plt.xlim(250,1000)

            if save == True:
                print('Saving waveform templates for ROI: %s'%roi_key)
                resampled_averaged_waveforms = numpy.zeros((8,len(self.tct.waveform_times_corr)))
                resampled_averaged_waveforms_original_length = numpy.zeros((8,len(self.reader.t())))
                for channel in range(8):
                    #Resampling averaged waveforms to be more compatible with cross correlation framework. 
                    resampled_averaged_waveforms[channel] = scipy.interpolate.interp1d(times,averaged_waveforms[channel],kind='cubic',bounds_error=False,fill_value=0)(self.tct.waveform_times_corr)
                    resampled_averaged_waveforms_original_length[channel] = scipy.interpolate.interp1d(times,averaged_waveforms[channel],kind='cubic',bounds_error=False,fill_value=0)(self.reader.t())

                if plot_saved_templates:
                    plt.figure()
                    for channel, wf in enumerate(resampled_averaged_waveforms_original_length):
                        plt.plot(self.reader.t(),wf,label='%i'%channel)
                    plt.xlabel('t (ns)')
                    plt.ylabel('adu (not digitized)')

                template_dir = os.environ['BEACON_ANALYSIS_DIR'] + '/templates/roi/'
                numpy.savetxt(template_dir + 'template_run%i_roi_%s.csv'%(self.reader.run,roi_key.replace(' ','_')),resampled_averaged_waveforms_original_length, delimiter=",")
                meta_file = open(template_dir + 'template_run%i_roi_%s_meta_info.txt'%(self.reader.run,roi_key.replace(' ','_')), 'w')
                meta_file.write(str(self.roi[roi_key]))
                meta_file.close()

class dataSlicer():
    '''
    This will perform the same functions as dataSlicerSingleRun, however instead of being passed a single reader 
    (which is associated with a single run), this class will accept a list of several runs.  Then it will create
    and requested histogram plots using counts from all listed runs.  It does this by creating multiple
    dataSlicerSingleRun objects, and calling their functions, combining results, and plotting. 

    Given the list of runs, this can produce 2d histogram plots for each of the known measureable 
    quantities, with contour functionality for highlighting regions of interest (ROI) as created by the user.  To see
    a list of the currently accepted known parameter keys use the printKnownParamKeys function.

    FOr ROI support, supply a dictionary to the addROI function to apply a cut on any (up to all) of the known parameters
    using this dictionary.  Cuts on each parameter are simple min/max window cuts, but by combining many parameters can
    be used to create higher dimensional parameter box cuts.
    
    With any ROI's defined, call the plotROI2dHist function, selecting 2 parameters you want histogrammed (2DHist).  
    This histogram will be plotted for all events passing early cuts like trigger_types and included_antennas (defined
    in the calling of the class).  Then the ROI's will be plotted on top of this underlying plot - shown as Contours.  
    These contours effectively circle where events passing a particular ROI's cuts show up as a population in the 
    plotted 2d histogram plot.  This can be used to further constrain the ROI or to simply understand it's behaviour
    better.   

    Parameters
    ----------
    runs : list of ints
        This should be a list of run numbers, all of which should be formatted identically with similar analysis
        datasets (especially those specified by impulsivity_dset_key and )
    impulsivity_dset_key : str
        This is a string that must correspond to a specific stored and precalculated impulsivity dataset stored in the
        analysis h5py files.  This will be used whenever attempting to plot or cut on impulsivity values.
    time_delays_dset_key : str
        This is a string that must correspond to a specific stored and precalculated time delay dataset stored in the
        analysis h5py files.  This will be used whenever attempting to plot or cut on time delay values.
    map_dset_key : str
        This is a string that must correspond to a specific stored and precalculated map/alt-az dataset stored in the
        analysis h5py files.  This will be used whenever attempting to plot or cut on map/alt-az values.
    curve_choice : int
        Which curve/cr template you wish to use when loading the template search results from the cr template search.
        Currently the only option is 0, which corresponds to a simple bi-polar delta function convolved with the known
        BEACON responses. 
    trigger_types : list of ints
        The trigger types you want to include in each plot.  This may be later done on a function by function basis, but
        for now is called at the beginning. 
    included_antennas : list of ints
        List should contain list of antennas trusted for this particular run.  This will be used in certain cuts which
        may look at the max or average values of a certain polarization.  If the antenna is not included on this list
        then it will not be included in those calculations.  This does not apply to any precalculated values that already
        washed out antenna specific information.  This can also be used to investigate particular antenna pairings by
        leaving only them remaining.  
    cr_template_n_bins_h : int
        The number of bins in the x dimension of the cr template search plot.
    cr_template_n_bins_v : int
        The number of bins in the y dimension of the cr template search plot.
    impulsivity_hv_n_bins : int
        The number of bins in the impulsivity_hv plot.
    std_n_bins_h :
        The number of bins for plotting std of the h antennas.
    std_n_bins_v :
        The number of bins for plotting std of the v antennas.
    max_std_val :
        The max bin edge value for plotting std on both antennas.  Default is None.  If None is given then this will be
        automatically calculated (though likely too high).
    p2p_n_bins_h :
        The number of bins for plotting p2p of the h antennas.
    p2p_n_bins_v :
        The number of bins for plotting p2p of the v antennas.
    max_p2p_val :
        The max bin edge value for plotting p2p on both antennas.  Default is 128.  If None is given then this will be
        automatically calculated.
    snr_n_bins_h :
        The number of bins for plotting snr of the h antennas.
    snr_n_bins_v :
        The number of bins for plotting snr of the v antennas.
    max_snr_val :
        The max bin edge value for plotting snr on both antennas.  Default is None.  If None is given then this will be
        automatically calculated (though likely too high).
    include_test_roi :
        This will include test regions of interest that are more for testing the class itself. 
    '''
    def __init__(self,  runs, impulsivity_dset_key, time_delays_dset_key, map_dset_key, remove_incomplete_runs=True, **kwargs):
        try:
            self.conference_mode = False #Enable to apply any temporary adjustments such as fontsizes or title labels. 
            self.roi = {}
            self.data_slicers = []
            self.runs = []#numpy.sort(runs).astype(int)
            self.cor = None


            try:
                #Angular ranges are handled such that their bin centers are the same as the values sampled by the corrolator class given the same min, max, and n.  
                dataSlicerSingleRun.n_phi = kwargs['n_phi']
                dataSlicerSingleRun.range_phi_deg = numpy.asarray(kwargs['range_phi_deg'])
                dphi = (max(dataSlicerSingleRun.range_phi_deg) - min(dataSlicerSingleRun.range_phi_deg)) / (dataSlicerSingleRun.n_phi - 1)
                dataSlicerSingleRun.phi_edges = numpy.arange(min(dataSlicerSingleRun.range_phi_deg),max(dataSlicerSingleRun.range_phi_deg) + 2*dphi, dphi) - dphi/2.0
                dataSlicerSingleRun.phi_centers = 0.5*(dataSlicerSingleRun.phi_edges[1:]+dataSlicerSingleRun.phi_edges[:-1])

                dataSlicerSingleRun.n_theta = kwargs['n_theta']
                dataSlicerSingleRun.range_theta_deg = numpy.asarray(kwargs['range_theta_deg'])
                dtheta = (max(dataSlicerSingleRun.range_theta_deg) - min(dataSlicerSingleRun.range_theta_deg)) / (dataSlicerSingleRun.n_theta - 1)
                dataSlicerSingleRun.theta_edges = numpy.arange(min(dataSlicerSingleRun.range_theta_deg),max(dataSlicerSingleRun.range_theta_deg) + 2*dtheta, dtheta) - dtheta/2.0
                dataSlicerSingleRun.theta_centers = 0.5*(dataSlicerSingleRun.theta_edges[1:]+dataSlicerSingleRun.theta_edges[:-1])

                dataSlicerSingleRun.elevation_edges = 90 - dataSlicerSingleRun.theta_edges
                dataSlicerSingleRun.elevation_centers = 0.5*(dataSlicerSingleRun.elevation_edges[1:]+dataSlicerSingleRun.elevation_edges[:-1])
                skip_common_setup=True
                print('WARNING! dataSlicer has just set class attributes for dataSlicerSingleRun')
            except Exception as e:
                skip_common_setup=False
                print(e)

            for run in numpy.sort(runs).astype(int):
                try:
                    reader = Reader(raw_datapath,run)
                    if reader.failed_setup == False:
                        ds = dataSlicerSingleRun(reader,impulsivity_dset_key, time_delays_dset_key, map_dset_key, skip_common_setup=skip_common_setup, **kwargs)
                        can_open = ds.openSuccess()
                        if can_open == True:
                            #If above fails then it won't be appended to either runs or data slicers.
                            self.data_slicers.append(ds)
                            self.runs.append(run)
                except Exception as e:
                    print('Error loading dataSlicer for run %i, excluding.  Error:'%run)
                    print(e)

            self.data_slicers = numpy.array(self.data_slicers)
            self.runs = numpy.asarray(self.runs)
            cut = numpy.array([ds.dsets_present*ds.openSuccess() for ds in self.data_slicers])

            #self.data_slicers = numpy.array([dataSlicerSingleRun(Reader(raw_datapath,run),impulsivity_dset_key, time_delays_dset_key, map_dset_key, **kwargs) for run in self.runs])
            if len(self.runs) > 0:
                self.trigger_types = self.data_slicers[0].trigger_types
            else:
                self.trigger_types = [1,2,3] #just a place holder, not runs loaded. 

            if remove_incomplete_runs:
                self.removeIncompleteDataSlicers()

            if len(self.runs) == 0:
                print('\nWARNING!!! No runs worked on dataSlicer preperations.\n')

            if skip_common_setup:
                print('WARNING! dataSlicer has just set class attributes for dataSlicerSingleRun')

            #self.checkForComplementaryBothMapDatasets(verbose=False)

            print('\ndataSlicer Peparations Complete.  Excluding Runs:')
            print(numpy.array(runs)[~numpy.isin(runs,self.runs)])
            print('\n')
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def prepareCorrelator(self):
        '''
        Will check if a Correlator class already exists for this data slicers run.  If it does not then one will be
        generated.

        Note that Correlator makes it's own FFTPrep class. 

        This is mostly hardcoded.  If you want higher control then make your own correlator and assign it to this object.
        '''
        if self.cor is None:
            #Using the values that are commonly used in rf_bg_search.py
            crit_freq_low_pass_MHz = 85
            low_pass_filter_order = 6

            crit_freq_high_pass_MHz = 25
            high_pass_filter_order = 8

            sine_subtract = True
            sine_subtract_min_freq_GHz = 0.00
            sine_subtract_max_freq_GHz = 0.25
            sine_subtract_percent = 0.03

            apply_phase_response = True

            map_resolution_theta = 0.25 #degrees
            min_theta   = 0
            max_theta   = 120
            n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

            map_resolution_phi = 0.1 #degrees
            min_phi     = -90#-180
            max_phi     = 90#180
            n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

            upsample = 2**14 #Just upsample in this case, Reduced to 2**14 when the waveform length was reduced, to maintain same time precision with faster execution.
            max_method = 0

            waveform_index_range = info.returnDefaultWaveformIndexRange()
            map_source_distance_m = info.returnDefaultSourceDistance()

            self.cor = Correlator(self.data_slicers[0].reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=True, sine_subtract=sine_subtract, deploy_index=self.data_slicers[0].map_deploy_index, map_source_distance_m=map_source_distance_m,notch_tv=True,misc_notches=True)
            if sine_subtract:
                self.cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

    def delCorrelator(self):
        '''
        Deletes the currently saved correlator. 
        '''
        if hasattr(self, 'cor'):
            del self.cor
        self.cor = None

    def removeIncompleteDataSlicers(self):
        '''
        This will loop over the data slicers and remove any that do not have all required datasets.
        '''
        try:
            cut = numpy.array([ds.dsets_present for ds in self.data_slicers])
            if sum(~cut) > 0:
                print('Removing the following runs for not having all required datasets:')
                print(self.runs[~cut])

            self.runs = self.runs[cut]
            self.data_slicers = self.data_slicers[cut]
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def returnMaskedArray(self,*args,**kwargs):
        '''
        Uses the returnMaskedArray function from the dataSlicerSingleRun class.
        '''
        return self.data_slicers[0].returnMaskedArray(*args,**kwargs)

    def returnEventsAWithoutB(self, eventids_dict_A, eventids_dict_B):
        '''
        Given 2 eventid dictionaries, this will return a third dictionary containing all eventids in A that are not
        contained in B.

        '''
        try:
            runs_A = numpy.asarray(list(eventids_dict_A.keys()))
            runs_B = numpy.asarray(list(eventids_dict_B.keys()))
            eventids_dict_C = {}
            for run in runs_A:
                if run in runs_B:
                    eventids_dict_C[run] = eventids_dict_A[run][~numpy.isin(eventids_dict_A[run],eventids_dict_B[run])] #return events from A that are not in B
                else:
                    eventids_dict_C[run] = eventids_dict_A[run] #Return all from A
                eventids_dict_C[run] = numpy.sort(numpy.unique(eventids_dict_C[run])) 
            return eventids_dict_C
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def removeEmptyRunsFromDict(self, eventids_dict):
        '''
        '''
        try:
            runs = numpy.asarray(list(eventids_dict.keys()))
            eventids_dict_out = {}
            for run in runs:
                if len(eventids_dict[run]) > 0:
                    eventids_dict_out[run] = eventids_dict[run]
            return eventids_dict_out
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def returnCommonEvents(self, eventids_dict_A, eventids_dict_B):
        '''
        Given 2 eventid dictionaries, this will return a third dictionary containing all eventids that exist in  both 
        sets.
        '''
        try:
            runs_A = numpy.asarray(list(eventids_dict_A.keys()))
            runs_B = numpy.asarray(list(eventids_dict_B.keys()))
            eventids_dict_C = {}
            for run in runs_A:
                if run in runs_B:
                    eventids_dict_C[run] = numpy.sort(numpy.unique(eventids_dict_A[run][numpy.isin(eventids_dict_A[run],eventids_dict_B[run])])) #Add events from A that are in B.               
            return eventids_dict_C
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def returnUniqueEvents(self, eventids_dict_A, eventids_dict_B):
        '''
        Given 2 eventid dictionaries, this will return a third dictionary containing all eventids from both sets, but
        with no duplicates.
        '''
        try:
            runs_A = numpy.asarray(list(eventids_dict_A.keys()))
            runs_B = numpy.asarray(list(eventids_dict_B.keys()))
            runs = numpy.unique(numpy.append(runs_A,runs_B))
            eventids_dict_C = {}
            for run in runs:
                eventids_dict_C[run] = numpy.array([])
                if run in runs_A:
                    eventids_dict_C[run] = numpy.append(eventids_dict_C[run],eventids_dict_A[run])
                if run in runs_B:
                    eventids_dict_C[run] = numpy.append(eventids_dict_C[run],eventids_dict_B[run])
                eventids_dict_C[run] = numpy.sort(numpy.unique(eventids_dict_C[run]))
            return eventids_dict_C
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def concatenateParamDict(self, param_data_dict):
        '''
        Given a dictionary (keys indicating run number) for a parameter of data.  This will turn that into a 1d array.
        Useful for histogramming when run information is unimportant.
        '''
        try:
            data = []
            for key, _data in param_data_dict.items():
                if _data is not None:
                    data.append(_data)
                else:
                    print('Skipping %s, no data present for this key.'%(key))
            return numpy.concatenate(data)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def printKnownParamKeys(self):
        return self.data_slicers[0].printKnownParamKeys()
    def checkForComplementaryBothMapDatasets(self, verbose=False):
        for run_index, run in enumerate(self.runs):
            if verbose:
                print('Run %i'%run)
            self.data_slicers[run_index].checkForComplementaryBothMapDatasets()
    def printDatasets(self, verbose=False):
        for run_index, run in enumerate(self.runs):
            if verbose:
                print('Run %i'%run)
            self.data_slicers[run_index].printDatasets()
    def printSampleROI(self, verbose=False):
        return self.data_slicers[0].printDatasets
    def addROI(self, roi_key, roi_dict, verbose=False):
        '''
        Note that cuts are conducted in the order the are supplied to the ROI definition.
        '''
        self.roi[roi_key] = roi_dict
        self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, len(list(self.roi.keys())))]
        for run_index, run in enumerate(self.runs):
            if verbose:
                print('Run %i'%run)
            self.data_slicers[run_index].addROI(roi_key, roi_dict)
    def resetAllROI(self, verbose=False):
        for run_index, run in enumerate(self.runs):
            if verbose:
                print('Run %i'%run)
            self.data_slicers[run_index].resetAllROI()
        self.roi = {}
        self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, len(list(self.roi.keys())))]
    def getEventidsFromTriggerType(self, trigger_types=None, verbose=False):
        eventids_dict = {}
        for run_index, run in enumerate(self.runs):
            if verbose:
                print('Run %i'%run)
            eventids_dict[run] = self.data_slicers[run_index].getEventidsFromTriggerType(trigger_types=trigger_types)
        return eventids_dict
    def getDataFromParam(self, eventids_dict, param_key, verbose=False):
        '''
        Key should be run number as int.  Will return data for all events given in the eventids_dict.
        '''
        try:
            data = {}
            for run_index, run in enumerate(self.runs):
                if run in list(eventids_dict.keys()):
                    if verbose:
                        print('Run %i'%run)
                    eventids = eventids_dict[run]
                    data[run] = self.data_slicers[run_index].getDataFromParam(eventids, param_key)
            return data
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def getCutsFromROI(self,roi_key,eventids_dict=None,load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False):
        '''
        Note that cuts are conducted in the order the are supplied to the ROI definition.
        '''
        if eventids_dict is None:
            eventids_dict = {}
            all_runs = True
        else:
            all_runs = False

        if return_successive_cut_counts == True:
            successive_cut_counts = OrderedDict()
        if return_total_cut_counts == True:
            total_cut_counts = OrderedDict()

        for run_index, run in enumerate(self.runs):
            if verbose:
                sys.stdout.write('Run %i/%i\r'%(run_index+1,len(self.runs)))
                sys.stdout.flush()

            if run in eventids_dict.keys():
                if return_successive_cut_counts and return_total_cut_counts:
                    eventids_dict[run], _successive_cut_counts, _total_cut_counts    = self.data_slicers[run_index].getCutsFromROI(roi_key,eventids=eventids_dict[run],load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
                elif return_successive_cut_counts:
                    eventids_dict[run], _successive_cut_counts                       = self.data_slicers[run_index].getCutsFromROI(roi_key,eventids=eventids_dict[run],load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
                elif return_total_cut_counts:
                    eventids_dict[run], _total_cut_counts                            = self.data_slicers[run_index].getCutsFromROI(roi_key,eventids=eventids_dict[run],load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
                else:
                    eventids_dict[run]                                               = self.data_slicers[run_index].getCutsFromROI(roi_key,eventids=eventids_dict[run],load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)

            elif all_runs == True:
                if return_successive_cut_counts and return_total_cut_counts:
                    eventids_dict[run], _successive_cut_counts, _total_cut_counts    = self.data_slicers[run_index].getCutsFromROI(roi_key, load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
                elif return_successive_cut_counts:
                    eventids_dict[run], _successive_cut_counts                       = self.data_slicers[run_index].getCutsFromROI(roi_key, load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
                elif return_total_cut_counts:
                    eventids_dict[run], _total_cut_counts                            = self.data_slicers[run_index].getCutsFromROI(roi_key, load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
                else:
                    eventids_dict[run]                                               = self.data_slicers[run_index].getCutsFromROI(roi_key, load=load,save=save, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
            else:
                eventids_dict[run] = numpy.array([])

            if return_successive_cut_counts:
                for key in list(_successive_cut_counts.keys()):
                    if key in list(successive_cut_counts.keys()):
                        successive_cut_counts[key] += _successive_cut_counts[key]
                    else:
                        successive_cut_counts[key] = _successive_cut_counts[key]
            if return_total_cut_counts:
                for key in list(_total_cut_counts.keys()):
                    if key in list(total_cut_counts.keys()):
                        total_cut_counts[key] += _total_cut_counts[key]
                    else:
                        total_cut_counts[key] = _total_cut_counts[key]

        if verbose == True:
            if return_successive_cut_counts or return_total_cut_counts:
                print('Cut breakdown for ROI %s'%(roi_key))
                if return_successive_cut_counts and return_total_cut_counts:
                    for key in list(successive_cut_counts.keys()):
                        #self.roi[roi_key][key]
                        if key == 'initial':
                            print('Initial Event Count is %i'%(successive_cut_counts[key]))
                        else:
                            print('\nRemaining Events After Step %s is %i'%(key, successive_cut_counts[key]))
                            print('%0.3f%% events then cut by %s with bounds %s'%(100*(previous_count-successive_cut_counts[key])/previous_count , key, str(self.roi[roi_key][key])))
                            print('%0.3f%% of initial events would be cut by %s with bounds %s'%(100*(total_cut_counts['initial']-total_cut_counts[key])/total_cut_counts['initial'] , key, str(self.roi[roi_key][key])))
                        previous_count = successive_cut_counts[key]

                elif return_successive_cut_counts:
                    for key in list(successive_cut_counts.keys()):
                        if key == 'initial':
                            print('Initial Event Count is %i'%(successive_cut_counts[key]))
                        else:
                            print('\nRemaining Events After Step %s is %i'%(key, successive_cut_counts[key]))
                            print('%0.3f%% events then cut by %s with bounds %s'%(100*(previous_count-successive_cut_counts[key])/previous_count , key, str(self.roi[roi_key][key])))
                        previous_count = successive_cut_counts[key]

                elif return_total_cut_counts:
                    for key in list(total_cut_counts.keys()):
                        if key == 'initial':
                            print('Initial Event Count is %i'%(total_cut_counts[key]))
                        else:
                            print('%0.3f%% of initial events would be cut by %s with bounds %s'%(100*(total_cut_counts['initial']-total_cut_counts[key])/total_cut_counts['initial'] , key, str(self.roi[roi_key][key])))
                print('\n')
        if return_successive_cut_counts and return_total_cut_counts:
            return eventids_dict, successive_cut_counts, total_cut_counts
        elif return_successive_cut_counts:
            return eventids_dict, successive_cut_counts
        elif return_total_cut_counts:
            return eventids_dict, total_cut_counts
        else:
            return eventids_dict

    def setCurrentPlotBins(self, main_param_key_x, main_param_key_y, eventids_dict):
        '''
        Loops over all saved dataSlicerSingleRun, gets their bounds, then makes sure all match using the largest windows.
        '''
        try:
            x_bin_min = 1e12
            y_bin_min = 1e12
            x_bin_max = -1e12
            y_bin_max = -1e12
            len_x = []
            len_y = []

            if eventids_dict is None:
                eventids_dict = {}
                for run_index, run in enumerate(self.runs):
                    eventids_dict[run] = self.data_slicers[run_index].getEventidsFromTriggerType()

            for run_index, run in enumerate(self.runs):
                if run in list(eventids_dict.keys()):
                    self.data_slicers[run_index].setCurrentPlotBins(main_param_key_x, main_param_key_y, eventids_dict[run])

                    current_bin_edges_x, current_label_x = self.data_slicers[run_index].getSingleParamPlotBins(main_param_key_x, eventids_dict[run])
                    if main_param_key_x == main_param_key_y:
                        current_bin_edges_y = current_bin_edges_x
                        current_label_y = current_label_x
                    else:
                        current_bin_edges_y, current_label_y = self.data_slicers[run_index].getSingleParamPlotBins(main_param_key_y, eventids_dict[run])

                    if min(current_bin_edges_x) < x_bin_min:
                        x_bin_min = min(current_bin_edges_x)

                    if min(current_bin_edges_y) < y_bin_min:
                        y_bin_min = min(current_bin_edges_y)

                    if max(current_bin_edges_x) > x_bin_max:
                        x_bin_max = max(current_bin_edges_x)

                    if max(current_bin_edges_y) > y_bin_max:
                        y_bin_max = max(current_bin_edges_y)


                    len_x.append(len(current_bin_edges_x))
                    len_y.append(len(current_bin_edges_y))

            self.current_label_x = current_label_x # Assuming same for all runs
            self.current_label_y = current_label_y # Assuming same for all runs

            self.current_bin_edges_x = numpy.linspace(x_bin_min,x_bin_max,int(numpy.mean(len_x)))
            self.current_bin_edges_y = numpy.linspace(y_bin_min,y_bin_max,int(numpy.mean(len_y)))

            #These are centered on what WOULD be a 2d histogram bin because the hist is being calculated seperate from the plot.  
            #import pdb; pdb.set_trace()
            self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y = numpy.meshgrid((self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2, (self.current_bin_edges_y[:-1] + self.current_bin_edges_y[1:]) / 2) # Use for contours
            self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y = numpy.meshgrid(self.current_bin_edges_x , self.current_bin_edges_y)#use for pcolormesh

            for ds in self.data_slicers:
                ds.current_bin_edges_x = self.current_bin_edges_x
                ds.current_bin_edges_y = self.current_bin_edges_y
                ds.current_bin_centers_mesh_x = self.current_bin_centers_mesh_x
                ds.current_bin_centers_mesh_v = self.current_bin_centers_mesh_y
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def get2dHistCounts(self, main_param_key_x, main_param_key_y, eventids_dict, set_bins=True,mask_top_N_bins=0,fill_value=0):
        try:
            if set_bins == True:
                self.setCurrentPlotBins(main_param_key_x,main_param_key_y,eventids_dict)
            print('\tGetting counts from 2dhists for %s v.s. %s'%(main_param_key_x,main_param_key_y))
            counts = numpy.zeros_like(self.current_bin_centers_mesh_x) #Need to double check this works. 
            for key, item in eventids_dict.items():
                #Loop to save memory, only need to store one runs worth of values at a time. 
                param_x = self.getDataFromParam({key:item}, main_param_key_x)
                param_y = self.getDataFromParam({key:item}, main_param_key_y)
                counts += numpy.histogram2d(self.concatenateParamDict(param_x), self.concatenateParamDict(param_y), bins = [self.current_bin_edges_x,self.current_bin_edges_y])[0].T #Outside of file being open 

            if mask_top_N_bins > 0:
                counts = self.returnMaskedArray(counts,mask_top_N_bins,fill_value=fill_value)
            return counts
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def get1dHistCounts(self, main_param_key, eventids_dict, set_bins=True):
        try:
            if set_bins == True:
                self.setCurrentPlotBins(main_param_key,main_param_key,eventids_dict)
            print('\tGetting counts from 1dhists for %s'%(main_param_key))
            counts = numpy.zeros(len(self.current_bin_edges_x) - 1)
            for key, item in eventids_dict.items():
                #Loop to save memory, only need to store one runs worth of values at a time. 
                param = self.getDataFromParam({key:item}, main_param_key)
                counts += numpy.histogram(self.concatenateParamDict(param), bins = self.current_bin_edges_x)[0]
            return counts
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def plot2dHist(self, main_param_key_x,  main_param_key_y, eventids_dict, title=None,cmap='coolwarm', lognorm=True, return_counts=False, mask_top_N_bins=0, fill_value=0):
        '''
        This is meant to be a function the plot corresponding to the main parameter, and will plot the same quantity 
        (corresponding to main_param_key) with just events corresponding to the cut being used.  This subset will show
        up as a contour on the plot.  
        '''
        try:
            #Should make eventids a self.eventids so I don't need to call this every time.
            counts = self.get2dHistCounts(main_param_key_x,main_param_key_y,eventids_dict,set_bins=True,mask_top_N_bins=mask_top_N_bins, fill_value=fill_value) #set_bins should only be called on first call, not on contours.
            _fig, _ax = plt.subplots()

            if title is None:
                if numpy.all(numpy.diff(list(eventids_dict.keys()))) == 1:
                    title = '%s, Runs = %i-%i\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,list(eventids_dict.keys())[0],list(eventids_dict.keys())[-1],str(self.trigger_types))
                else:
                    title = '%s, Runs = %s\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,str(list(eventids_dict.keys())),str(self.trigger_types))
                if mask_top_N_bins > 0:
                    title += ', masked_bins = %i'%masked_bins
            plt.title(title)
            if lognorm == True:
                _im = _ax.pcolormesh(self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y, counts,norm=colors.LogNorm(vmin=0.5, vmax=counts.max()),cmap=cmap)#cmap=plt.cm.coolwarm
            else:
                _im = _ax.pcolormesh(self.current_bin_edges_mesh_x, self.current_bin_edges_mesh_y, counts,cmap=cmap)#cmap=plt.cm.coolwarm
            if 'theta_best_' in main_param_key_y:
                _ax.invert_yaxis()
            if True:
                if numpy.logical_or(numpy.logical_and('phi_best_' in main_param_key_x,'phi_best_' in main_param_key_y),numpy.logical_or(numpy.logical_and('theta_best_' in main_param_key_x,'theta_best_' in main_param_key_y),numpy.logical_and('elevation_best_' in main_param_key_x,'elevation_best_' in main_param_key_y))):
                    plt.plot(self.current_bin_centers_mesh_y[:,0],self.current_bin_centers_mesh_y[:,0],linewidth=1,linestyle='--',color='tab:gray',alpha=0.5)
                if numpy.logical_and('phi_best_' in main_param_key_x,numpy.logical_or('theta_best_' in main_param_key_y,'elevation_best_' in main_param_key_y)):
                    #Make cor to plot the array plane.
                    cor = Correlator(self.data_slicers[0].reader,  upsample=2**10, n_phi=720, n_theta=720, waveform_index_range=(None,None),crit_freq_low_pass_MHz=None, crit_freq_high_pass_MHz=None, low_pass_filter_order=None, high_pass_filter_order=None, plot_filter=False,apply_phase_response=False, tukey=False, sine_subtract=False, deploy_index=self.data_slicers[0].map_deploy_index) #only for array plane
                    if numpy.logical_and(main_param_key_x.replace('_allsky','').replace('_abovehorizon','').replace('_belowhorizon','').split('_')[-1] == 'h', main_param_key_y.replace('_allsky','').replace('_abovehorizon','').replace('_belowhorizon','').split('_')[-1] == 'h'):
                        plane_xy = cor.getPlaneZenithCurves(cor.n_hpol.copy(), 'hpol', 90.0, azimuth_offset_deg=0.0)
                    elif numpy.logical_and(main_param_key_x.replace('_allsky','').replace('_abovehorizon','').replace('_belowhorizon','').split('_')[-1] == 'v', main_param_key_y.replace('_allsky','').replace('_abovehorizon','').replace('_belowhorizon','').split('_')[-1] == 'v'):
                        plane_xy = cor.getPlaneZenithCurves(cor.n_vpol.copy(), 'vpol', 90.0, azimuth_offset_deg=0.0)
                    else:
                        plane_xy = None
                    if plane_xy is not None:
                        if 'elevation_best_' in main_param_key_y:
                            plane_xy[1] = 90.0 - plane_xy[1]
                        plt.plot(plane_xy[0], plane_xy[1],linestyle='-',linewidth=1,color='k')
                        plt.xlim([-180,180])
                    if self.conference_mode:
                        ticks_deg = numpy.array([-60,-40,-30,-15,0,15,30,45,60,75])
                        plt.yticks(ticks_deg)
                        x = plane_xy[0]
                        y1 = plane_xy[1]
                        y2 = -90 * numpy.ones_like(plane_xy[0])#lower_plane_xy[1]
                        _ax.fill_between(x, y1, y2, where=y2 <= y1,facecolor='#9DC3E6', interpolate=True,alpha=1)#'#EEC6C7'
                        plt.ylim(min(y1) - 5, 30)
                        plt.xlim(-90,90)

            plt.xlabel(self.current_label_x)
            plt.ylabel(self.current_label_y)
            plt.grid(which='both', axis='both')
            _ax.minorticks_on()
            _ax.grid(b=True, which='major', color='k', linestyle='-')
            _ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            if self.conference_mode:
                plt.title('')
                plt.tight_layout()
                if numpy.logical_and('phi_best_' in main_param_key_x,numpy.logical_or('theta_best_' in main_param_key_y,'elevation_best_' in main_param_key_y)):
                    plt.ylabel('Elevation (deg)', fontsize=24)
                    plt.xlabel('Azimuth (Deg)\n(East = 0 deg, North = 90 deg)', fontsize=24)
                elif numpy.logical_and('std_' in main_param_key_x,'std_' in main_param_key_y):
                    plt.title('')
                    plt.xlabel(self.current_label_x.replace('hpol','HPol'), fontsize=24)
                    plt.ylabel(self.current_label_y.replace('vpol','VPol'), fontsize=24)
                    plt.xlim(0,9)
                    plt.ylim(0,9)
                elif numpy.logical_and('p2p_' in main_param_key_x,'p2p_' in main_param_key_y):
                    plt.title('')
                    plt.xlabel(self.current_label_x.replace('hpol','HPol'), fontsize=24)
                    plt.ylabel(self.current_label_y.replace('vpol','VPol'), fontsize=24)
                    plt.xlim(20,128)
                _fig.canvas.set_window_title(self.current_label_x)

            try:
                cbar = _fig.colorbar(_im)
                cbar.set_label('Counts')
            except Exception as e:
                print('Error in colorbar, often caused by no events.')
                print(e)

            if return_counts:
                return _fig, _ax, counts
            else:
                return _fig, _ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def plot1dHist(self, main_param_key, eventids_dict, title=None, lognorm=True, return_counts=False):
        '''
        This is meant to be a function the plot corresponding to the main parameter, and will plot the same quantity 
        (corresponding to main_param_key) with just events corresponding to the cut being used.  This subset will show
        up as a contour on the plot.  
        '''
        try:
            #Should make eventids a self.eventids so I don't need to call this every time.
            counts = self.get1dHistCounts(main_param_key,eventids_dict,set_bins=True) #set_bins should only be called on first call, not on contours.
            _fig, _ax = plt.subplots()

            if title is None:
                if numpy.all(numpy.diff(list(eventids_dict.keys()))) == 1:
                    title = '%s, Runs = %i-%i\nIncluded Triggers = %s'%(main_param_key,list(eventids_dict.keys())[0],list(eventids_dict.keys())[-1],str(self.trigger_types))
                else:
                    title = '%s, Runs = %s\nIncluded Triggers = %s'%(main_param_key,str(list(eventids_dict.keys())),str(self.trigger_types))
            plt.title(title)

            bin_centers = (self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2
            bin_width = self.current_bin_edges_x[1] - self.current_bin_edges_x[0]

            plt.bar(bin_centers, counts, width=bin_width)

            if lognorm == True:
                _ax.set_yscale('log')
            plt.xlabel(self.current_label_x)
            plt.ylabel(self.current_label_y)
            plt.grid(which='both', axis='both')
            _ax.minorticks_on()
            _ax.grid(b=True, which='major', color='k', linestyle='-')
            _ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            if return_counts:
                return _fig, _ax, counts
            else:
                return _fig, _ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addContour(self, ax, main_param_key_x, main_param_key_y, contour_eventids_dict, contour_color, n_contour=5, alpha=0.85, log_contour=True, mask=None, fill_value=None):
        '''
        Given the plot made from plot2dHist, this will add contours to it for the events specified by contour_eventids.
        This assumes that setCurrentPlotBins has already been called by plot2dHist. 

        Parameters
        ----------
        ax : matplotlib axes
            This should be the output axes from plot2dHist.
        main_param_key : str
            This should match the one used for plt2dHist, as it is used to determine the xy coords of the contour and
            what parameter should be calculated (or loaded) for each selected eventid.
        contour_eventids : numpy.ndarray of int
            This should be a list of eventids that meet the cut you wish to plot.
        contour_color : tuple
            The color which you want the contour to be plotted on top of the 
        n_contour : int
            The number of contours to plot.  This will create contours ranging from the min to max count value, with n+1
            lines.  The lowest line will NOT be plotted (thus plotting n total), because this is often 0 and clutters
            the plot.
        log_contour : bool
            If True then the Locator used in matplotlib.axes.Axes.contour will be matplotlib.ticker.LogLocator(), 
            rather than the default contour behaviour of matplotlib.ticker.MaxNLocator.

        Returns
        -------
        ax : matplotlib axes
            This is the updated axes that the contour was added to.
        cs : QuadContourSet
            This is the contour object, that can be used for adding labels to the legend for this contour.
        '''
        try:
            counts = self.get2dHistCounts(main_param_key_x, main_param_key_y, contour_eventids_dict, set_bins=False)

            if mask is not None:
                #Want mask to be applied on top N bins of original data, not the countoured data.  So mask is given.
                if fill_value is not None:
                    fill_value = numpy.ma.default_fill_value(counts.flat[0])
                counts = numpy.ma.masked_array(counts, mask=mask, fill_value=fill_value)

            if log_contour:
                #Log?
                #locator = ticker.LogLocator()
                levels = numpy.ceil(numpy.logspace(0,numpy.log10(numpy.max(counts)),n_contour))[1:n_contour+1] #Not plotting bottom contour because it is often background and clutters plot.
            else:
                #Linear?
                #locator = ticker.MaxNLocator()
                levels = numpy.ceil(numpy.linspace(0,numpy.max(counts),n_contour))[1:n_contour+1] #Not plotting bottom contour because it is often background and clutters plot.

            levels = numpy.unique(levels) #Covers edge case of very low counts resulting in degenerate contour labels (Must be increasing)

            # The below code was intended to make outer contours more transparent, but it produced differing results from plotting contour once, and I don't know why, so I am just commenting it out :/
            # import pdb; pdb.set_trace()
            # print(levels)
            # for index, level in enumerate(levels[::-1]):
            #     # if index != len(levels)-1:
            #     #     continue
            #     _alpha = alpha*( 0.7*((index+1)/len(levels)) + 0.3) #Highest level contour should be specified alpha, others are dimmer to make it clear where high concentration is.  Doesn't let opacity go below 0.3.
            #     # print(_alpha)
            #     print(level)
            #     cs = ax.contour(self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y, counts, colors=[contour_color],levels=[level],alpha=_alpha, antialiased=True)
            # print(levels)
            if numpy.shape(counts != numpy.shape(self.current_bin_centers_mesh_x)):
                import pdb; pdb.set_trace()
            cs = ax.contour(self.current_bin_centers_mesh_x, self.current_bin_centers_mesh_y, counts, colors=[contour_color],levels=levels,alpha=alpha, antialiased=True)

            return ax, cs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def plotROI2dHist(self, main_param_key_x, main_param_key_y, eventids_dict=None, cmap='coolwarm', return_counts=False, include_roi=True, lognorm=True, mask_top_N_bins=0, fill_value=0, suppress_legend=False):
        '''
        This is the "do it all" function.  Given the parameter it will plot the 2dhist of the corresponding param by
        calling plot2dHist.  It will then plot the contours for each ROI on top.  It will do so assuming that each 
        ROI has a box cut in self.roi for EACH parameter.  I.e. it expects the # of listed entries in each ROI to be
        the same, and it will add a contour for the eventids that pass ALL cuts for that specific roi. 

        If eventids is given then then those events will be used to create the plot, and the trigger type cut will be ignored.
        '''
        try:
            if eventids_dict is None:
                eventids_dict = {}
                for run_index, run in enumerate(self.runs):
                    eventids_dict[run] = self.data_slicers[run_index].getEventidsFromTriggerType()
                #The given eventids don't necessarily match the default cut, and thus shouldn't be labeled as such in the title.
                if numpy.all(numpy.diff(list(eventids_dict.keys()))) == 1:
                    title = '%s, Runs = %i-%i\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,list(eventids_dict.keys())[0],list(eventids_dict.keys())[-1],str(self.trigger_types))
                else:
                    title = '%s, Runs = %s\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,str(list(eventids_dict.keys())),str(self.trigger_types))
            else:
                if numpy.all(numpy.diff(list(eventids_dict.keys()))) == 1:
                    title = '%s, Runs = %i-%i'%(main_param_key_x + ' vs ' + main_param_key_y,list(eventids_dict.keys())[0],list(eventids_dict.keys())[-1])
                else:
                    title = '%s, Runs = %s'%(main_param_key_x + ' vs ' + main_param_key_y,str(list(eventids_dict.keys())))
            fig, ax, counts = self.plot2dHist(main_param_key_x, main_param_key_y, eventids_dict, title=title, cmap=cmap, lognorm=lognorm, return_counts=True, mask_top_N_bins=mask_top_N_bins, fill_value=fill_value) #prepares binning, must be called early (before addContour)
            #these few lines below this should be used for adding contours to the map. 
            if include_roi:
                legend_properties = []
                legend_labels = []

                for roi_index, roi_key in enumerate(list(self.roi.keys())):
                    contour_eventids_dict = self.getCutsFromROI(roi_key)
                    for run_index, run in enumerate(self.runs):
                        if run in list(eventids_dict.keys()): 
                            contour_eventids_dict[run] = numpy.intersect1d(contour_eventids_dict[run],eventids_dict[run]) #getCutsFromROI doesn't account for eventids, this makes it only those that are in ROI and given.
                        else:
                            del contour_eventids_dict[run]
                    if type(counts) == 'numpy.ma.core.MaskedArray':
                        ax, cs = self.addContour(ax, main_param_key_x, main_param_key_y, contour_eventids_dict, self.roi_colors[roi_index], n_contour=6, mask=counts.mask, fill_value=fill_value)
                    else:
                        ax, cs = self.addContour(ax, main_param_key_x, main_param_key_y, contour_eventids_dict, self.roi_colors[roi_index], n_contour=6)
                    legend_properties.append(cs.legend_elements()[0][0])
                    if self.conference_mode == False:
                        legend_labels.append('roi %i: %s'%(roi_index, roi_key))
                    else:
                        legend_labels.append('%s'%(roi_key))
                if suppress_legend == False:
                    plt.legend(legend_properties,legend_labels,loc='upper left')

            if return_counts == True:
                return fig, ax, counts
            else:
                return fig, ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def plotTimeDelayHist(self,include_roi=True):
        '''
        This will plot 1d histograms for all time delays, plotted in a grid.  It will make a seperate plot
        for both hpol and vpol (total of 12 subplots).
        '''
        try:
            eventids_dict = {}
            for run_index, run in enumerate(self.runs):
                eventids_dict[run] = self.data_slicers[run_index].getEventidsFromTriggerType()
                

            if include_roi:
                roi_eventids_dict = {}
                for roi_index, roi_key in enumerate(list(self.roi.keys())):
                    roi_eventids_dict[roi_key] = self.getCutsFromROI(roi_key)

            for mode in ['h','v']:
                fig = plt.figure()
                fig.canvas.set_window_title('%spol Time Delays'%(mode.title()))
                for baseline_index, baseline in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
                    param_key = 'time_delay_%isubtract%i_%s'%(baseline[0],baseline[1],mode)
                    self.setCurrentPlotBins(param_key, param_key, eventids_dict)
                    label = self.current_label_x
                    param = concatenateParamDict(self.getDataFromParam(eventids_dict, param_key))
                    ax = plt.subplot(3,2,baseline_index+1)
                    if include_roi:
                        alpha = 0.5
                    else:
                        alpha = 1.0

                    plt.hist(param, bins = self.current_bin_edges_x,label=label,alpha=alpha)
                    plt.minorticks_on()
                    plt.yscale('log', nonposy='clip')
                    plt.grid(b=True, which='major', color='k', linestyle='-')
                    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

                    if baseline_index in [4,5]:
                        plt.xlabel('Time Delay (ns)')
                    if baseline_index in [0,2,4]:
                        plt.ylabel('Counts')

                    if include_roi:
                        for roi_index, roi_key in enumerate(list(self.roi.keys())):
                            param = concatenateParamDict(self.getDataFromParam(roi_eventids_dict[roi_key], param_key))
                            c = numpy.copy(self.roi_colors[roi_index])
                            c[3] = 0.7 #setting alpha for fill but not for edge.
                            plt.hist(param, bins = self.current_bin_edges_x,label = 'roi %i: %s'%(roi_index, roi_key), color = c, edgecolor='black', linewidth=1)
                    plt.legend(fontsize=10)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            
    def trackROICounts(self, roi_keys=None,time_bin_width_s=60,plot_run_start_times=True):
        '''
        This will loop over the given runs, and count the number of events in the given run.  These will be plotted
        as a function of time.  If roi_keys is None then this will perform the calculation for ALL roi.  Otherwise
        it will only include roi specified in a list given by roi_keys
        '''
        try:
            if roi_keys is None:
                roi_keys = list(self.roi.keys())
            eventids_dict = {}
            for run_index, run in enumerate(self.runs):
                eventids_dict[run] = self.data_slicers[run_index].getEventidsFromTriggerType()
            self.setCurrentPlotBins('calibrated_trigtime', 'calibrated_trigtime', eventids_dict)

            bin_edges = numpy.arange(min(self.current_bin_edges_x),max(self.current_bin_edges_x)+time_bin_width_s,time_bin_width_s)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            roi_eventids_dict = {}
            for roi_index, roi_key in enumerate(list(self.roi.keys())):
                if roi_key in roi_keys:
                    roi_eventids_dict[roi_key] = self.getCutsFromROI(roi_key)

            fig = plt.figure()
            ax = plt.gca()
            if plot_run_start_times:
                max_counts = 0
            for roi_index, roi_key in enumerate(list(roi_eventids_dict.keys())):
                _eventids_dict = {}
                for run_index, run in enumerate(self.runs):
                    _eventids_dict[run] = roi_eventids_dict[roi_key][run][numpy.isin(roi_eventids_dict[roi_key][run],eventids_dict[run])] #Pull eventids in dict that are of the specified trigger types.
                event_times_dict = self.getDataFromParam(_eventids_dict, 'calibrated_trigtime', verbose=False)

                counts = numpy.zeros(len(bin_edges) - 1)
                for run_index, run in enumerate(self.runs):
                    counts += numpy.histogram(event_times_dict[run],bins=bin_edges)[0]

                if plot_run_start_times:
                    max_counts = max(max_counts, max(counts))
                plt.plot((bin_centers - min(bin_centers))/3600,counts,label=roi_key)

            if plot_run_start_times:
                for data_slicer in self.data_slicers:
                    t = (data_slicer.getDataFromParam([0],'calibrated_trigtime')[0] - min(bin_centers))/3600
                    plt.axvline(t,c='k',alpha=0.5)
                    plt.text(t,0.9*max_counts,str(data_slicer.reader.run),c='k',alpha=0.5,rotation=90)


            plt.legend()
            plt.xlabel('Calibrated Trigger Time (h)\nFrom Timestamp %0.2f s'%min(bin_centers))
            plt.ylabel('Binned Counts\n%0.2f s Bins'%time_bin_width_s)
            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            return fig, ax
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
    def getDataArrayFromParam(self, param_key, trigger_types=None, eventids_dict=None):
        '''
        This will return the data corresponding to param_key for all events specified by the given trigger types.  
        If trigger_types is None then the previously assigned trigger types will be used. 
        '''
        if eventids_dict is None:
            eventids_dict = self.getEventidsFromTriggerType(trigger_types=trigger_types)
        return self.concatenateParamDict(self.getDataFromParam(eventids_dict,param_key))

    def organizeEventDict(self, eventids_dict):
        '''
        Given an eventids dict this will turn it into an array with attributes: run, run_index, eventid.  This format
        is more useful for quickly iterating through events and having the associated index for use in the data slicer
        lists.
        '''
        event_dtype = numpy.dtype([('run','i'),('run_index','i'),('eventid','i')])
        out = numpy.zeros(len(self.concatenateParamDict(eventids_dict)),dtype=event_dtype)
        index = 0
        for run in list(eventids_dict.keys()):
            l = len(eventids_dict[run])
            if l > 0:
                run_index = int(numpy.where(self.runs == run)[0])
                out[index:index+l]['eventid'] = eventids_dict[run]
                out[index:index+l]['run'] = run
                out[index:index+l]['run_index'] = run_index
                index += l
        return out

    def updateEventInspect(self, run_index, eventid, mollweide=False):
        '''
        '''
        try:
            start_data = numpy.round(self.getSingleEventTableValues(self.table_params, run_index, eventid),decimals=3) #Meta information about the event that will be put in the table.
            self.mollweide = mollweide
            try:
                self.az_best = start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_choice']
                self.el_best = start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_choice']
                self.az_h = numpy.append(start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_h_allsky'] , self.az_best)
                self.el_h = numpy.append(start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_h_allsky'] , self.el_best)
                self.az_v = numpy.append(start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_v_allsky'] , self.az_best)
                self.el_v = numpy.append(start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_v_allsky'] , self.el_best)

                if self.show_all:
                    self.az_all = numpy.append(start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_all_allsky'], self.az_best)
                    self.el_all = numpy.append(start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_all_allsky'], self.el_best)
            except:
                self.az_h = start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_h_allsky']
                self.el_h = start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_h_allsky']
                self.az_v = start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_v_allsky']
                self.el_v = start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_v_allsky']
                if self.show_all:
                    self.az_all = start_data[numpy.array(list(self.table_params.keys())) == 'phi_best_all_allsky']
                    self.el_all = start_data[numpy.array(list(self.table_params.keys())) == 'elevation_best_all_allsky']

            self.inspector_mpl['current_table'] = list(zip(self.table_params.values(), start_data))

            self.inspector_mpl['fig1'].canvas.set_window_title('r%ie%i'%(self.runs[run_index],eventid))
            #Clear plot axes
            for key in list(self.inspector_mpl.keys()):
                if str(type(self.inspector_mpl[key])) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
                    if 'fig1_map_' in key:
                        self.inspector_mpl[key].clear()
                    else:
                        for artist in self.inspector_mpl[key].lines + self.inspector_mpl[key].collections:
                            artist.remove()
            #Plot Waveforms
            self.cor.setReader(self.data_slicers[run_index].reader, verbose=False)
            self.cor.reader.setEntry(eventid)

            #t = self.data_slicers[run_index].reader.t()
            #t = self.cor.times_resampled
            waveforms = self.cor.wf(eventid, numpy.array([0,1,2,3,4,5,6,7]),div_std=False,hilbert=False,apply_filter=True,tukey=True, sine_subtract=True)
            
            self.inspector_mpl['fig1_wf_0'].plot(self.cor.times_resampled, waveforms[0], c=self.mpl_colors[0])
            self.inspector_mpl['fig1_wf_0'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_1'].plot(self.cor.times_resampled, waveforms[1], c=self.mpl_colors[1])
            self.inspector_mpl['fig1_wf_1'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_2'].plot(self.cor.times_resampled, waveforms[2], c=self.mpl_colors[2])
            self.inspector_mpl['fig1_wf_2'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_3'].plot(self.cor.times_resampled, waveforms[3], c=self.mpl_colors[3])
            self.inspector_mpl['fig1_wf_3'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_4'].plot(self.cor.times_resampled, waveforms[4], c=self.mpl_colors[4])
            self.inspector_mpl['fig1_wf_4'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_5'].plot(self.cor.times_resampled, waveforms[5], c=self.mpl_colors[5])
            self.inspector_mpl['fig1_wf_5'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_6'].plot(self.cor.times_resampled, waveforms[6], c=self.mpl_colors[6])
            self.inspector_mpl['fig1_wf_6'].set_ylim(-70,70)
            self.inspector_mpl['fig1_wf_7'].plot(self.cor.times_resampled, waveforms[7], c=self.mpl_colors[7])
            self.inspector_mpl['fig1_wf_7'].set_ylim(-70,70)

            raw_t = self.data_slicers[run_index].reader.t()
            #Zoom in
            start = self.cor.prep.start_waveform_index
            stop = self.cor.prep.end_waveform_index+1
            self.inspector_mpl['fig1_wf_0'].set_xlim(min(raw_t[start:stop]), max(raw_t[start:stop])) #All others will follow

            #Plot Maps
            m, self.inspector_mpl['fig1'], self.inspector_mpl['fig1_map_h'] = self.cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=self.inspector_mpl['fig1_map_h'], plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=self.mollweide, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=90 - self.el_h, circle_az=self.az_h, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False)
            self.inspector_mpl['fig1_map_h'].set_xlim(self.cor.range_phi_deg)
            m, self.inspector_mpl['fig1'], self.inspector_mpl['fig1_map_v'] = self.cor.map(eventid, 'vpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=self.inspector_mpl['fig1_map_v'], plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=self.mollweide, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=90 - self.el_v, circle_az=self.az_v, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False)
            self.inspector_mpl['fig1_map_v'].set_xlim(self.cor.range_phi_deg)
            if self.show_all:
                m, self.inspector_mpl['fig1'], self.inspector_mpl['fig1_map_all'] = self.cor.map(eventid, 'all', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=self.inspector_mpl['fig1_map_all'], plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=self.mollweide, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=90 - self.el_all, circle_az=self.az_all, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False)
                self.inspector_mpl['fig1_map_all'].set_xlim(self.cor.range_phi_deg)
            #Plot Spectra
            sine_subtract = True
            for apply_filter, ax in [[False, self.inspector_mpl['fig1_spec_raw']], [True, self.inspector_mpl['fig1_spec_filt']]]:
                for channel in numpy.arange(8):
                    channel=int(channel)
                    wf = self.cor.prep.wf(channel,apply_filter=apply_filter,hilbert=False,tukey=apply_filter,sine_subtract=apply_filter and sine_subtract, return_sine_subtract_info=False)

                    freqs, spec_dbish, spec = self.cor.prep.rfftWrapper(raw_t[start:stop], wf)
                    ax.plot(freqs/1e6,spec_dbish/2.0,label='Ch %i'%channel, c=self.mpl_colors[channel])
                ax.set_ylim(-20,50)
                ax.set_ylabel(apply_filter*'filtered ' + 'db ish')
                if self.show_all:
                    ax.set_xlim(0,150)

            #Populate Table
            name_column = list(self.table_params.values())
            self.inspector_mpl['fig1_table'].clear() #Clear previous table.
            table = self.inspector_mpl['fig1_table'].table(cellText=list(zip(name_column,start_data)), loc='center', in_layout=True)
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            #table.scale(1,4)
            self.inspector_mpl['fig1_table'].axis('tight')
            self.inspector_mpl['fig1_table'].axis('off')

            self.inspector_mpl['fig1'].canvas.draw()
            # self.inspector_mpl['fig1'].canvas.flush_events()
            # plt.show()
            # plt.pause(0.05)
            return
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getSingleEventTableValues(self, table_dict, run_index, eventid):
        '''
        Given a run and eventid this will pull the values for the param keys listed in the table dict.
        '''
        try:
            out_array = numpy.zeros(len(list(table_dict.keys())))
            for param_index, param_key in enumerate(list(table_dict.keys())):
                if param_key == 'eventid':
                    out_array[param_index] = eventid
                elif param_key == 'run':
                    out_array[param_index] = self.runs[run_index]
                else:
                    out_array[param_index] = self.data_slicers[run_index].getDataFromParam([eventid], param_key)
            return out_array
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def delInspector(self):
        for ds in self.data_slicers:
            ds.delCorrelator()
        if hasattr(self,'table_params'):
            del self.table_params
        if hasattr(self,'inspector_mpl'):
            for key in list(self.inspector_mpl.keys()):
                del self.inspector_mpl[key]
            del self.inspector_mpl


    def eventInspector(self, eventids_dict, mollweide=False, show_all=False):
        '''
        This is meant to provide a tool to quickly flick through events from multiple runs.  It will create a one panel
        view of the events info as best as I can manage, and provide easy support for choosing which event you want to
        inspect. 
        '''
        try:
            self.show_all = show_all

            self.mpl_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']

            non_zero_runs = numpy.array(list(eventids_dict.keys()))[numpy.array([len(eventids_dict[k]) for k in list(eventids_dict.keys())]) > 0]
            non_zero_run_indices = numpy.where(numpy.isin(self.runs, non_zero_runs))[0]

            self.prepareCorrelator()
            
            fig1 = plt.figure(constrained_layout=True)
            gs = fig1.add_gridspec(4,5, width_ratios=[1,1,1,1,0.75])

            fig1_wf_0 = fig1.add_subplot(gs[0,0])
            fig1_wf_0.set_ylabel('0H')
            fig1_wf_0.yaxis.label.set_color(self.mpl_colors[0])
            fig1_wf_0.minorticks_on()
            fig1_wf_0.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_0.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_1 = fig1.add_subplot(gs[0,1], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_1.set_ylabel('0V')
            fig1_wf_1.yaxis.label.set_color(self.mpl_colors[1])
            fig1_wf_1.minorticks_on()
            fig1_wf_1.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_2 = fig1.add_subplot(gs[0,2], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_2.set_ylabel('1H')
            fig1_wf_2.yaxis.label.set_color(self.mpl_colors[2])
            fig1_wf_2.minorticks_on()
            fig1_wf_2.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_3 = fig1.add_subplot(gs[0,3], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_3.set_ylabel('1V')
            fig1_wf_3.yaxis.label.set_color(self.mpl_colors[3])
            fig1_wf_3.minorticks_on()
            fig1_wf_3.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_3.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_4 = fig1.add_subplot(gs[1,0], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_4.set_ylabel('2H')
            fig1_wf_4.yaxis.label.set_color(self.mpl_colors[4])
            fig1_wf_4.minorticks_on()
            fig1_wf_4.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_4.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_5 = fig1.add_subplot(gs[1,1], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_5.set_ylabel('2V')
            fig1_wf_5.yaxis.label.set_color(self.mpl_colors[5])
            fig1_wf_5.minorticks_on()
            fig1_wf_5.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_5.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_6 = fig1.add_subplot(gs[1,2], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_6.set_ylabel('3H')
            fig1_wf_6.yaxis.label.set_color(self.mpl_colors[6])
            fig1_wf_6.minorticks_on()
            fig1_wf_6.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_6.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_wf_7 = fig1.add_subplot(gs[1,3], sharex=fig1_wf_0, sharey=fig1_wf_0)
            fig1_wf_7.set_ylabel('3V')
            fig1_wf_7.yaxis.label.set_color(self.mpl_colors[7])
            fig1_wf_7.minorticks_on()
            fig1_wf_7.grid(b=True, which='major', color='k', linestyle='-')
            fig1_wf_7.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


            if mollweide == True:
                fig1_map_h      = fig1.add_subplot(gs[2:,0], projection='mollweide')
                fig1_map_v      = fig1.add_subplot(gs[2:,1], projection='mollweide', sharex=fig1_map_h, sharey=fig1_map_h)
                if self.show_all:
                    fig1_map_all    = fig1.add_subplot(gs[2:,2], projection='mollweide', sharex=fig1_map_h, sharey=fig1_map_h)
            else:
                fig1_map_h      = fig1.add_subplot(gs[2:,0])
                fig1_map_v      = fig1.add_subplot(gs[2:,1], sharex=fig1_map_h, sharey=fig1_map_h)
                if self.show_all:
                    fig1_map_all    = fig1.add_subplot(gs[2:,2], sharex=fig1_map_h, sharey=fig1_map_h)

            if self.show_all:
                fig1_spec_raw = fig1.add_subplot(gs[2,3:4])
            else:
                fig1_spec_raw = fig1.add_subplot(gs[2,2:4])
            fig1_spec_raw.minorticks_on()
            fig1_spec_raw.grid(b=True, which='major', color='k', linestyle='-')
            fig1_spec_raw.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            if self.show_all:
                fig1_spec_filt = fig1.add_subplot(gs[3,3:4], sharex=fig1_spec_raw)
            else:
                fig1_spec_filt = fig1.add_subplot(gs[3,2:4], sharex=fig1_spec_raw)
            fig1_spec_filt.minorticks_on()
            fig1_spec_filt.grid(b=True, which='major', color='k', linestyle='-')
            fig1_spec_filt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

            fig1_table = fig1.add_subplot(gs[:,4])
            
            #[['phi_best_h','elevation_best_h'],['hpol_max_map_value_abovehorizon','vpol_max_map_value_abovehorizon'], ['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value','vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value'], ['hpol_max_map_value','vpol_max_map_value'], ['hpol_peak_to_sidelobe_abovehorizon', 'vpol_peak_to_sidelobe_abovehorizon'],['hpol_peak_to_sidelobe_belowhorizon','hpol_peak_to_sidelobe_abovehorizon'], ['impulsivity_h','impulsivity_v'], ['cr_template_search_h', 'cr_template_search_v'], ['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'], ['hpol_max_possible_map_value','vpol_max_possible_map_value']]
            self.table_params = OrderedDict()
            #Format is param_key : 'Name In Table'
            self.table_params['run'] = 'Run'
            self.table_params['eventid'] = 'Event id'
            self.table_params['phi_best_choice'] = 'Az (BEST)'
            self.table_params['elevation_best_choice'] = 'El (BEST)'
            self.table_params['hpol_max_map_value_allsky'] = 'Map Max H'
            self.table_params['vpol_max_map_value_allsky'] = 'Map Max V'
            self.table_params['impulsivity_h'] = 'Imp H'
            self.table_params['impulsivity_v'] = 'Imp V'
            self.table_params['cr_template_search_h'] = 'CR XC H'
            self.table_params['cr_template_search_v'] = 'CR XC V'
            self.table_params['std_h'] = 'SDev H'
            self.table_params['std_v'] = 'SDev V'
            self.table_params['p2p_h'] = 'P2P H'
            self.table_params['p2p_v'] = 'P2P V'
            self.table_params['similarity_count_h'] = 'H Simlr'
            self.table_params['similarity_count_v'] = 'V Simlr'



            self.table_params['hpol_peak_to_sidelobe_allskySLICERMULTIPLYhpol_max_map_value_allsky'] = 'AS - H'
            self.table_params['vpol_peak_to_sidelobe_allskySLICERMULTIPLYvpol_max_map_value_allsky'] = 'AS - V'
            self.table_params['all_peak_to_sidelobe_allskySLICERMULTIPLYall_max_map_value_allsky'] = 'AS - All'
            self.table_params['hpol_peak_to_sidelobe_belowhorizonSLICERMULTIPLYhpol_max_map_value_belowhorizon'] = 'BH - H'
            self.table_params['vpol_peak_to_sidelobe_belowhorizonSLICERMULTIPLYvpol_max_map_value_belowhorizon'] = 'BH - V'
            self.table_params['all_peak_to_sidelobe_belowhorizonSLICERMULTIPLYall_max_map_value_belowhorizon'] = 'BH - All'
            self.table_params['hpol_peak_to_sidelobe_abovehorizonSLICERMULTIPLYhpol_max_map_value_abovehorizon'] = 'AH - H'
            self.table_params['vpol_peak_to_sidelobe_abovehorizonSLICERMULTIPLYvpol_max_map_value_abovehorizon'] = 'AH - V'
            self.table_params['all_peak_to_sidelobe_abovehorizonSLICERMULTIPLYall_max_map_value_abovehorizon'] = 'AH - All'
            

            # self.table_params['hpol_peak_to_sidelobe_allsky'] = 'AS P2S H'
            # self.table_params['vpol_peak_to_sidelobe_allsky'] = 'AS P2S V'
            # self.table_params['all_peak_to_sidelobe_allsky'] = 'AS P2S All'
            # self.table_params['hpol_peak_to_sidelobe_belowhorizon'] = 'BH P2S H'
            # self.table_params['vpol_peak_to_sidelobe_belowhorizon'] = 'BH P2S V'
            # self.table_params['all_peak_to_sidelobe_belowhorizon'] = 'BH P2S All'
            # self.table_params['hpol_peak_to_sidelobe_abovehorizon'] = 'AH P2S H'
            # self.table_params['vpol_peak_to_sidelobe_abovehorizon'] = 'AH P2S V'
            # self.table_params['all_peak_to_sidelobe_abovehorizon'] = 'AH P2S All'
            

            self.table_params['phi_best_h_allsky'] = 'Az H'
            self.table_params['elevation_best_h_allsky'] = 'El H'
            self.table_params['phi_best_v_allsky'] = 'Az V'
            self.table_params['elevation_best_v_allsky'] = 'El V'
            self.table_params['phi_best_all_allsky'] = 'Az All'
            self.table_params['elevation_best_all_allsky'] = 'El All'

            self.table_params['hpol_max_map_value_abovehorizonSLICERDIVIDEhpol_max_possible_map_value'] = 'H AH Peak/Opt'
            self.table_params['vpol_max_map_value_abovehorizonSLICERDIVIDEvpol_max_possible_map_value'] = 'V AH Peak/Opt'

            self.table_params['coincidence_method_1_h'] = 'coin m1H'
            self.table_params['coincidence_method_1_v'] = 'coin m1V'
            self.table_params['coincidence_method_2_h'] = 'coin m2H'
            self.table_params['coincidence_method_2_v'] = 'coin m2V'


            #Sample eventid, would normally be selected from and changeable
            run = non_zero_runs[0]
            eventid = eventids_dict[run][0]
            run_index = int(numpy.where(self.runs == run)[0])#self.data_slicers[]

            self.inspector_mpl = {}
            self.inspector_mpl['fig1'] = fig1
            self.inspector_mpl['gs'] = gs
            self.inspector_mpl['fig1_wf_0'] = fig1_wf_0
            self.inspector_mpl['fig1_wf_1'] = fig1_wf_1
            self.inspector_mpl['fig1_wf_2'] = fig1_wf_2
            self.inspector_mpl['fig1_wf_3'] = fig1_wf_3
            self.inspector_mpl['fig1_wf_4'] = fig1_wf_4
            self.inspector_mpl['fig1_wf_5'] = fig1_wf_5
            self.inspector_mpl['fig1_wf_6'] = fig1_wf_6
            self.inspector_mpl['fig1_wf_7'] = fig1_wf_7
            self.inspector_mpl['fig1_map_h'] = fig1_map_h
            self.inspector_mpl['fig1_map_v'] = fig1_map_v
            if self.show_all:
                self.inspector_mpl['fig1_map_all'] = fig1_map_all
            self.inspector_mpl['fig1_spec_raw'] = fig1_spec_raw
            self.inspector_mpl['fig1_spec_filt'] = fig1_spec_filt
            self.inspector_mpl['fig1_table'] = fig1_table
            
            
            self.updateEventInspect(run_index, eventid, mollweide=mollweide)

            self.inspector_eventids_list = self.organizeEventDict(eventids_dict)
            self.inspector_eventids_index = 0

            '''
            Should add a button to cycle through all events and save a figure for them.
            '''

            class NextEventIterator(ToolBase):
                description = 'This will skip to the next event in the selected event dictionary.'
                image = os.path.join(matplotlib.matplotlib_fname().replace('matplotlibrc','images'), 'forward.png')
                def __init__(self, toolmanager, name, outer):
                    self._name = name
                    self._toolmanager = toolmanager
                    self._figure = None
                    self.outer = outer
                def trigger(self, sender, event, data=None):
                    '''
                    What actually happens when the button is pressed.
                    '''
                    self.outer.inspector_eventids_index = (self.outer.inspector_eventids_index + 1)%len(self.outer.inspector_eventids_list)
                    self.outer.updateEventInspect(self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['run_index'],self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['eventid'])

            class PreviousEventIterator(ToolBase):
                description = 'This will skip to the previous event in the selected event dictionary.'
                image = os.path.join(matplotlib.matplotlib_fname().replace('matplotlibrc','images'), 'back.png')
                def __init__(self, toolmanager, name, outer):
                    self._name = name
                    self._toolmanager = toolmanager
                    self._figure = None
                    self.outer = outer
                def trigger(self, sender, event, data=None):
                    '''
                    What actually happens when the button is pressed.
                    '''
                    self.outer.inspector_eventids_index = (self.outer.inspector_eventids_index - 1)%len(self.outer.inspector_eventids_list)
                    self.outer.updateEventInspect(self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['run_index'],self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['eventid'])

            class SaveBook(ToolBase):
                description = 'This save an image of every event in the current event browser.'
                image = os.path.join(matplotlib.matplotlib_fname().replace('matplotlibrc','images'), 'filesave.png')
                def __init__(self, toolmanager, name, outer):
                    self._name = name
                    self._toolmanager = toolmanager
                    self._figure = None
                    self.outer = outer
                def trigger(self, sender, event, data=None):
                    '''
                    What actually happens when the button is pressed.
                    '''
                    outpath = './event_flipbook_%i'%time.time() 
                    os.mkdir(outpath)

                    self.outer.inspector_mpl['fig1'].set_size_inches(25, 12.5)
                    print()
                    for i in range(len(self.outer.inspector_eventids_list)):
                        sys.stdout.write('Saving Figure %i/%i\r'%(i+1,len(self.outer.inspector_eventids_list)))
                        sys.stdout.flush()
                        self.outer.inspector_eventids_index = (self.outer.inspector_eventids_index + 1)%len(self.outer.inspector_eventids_list)
                        run_index = self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['run_index']
                        run = self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['run']
                        eventid = self.outer.inspector_eventids_list[self.outer.inspector_eventids_index]['eventid']
                        self.outer.updateEventInspect(run_index,eventid)
                        self.outer.inspector_mpl['fig1'].savefig(os.path.join(outpath, 'r%ie%i.png'%(run,eventid)), dpi=300, bbox_inches='tight')

            class UpdateMapWindow(ToolBase):
                description = 'This will generate a map using the currently cropped and visible portion of the waveforms.'
                image = os.path.join(matplotlib.matplotlib_fname().replace('matplotlibrc','images'), 'matplotlib.png')
                def __init__(self, toolmanager, name, outer):
                    self._name = name
                    self._toolmanager = toolmanager
                    self._figure = None
                    self.outer = outer
                def trigger(self, sender, event, data=None):
                    '''
                    What actually happens when the button is pressed.
                    '''
                    override_to_time_window = self.outer.inspector_mpl['fig1_wf_0'].get_xlim()
                    #import pdb; pdb.set_trace()

                    #Plot Maps
                    m, self.outer.inspector_mpl['fig1'], self.outer.inspector_mpl['fig1_map_h'] = self.outer.cor.map(eventid, 'hpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=self.outer.inspector_mpl['fig1_map_h'], plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=self.outer.mollweide, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=90 - self.outer.el_h, circle_az=self.outer.az_h, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False, override_to_time_window=override_to_time_window)
                    m, self.outer.inspector_mpl['fig1'], self.outer.inspector_mpl['fig1_map_v'] = self.outer.cor.map(eventid, 'vpol', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=self.outer.inspector_mpl['fig1_map_v'], plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=self.outer.mollweide, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=90 - self.outer.el_v, circle_az=self.outer.az_v, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False, override_to_time_window=override_to_time_window)
                    if self.outer.show_all:
                        m, self.outer.inspector_mpl['fig1'], self.outer.inspector_mpl['fig1_map_all'] = self.outer.cor.map(eventid, 'all', include_baselines=numpy.array([0,1,2,3,4,5]), plot_map=True, map_ax=self.outer.inspector_mpl['fig1_map_all'], plot_corr=False, hilbert=False, interactive=True, max_method=None, waveforms=None, verbose=False, mollweide=self.outer.mollweide, zenith_cut_ENU=None, zenith_cut_array_plane=(0,90), center_dir='E', circle_zenith=90 - self.outer.el_all, circle_az=self.outer.az_all, radius=1.0, time_delay_dict={},window_title=None,add_airplanes=False, return_max_possible_map_value=False, plot_peak_to_sidelobe=True, shorten_signals=False, shorten_thresh=0.7, shorten_delay=10.0, shorten_length=90.0, shorten_keep_leading=100.0, minimal=True, circle_map_max=False, override_to_time_window=override_to_time_window)
                    self.outer.inspector_mpl['fig1'].canvas.draw()

            tm = self.inspector_mpl['fig1'].canvas.manager.toolmanager
            #import pdb; pdb.set_trace()
            tm.add_tool('Save Book', SaveBook, self)
            self.inspector_mpl['fig1'].canvas.manager.toolbar.add_tool(tm.get_tool('Save Book'), 'toolgroup')
            tm.add_tool('Previous', PreviousEventIterator, self)
            self.inspector_mpl['fig1'].canvas.manager.toolbar.add_tool(tm.get_tool('Previous'), 'toolgroup')
            tm.add_tool('Next', NextEventIterator, self)
            self.inspector_mpl['fig1'].canvas.manager.toolbar.add_tool(tm.get_tool('Next'), 'toolgroup')
            tm.add_tool('Update Map', UpdateMapWindow, self)
            self.inspector_mpl['fig1'].canvas.manager.toolbar.add_tool(tm.get_tool('Update Map'), 'toolgroup')
            # tm.add_tool('Quit', ToolQuit)
            # self.inspector_mpl['fig1'].canvas.manager.toolbar.add_tool(tm.get_tool('Quit'), 'toolgroup')
            return

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateTimeDelayCommonality(self, eventids_dict, verbose=False, return_eventids_array=False):
        '''
        This will generate a histogram for each time delay.  The height of the bin containing each event will be added
        to that events total sum.  This is repeated for each baseline and polarization.  Events with a high value in the
        returned array thus should be clustered or "highly common".  Effects of sidelobes will effect the resulting output.
        '''
        try:
            eventids_array = self.organizeEventDict(eventids_dict)
            common = numpy.zeros(len(eventids_array))
            if verbose:
                print('Calculating Time Delay Commonality')
            for key in ['time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h','time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v']:
                if verbose:
                    print('On %s'%key)
                self.setCurrentPlotBins(key,key,eventids_dict)
                indices = numpy.digitize( self.getDataArrayFromParam(key, trigger_types=None, eventids_dict=eventids_dict) , self.current_bin_edges_x)
                uvals, uindices, ucounts = numpy.unique(indices,return_inverse=True, return_counts=True)
                common += ucounts[uindices]

            if return_eventids_array:
                return common, eventids_array
            else:
                return common
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    plt.close('all')
    
    #runs = [1642,1643,1644,1645,1646,1647]
    if True:
        for runs in [[5733,5735,5734,5741,5742,5736,5740,5763,5750,5757,5749,5751,5743,5773,5739,5758,5756,5774,5775,5738,5746,5776,5789,5771]]:

            impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'
            print("Preparing dataSlicer")

            map_resolution_theta = 0.25 #degrees
            min_theta   = 0
            max_theta   = 120
            n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

            map_resolution_phi = 0.1 #degrees
            min_phi     = -180
            max_phi     = 180
            n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

            ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                            cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                            std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                            snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                            n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta))

            print('Generating plots:')
            for key_x, key_y in [['impulsivity_h','impulsivity_v'],['phi_best_h','elevation_best_h'],['phi_best_v','elevation_best_v'],['p2p_h', 'p2p_v']]:#[['p2p_h','p2p_v'],['impulsivity_h','impulsivity_v'],['p2p_h','impulsivity_v'],['time_delay_0subtract1_h','time_delay_0subtract2_h']]:#ds.known_param_keys:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                print('Testing plot for calculating %s and %s'%(key_x,key_y))
                ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
    elif False:
        for runs in [[1644],[1645],[1644,1645]]:
            impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
            map_direction_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-upsample_32768-maxmethod_0-sinesubtract_1-deploy_calibration_15'
            print("Preparing dataSlicer")
            ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                            cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                            std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                            snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False)

            print('Adding ROI to dataSlicer')

            ds.addROI('corr A',{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]})
            ds.addROI('high v imp',{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]})
            ds.addROI('small h.4 v.4 imp',{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]})

            ds.addROI('imp cluster',{'impulsivity_h':[0.35,0.46],'impulsivity_v':[0.45,0.50]})

            print('Generating plots:')
            for key_x, key_y in [['impulsivity_h','impulsivity_v']]:#[['p2p_h','p2p_v'],['impulsivity_h','impulsivity_v'],['p2p_h','impulsivity_v'],['time_delay_0subtract1_h','time_delay_0subtract2_h']]:#ds.known_param_keys:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                print('Testing plot for calculating %s and %s'%(key_x,key_y))
                ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)

    if False:
        for run in runs:
            reader = Reader(raw_datapath,run)
            #.replace('-sinesubtract','sinesubtract') #catches one-off case misnaming dset
            impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_0-shorten_signals-1-shorten_thresh-0.70-shorten_delay-10.00-shorten_length-90.00-sinesubtract_1'#'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
            time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_32768-align_0-shorten_signals-1-shorten_thresh-0.70-shorten_delay-10.00-shorten_length-90.00-sinesubtract_1'#'LPf_None-LPo_None-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_262144-align_0'
            print("Preparing dataSlicerSingleRun")
            ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                            cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                            std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                            snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False)

            print('Adding ROI to dataSlicerSingleRun')

            ds.addROI('corr A',{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]})
            ds.addROI('high v imp',{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]})
            ds.addROI('small h.4 v.4 imp',{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]})

            ds.addROI('imp cluster',{'impulsivity_h':[0.35,0.46],'impulsivity_v':[0.45,0.50]})

            params_needed_for_roi = []
            for key in list(ds.roi.keys()):
                for k in list(ds.roi[key].keys()):
                    params_needed_for_roi.append(k)
            params_needed_for_roi = numpy.unique(params_needed_for_roi)
            for param in params_needed_for_roi:
                ds.saveHistogramData(param)

            import timeit
            print('Generating plots:')
            for key_x, key_y in [['p2p_h','p2p_v'],['impulsivity_h','impulsivity_v'],['p2p_h','impulsivity_v'],['time_delay_0subtract1_h','time_delay_0subtract2_h']]:#ds.known_param_keys:
                print('Generating %s plot'%(key_x + ' vs ' + key_y))
                print('Testing plot for calculating %s and %s'%(key_x,key_y))
                start_time = timeit.default_timer()
                ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                elapsed_1 = timeit.default_timer() - start_time
                print('[TIME] Calculation Plot: %0.1f seconds'%elapsed_1)

                print('Saving data for %s and %s'%(key_x,key_y))

                ds.saveHistogramData(key_x)
                ds.saveHistogramData(key_y)
                elapsed_2 = timeit.default_timer() - start_time
                print('[TIME] Saving Data: %0.1f seconds'%(elapsed_2 - elapsed_1))

                print('Testing plot for loading %s and %s'%(key_x,key_y))
                ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
                
                elapsed_3 = timeit.default_timer() - start_time
                print('[TIME] Loading Plot: %0.1f seconds'%(elapsed_3 - elapsed_2))

      