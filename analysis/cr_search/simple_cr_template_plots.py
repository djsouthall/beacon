#!/usr/bin/env python3
'''
The purpose of this script is to generate plots based on the simple_cr_template_search.py results.

The documention for each function needs to be updated to represent the new functionality.  Additionally this should
all be moved to the tools folder and just be called here.  

'''

import sys
import os
import inspect
import h5py

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
import tools.cosmic_ray_template as crt
from tools.data_handler import createFile
from analysis.background_identify_60hz import diffFromPeriodic
from objects.fftmath import TemplateCompareTool
from matplotlib.patches import Rectangle
from matplotlib import cm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.ion()

import numpy
import scipy
import scipy.signal
import scipy.signal
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']


# What do I want?
'''
I want a tool that given a run/reader, and stored analysis file, can produce 2d histogram plots for each of the known
measureable quantities, with contour functionality.  

This will mean that it needs support for applying cuts on each parameter, and storing the result (or having a function
to calculate it every time for the given cuts).  

This will have to support looping over events. 
'''

class dataSlicerSingleRun():
    '''
    This operates on a per run basis, but a daughter class could be used to incorporate support for multiple runs.
    As most calculations require running through a single runs analysis file, I think this makes sense.  The histograms
    can just be added accross runs and hopefully the plotting functions could work with that. 

    I want this to be an analysis tool that has the functionalities outlined below (roughly, as a statement of intent
    as I continue to develop this class).

    Goal
    ----
    I want a tool that given a run/reader, and stored analysis file, can produce 2d histogram plots for each of the 
    known measureable quantities, with contour functionality.  

    This will mean that it needs support for applying cuts on each parameter, and storing the result (or having a function
    to calculate it every time for the given cuts).  

    This will have to support looping over events. 

    I have yet to decide whether I want to keep the h5py files open, or if that is a problem.

    Parameters
    ----------
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
    def __init__(self,  reader, impulsivity_dset_key, curve_choice=0, trigger_types=[1,2,3],included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=None,p2p_n_bins_h=256,p2p_n_bins_v=256,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=None,include_test_roi=False):
        try:
            #
            self.included_antennas = included_antennas#[0,1,2,3,4,5,6,7]
            self.included_hpol_antennas = numpy.array([0,2,4,6])[numpy.isin([0,2,4,6],self.included_antennas)]
            self.included_vpol_antennas = numpy.array([1,3,5,7])[numpy.isin([1,3,5,7],self.included_antennas)]

            #I want to work on adding: 'std', 'p2p', and 'snr', where snr is p2p/std.  I think these could be interesting, and are already available by default per signal. 
            #self.known_param_keys = ['impulsivity_hv', 'cr_template_search', 'std', 'p2p', 'snr'] #If it is not listed in here then it cannot be used.
            self.known_param_keys = ['impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v']
            self.updateReader(reader)

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

            #std Plot Params:
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

            self.trigger_types = trigger_types

            self.eventids_matching_trig = self.getEventidsFromTriggerType() #Opens file, so has to be called outside of with statement.
            
            #In progress.
            #ROI  List
            #This should be a list with coords x1,x2, y1, y2 for each row.  These will be used to produce plots pertaining to 
            #these specific regions of interest.  x and y should be correlation values, and ordered.
            self.roi = {}
            if include_test_roi:
                self.roi['test'] = {'impulsivity_hv':numpy.array([0.5,0.9,0.5,1.0]),
                                    'cr_template_hv':numpy.array([0.5,0.9,0.5,1.0])}
                #'map_az_elev_deg':numpy.array([-60,60.-40,0.0])

            #self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, numpy.shape(roi)[0])]
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def printKnownParamKeys(self):
        '''
        This will return a list of the currently supported parameter keys for making 2d histogram plots.
        '''
        return print(self.known_param_keys)
    def updateReader(self, reader):
        '''
        This will let the user update the reader file if they are looping over multiple runs.
        '''
        try:
            self.reader = reader
            self.analysis_filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def printSampleROI(self):
        '''
        This will print and return an example ROI dictionary.  This is to provide examples of what a dictionary input
        to self.addROI might look like. 
        '''
        sample_ROI = {  'corr A':{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]},\
                        'high v imp':{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]},\
                        'small h.4 v.4 imp':{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]}}
        print('Sample ROI dict:')
        print(sample_ROI)
        return sample_ROI

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
            if numpy.any(~numpy.isin(list(roi_dict.keys()), self.known_param_keys)):
                print('WARNING!!!')
                for key in list(roi_dict.keys())[~numpy.isin(list(roi_dict.keys()), self.known_param_keys)]:
                    print('The given roi parameter [%s] not in the list of known parameters\n%s:'%(key,str(self.known_param_keys)))
                return
            else:
                self.roi[roi_key] = roi_dict
                self.roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, len(list(self.roi.keys())))]

            #Could add more checks here to ensure roi_dict is good.
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getEventidsFromTriggerType(self):
        '''
        This will get the eventids that match one of the specified trigger types in self.trigger_types
        '''
        with h5py.File(self.analysis_filename, 'r') as file:
            #Initial cut based on trigger type.
            eventids = numpy.where(numpy.isin(file['trigger_type'][...],self.trigger_types))[0]
            file.close()
        return eventids

    def getDataFromParam(self, eventids, param_key):
        '''
        Given eventids, this will load and return array for parameters associated with string param.
        '''
        if param_key in self.known_param_keys:
            with h5py.File(self.analysis_filename, 'r') as file:
                if param_key == 'impulsivity_h':
                    param = file['impulsivity'][self.impulsivity_dset_key]['hpol'][eventids]
                elif param_key == 'impulsivity_v':
                    param = file['impulsivity'][self.impulsivity_dset_key]['vpol'][eventids]
                elif param_key == 'cr_template_search_h':
                    this_dset = 'bi-delta-curve-choice-%i'%self.cr_template_curve_choice
                    output_correlation_values = file['cr_template_search'][this_dset][eventids]
                    param = numpy.max(output_correlation_values[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD 
                elif param_key == 'cr_template_search_v':
                    this_dset = 'bi-delta-curve-choice-%i'%self.cr_template_curve_choice
                    output_correlation_values = file['cr_template_search'][this_dset][eventids]
                    param = numpy.max(output_correlation_values[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                elif param_key == 'std_h':
                    std = file['std'][eventids]
                    param = numpy.mean(std[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                elif param_key == 'std_v':
                    std = file['std'][eventids]
                    param = numpy.mean(std[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                elif param_key == 'p2p_h': 
                    p2p = file['p2p'][eventids]
                    param = numpy.max(p2p[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                elif param_key == 'p2p_v': 
                    p2p = file['p2p'][eventids]
                    param = numpy.max(p2p[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                elif param_key == 'snr_h':
                    std = file['std'][eventids]
                    param_1 = numpy.mean(std[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                    p2p = file['p2p'][eventids]
                    param_2 = numpy.max(p2p[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                    param = numpy.divide(param_2, param_1)
                elif param_key == 'snr_v':
                    std = file['std'][eventids]
                    param_1 = numpy.mean(std[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                    p2p = file['p2p'][eventids]
                    param_2 = numpy.max(p2p[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                    param = numpy.divide(param_2, param_1)
                file.close()
        else:
            print('\nWARNING!!!\nOther parameters have not been accounted for yet.\n%s'%(param_key))
        return param

    def getCutsFromROI(self,roi_key):
        '''
        This will determine the eventids that match the cuts listed in the dictionary corresponding to roi_key, and will
        return them.  
        '''
        if roi_key in list(self.roi.keys()):
            eventids = self.eventids_matching_trig.copy()
            for param_key in list(self.roi[roi_key].keys()):
                param = self.getDataFromParam(eventids, param_key)
                #Reduce eventids by box cut
                eventids = eventids[numpy.logical_and(param >= self.roi[roi_key][param_key][0], param < self.roi[roi_key][param_key][1])]  #Should get smaller/faster with every param cut.
        else:
            print('WARNING!!!')
            print('Requested ROI [%s] is not specified in self.roi list:\n%s'%(roi_key,str(list(self.roi.keys()))))

        return eventids


    def get2dHistCounts(self, main_param_key_x, main_param_key_y, eventids):
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
        '''
        try:
            print('\tLoading data for %s'%main_param_key_x)
            param_x = self.getDataFromParam(eventids, main_param_key_x)
            print('\tLoading data for %s'%main_param_key_y)
            param_y = self.getDataFromParam(eventids, main_param_key_y)
            print('\tGetting counts from 2dhist')

            counts = numpy.histogram2d(param_x, param_y, bins = [self.current_bin_edges_x,self.current_bin_edges_y])[0].T #Outside of file being open 

            return counts

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def setCurrentPlotBins(self, main_param_key_x, main_param_key_y, eventids):
        '''
        Sets the current bins to be used for main_param_key.  These can then be accessed easily by other functions for
        producing the hist count data.  This should be called by any functions starting a plot. 
        '''
        if numpy.logical_and(main_param_key_x in self.known_param_keys, main_param_key_y in self.known_param_keys):
            std_requiring_params = ['std_h','std_v','snr_h','snr_v'] #List all params that might require max snr to be calculated if hard limit not given.
            p2p_requiring_params = ['p2p_h','p2p_v','snr_h','snr_v'] #List all params that might require max p2p to be calculated if hard limit not given.
            if numpy.logical_or(numpy.isin(main_param_key_x,std_requiring_params),numpy.isin(main_param_key_y,std_requiring_params)):
                if self.max_std_val is None:
                    with h5py.File(self.analysis_filename, 'r') as file:
                        print('Calculating max_std')
                        self.max_std_val = numpy.max(file['std'][eventids,:]) #Should only have to be done once on first call. 
                        self.min_std_val = numpy.min(file['std'][eventids,:])
                        file.close()
            if numpy.logical_or(numpy.isin(main_param_key_x,p2p_requiring_params),numpy.isin(main_param_key_y,p2p_requiring_params)):
                if self.max_p2p_val is None:
                    with h5py.File(self.analysis_filename, 'r') as file:
                        print('Calculating max_p2p')
                        self.max_p2p_val = numpy.max(file['p2p'][eventids,:]) #Should only have to be done once on first call. 
                        file.close()

        if main_param_key_x not in self.known_param_keys:
            print('WARNING!!!')
            print('Given key [%s] is not listed in known_param_keys:\n%s'%(main_param_key,str(self.known_param_keys)))
            return
        else:
            print('\tPreparing to get counts for %s'%main_param_key_x)

            if main_param_key_x == 'impulsivity_h':
                self.current_x_label = 'Impulsivity (hpol)'
                x_n_bins = self.impulsivity_n_bins_h
                x_max_val = 1
                x_min_val = 0
            elif main_param_key_x == 'impulsivity_v':
                self.current_x_label = 'Impulsivity (vpol)'
                x_n_bins = self.impulsivity_n_bins_v
                x_max_val = 1
                x_min_val = 0
            elif main_param_key_x == 'cr_template_search_h':
                self.current_x_label = 'HPol Correlation Values with CR Template'
                x_n_bins = self.cr_template_n_bins_h
                x_max_val = 1
                x_min_val = 0
            elif main_param_key_x == 'cr_template_search_v':
                self.current_x_label = 'VPol Correlation Values with CR Template'
                x_n_bins = self.cr_template_n_bins_v
                x_max_val = 1
                x_min_val = 0
            elif main_param_key_x == 'std_h':
                self.current_x_label = 'Mean Time Domain STD (hpol)'
                x_n_bins = self.std_n_bins_h
                x_max_val = self.max_std_val
                x_min_val = 0
            elif main_param_key_x == 'std_v':
                self.current_x_label = 'Mean Time Domain STD (vpol)'
                x_n_bins = self.std_n_bins_v
                x_max_val = self.max_std_val
                x_min_val = 0
            elif main_param_key_x == 'p2p_h': 
                self.current_x_label = 'Mean P2P (hpol)'
                x_n_bins = self.p2p_n_bins_h
                x_max_val = self.max_p2p_val
                x_min_val = 0
            elif main_param_key_x == 'p2p_v': 
                self.current_x_label = 'Mean P2P (vpol)'
                x_n_bins = self.p2p_n_bins_v
                x_max_val = self.max_p2p_val
                x_min_val = 0
            elif main_param_key_x == 'snr_h':
                self.current_x_label = 'Mean SNR (hpol)\n P2P/STD'
                x_n_bins = self.snr_n_bins_h
                x_max_val = self.max_p2p_val/self.min_std_val
                x_min_val = 0
            elif main_param_key_x == 'snr_v':
                self.current_x_label = 'Mean SNR (vpol)\n P2P/STD'
                x_n_bins = self.snr_n_bins_v
                x_max_val = self.max_p2p_val/self.min_std_val
                x_min_val = 0

        if main_param_key_y not in self.known_param_keys:
            print('WARNING!!!')
            print('Given key [%s] is not listed in known_param_keys:\n%s'%(main_param_key,str(self.known_param_keys)))
            return
        else:
            print('\tPreparing to get counts for %s'%main_param_key_y)

            if main_param_key_y == 'impulsivity_h':
                self.current_y_label = 'Impulsivity (hpol)'
                y_n_bins = self.impulsivity_n_bins_h
                y_max_val = 1
                y_min_val = 0
            elif main_param_key_y == 'impulsivity_v':
                self.current_y_label = 'Impulsivity (vpol)'
                y_n_bins = self.impulsivity_n_bins_v
                y_max_val = 1
                y_min_val = 0
            elif main_param_key_y == 'cr_template_search_h':
                self.current_y_label = 'HPol Correlation Values with CR Template'
                y_n_bins = self.cr_template_n_bins_h
                y_max_val = 1
                y_min_val = 0
            elif main_param_key_y == 'cr_template_search_v':
                self.current_y_label = 'VPol Correlation Values with CR Template'
                y_n_bins = self.cr_template_n_bins_v
                y_max_val = 1
                y_min_val = 0
            elif main_param_key_y == 'std_h':
                self.current_y_label = 'Mean Time Domain STD (hpol)'
                y_n_bins = self.std_n_bins_h
                y_max_val = self.max_std_val
                y_min_val = 0
            elif main_param_key_y == 'std_v':
                self.current_y_label = 'Mean Time Domain STD (vpol)'
                y_n_bins = self.std_n_bins_v
                y_max_val = self.max_std_val
                y_min_val = 0
            elif main_param_key_y == 'p2p_h': 
                self.current_y_label = 'Mean P2P (hpol)'
                y_n_bins = self.p2p_n_bins_h
                y_max_val = self.max_p2p_val
                y_min_val = 0
            elif main_param_key_y == 'p2p_v': 
                self.current_y_label = 'Mean P2P (vpol)'
                y_n_bins = self.p2p_n_bins_v
                y_max_val = self.max_p2p_val
                y_min_val = 0
            elif main_param_key_y == 'snr_h':
                self.current_y_label = 'Mean SNR (hpol)\n P2P/STD'
                y_n_bins = self.snr_n_bins_h
                y_max_val = self.max_p2p_val/self.min_std_val
                y_min_val = 0
            elif main_param_key_y == 'snr_v':
                self.current_y_label = 'Mean SNR (vpol)\n P2P/STD'
                y_n_bins = self.snr_n_bins_v
                y_max_val = self.max_p2p_val/self.min_std_val
                y_min_val = 0

        self.current_bin_edges_x = numpy.linspace(x_min_val,x_max_val,x_n_bins + 1) #These are bin edges
        self.current_bin_edges_y = numpy.linspace(y_min_val,y_max_val,y_n_bins + 1) #These are bin edges
        self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v = numpy.meshgrid((self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2, (self.current_bin_edges_y[:-1] + self.current_bin_edges_y[1:]) / 2)


    def plot2dHist(self, main_param_key_x,  main_param_key_y, eventids, title=None,cmap='coolwarm'):
        '''
        This is meant to be a function the plot corresponding to the main parameter, and will plot the same quantity 
        (corresponding to main_param_key) with just events corresponding to the cut being used.  This subset will show
        up as a contour on the plot.  
        '''
        self.setCurrentPlotBins(main_param_key_x,main_param_key_y,eventids)
        #Should make eventids a self.eventids so I don't need to call this every time.
        counts = self.get2dHistCounts(main_param_key_x,main_param_key_y,eventids)
        
        _fig, _ax = plt.subplots()
        if title is not None:
            plt.title(title)
        else:
            plt.title('%s, Run = %i\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,int(self.reader.run),str(self.trigger_types)))

        _im = _ax.pcolormesh(self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v, counts,norm=colors.LogNorm(vmin=0.5, vmax=counts.max()),cmap=cmap)#cmap=plt.cm.coolwarm

        plt.xlabel(self.current_x_label)
        plt.ylabel(self.current_y_label)

        try:
            cbar = _fig.colorbar(_im)
            cbar.set_label('Counts')
        except Exception as e:
            print('Error in colorbar, often caused by no events.')
            print(e)

        return _fig, _ax



    def addContour(self, ax, main_param_key_x, main_param_key_y, contour_eventids, contour_color, n_contour=6, alpha=0.75):
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

        Returns
        -------
        ax : matplotlib axes
            This is the updated axes that the contour was added to.
        cs : QuadContourSet
            This is the contour object, that can be used for adding labels to the legend for this contour.
        '''
        try:
            counts = self.get2dHistCounts(main_param_key_x, main_param_key_y, contour_eventids)
            levels = numpy.linspace(0,numpy.max(counts),n_contour)[1:n_contour+1] #Not plotting bottom contour because it is often background and clutters plot.
            cs = ax.contour(self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v, counts, colors=[contour_color],levels=levels,alpha=alpha)
            return ax, cs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            

    def plotROI2dHist(self, main_param_key_x, main_param_key_y, cmap='coolwarm', include_roi=True):
        '''
        This is the "do it all" function.  Given the parameter it will plot the 2dhist of the corresponding param by
        calling plot2dHist.  It will then plot the contours for each ROI on top.  It will do so assuming that each 
        ROI has a box cut in self.roi for EACH parameter.  I.e. it expects the # of listed entries in each ROI to be
        the same, and it will add a contour for the eventids that pass ALL cuts for that specific roi. 
        '''
        eventids = self.getEventidsFromTriggerType()
        fig, ax = self.plot2dHist(main_param_key_x, main_param_key_y, eventids, title=None, cmap=cmap) #prepares binning, must be called early (before addContour)

        #these few lines below this should be used for adding contours to the map. 
        if include_roi:
            legend_properties = []
            legend_labels = []

            for roi_index, roi_key in enumerate(list(self.roi.keys())):
                contour_eventids = self.getCutsFromROI(roi_key)

                ax, cs = self.addContour(ax, main_param_key_x, main_param_key_y, contour_eventids, self.roi_colors[roi_index], n_contour=6)
                legend_properties.append(cs.legend_elements()[0][0])
                legend_labels.append('roi %i: %s'%(roi_index, roi_key))

            plt.legend(legend_properties,legend_labels)

        return fig, ax


def main():
    print('WARNING THIS IS DEPRECATED AND EXPECTS OLDER FORM OF CLASS')
    plt.close('all')
    #Parameters:
    #General Params:
    curve_choice = 0 #Which curve to select from correlation data.


    #1dhist Params:
    plot_1dhists = True
    bins_1dhist = numpy.linspace(0,1,201)

    #2dhist Params:
    plot_2dhists = True
    bins_2dhist_h = numpy.linspace(0,1,201)
    bins_2dhist_v = numpy.linspace(0,1,201)
    
    bin_centers_mesh_h, bin_centers_mesh_v = numpy.meshgrid((bins_2dhist_h[:-1] + bins_2dhist_h[1:]) / 2, (bins_2dhist_v[:-1] + bins_2dhist_v[1:]) / 2)

    trigger_types = [2]#[1,2,3]
    #In progress.
    #ROI  List
    #This should be a list with coords x1,x2, y1, y2 for each row.  These will be used to produce plots pertaining to 
    #these specific regions of interest.  x and y should be correlation values, and ordered.
    plot_roi = True
    roi = numpy.array([])
    # roi = numpy.array([ [0.16,0.4,0.16,0.45],
    #                     [0.56,0.745,0.88,0.94],
    #                     [0.67,0.79,0.53,0.60],
    #                     [0.7,0.85,0.65,0.80],
    #                     [0.70,0.82,0.38,0.46],
    #                     [0.12,0.38,0.05,0.12],
    #                     [0.55,0.66,0.38,0.44],
    #                     [0.50,0.75,0.20,0.32],
    #                     [0.48,0.63,0.55,0.63]])
    roi_colors = [cm.rainbow(x) for x in numpy.linspace(0, 1, numpy.shape(roi)[0])]

    #Impulsivity Plot Params:
    plot_impulsivity = True
    impulsivity_dset = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
    impulsivity_bin_edges_h = numpy.linspace(0,1,201)
    impulsivity_bin_edges_v = numpy.linspace(0,1,201)
    
    impulsivity_bin_centers_mesh_h, impulsivity_bin_centers_mesh_v = numpy.meshgrid((impulsivity_bin_edges_h[:-1] + impulsivity_bin_edges_h[1:]) / 2, (impulsivity_bin_edges_v[:-1] + impulsivity_bin_edges_v[1:]) / 2)

    #60 Hz Background plotting Params:
    show_only_60hz_bg = False
    window_s = 20.0
    TS_cut_level = 0.15
    normalize_by_window_index = True
    plot_background_60hz_TS_hist = True
    bins_background_60hz_TS_hist = numpy.linspace(0,1,201) #Unsure what a good value is here yet.



    if len(sys.argv) >= 2:
        runs = numpy.array(sys.argv[1:],dtype=int)
        #runs = numpy.array(int(sys.argv[1]))
    else:
        runs = numpy.array([1642,1643,1644,1645,1646,1647])

    try:
        if plot_1dhists:
            hpol_counts = numpy.zeros((3,len(bins_1dhist)-1)) #3 rows, one for each trigger type.  
            vpol_counts = numpy.zeros((3,len(bins_1dhist)-1))
        if plot_2dhists:
            hv_counts = numpy.zeros((3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #v going to be plotted vertically, h horizontall, with 3 of such matrices representing the different trigger types.
        if plot_impulsivity:
            impulsivity_hv_counts = numpy.zeros((3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #v going to be plotted vertically, h horizontall, with 3 of such matrices representing the different trigger types.
            impulsivity_roi_counts = numpy.zeros((numpy.shape(roi)[0],3,len(bins_2dhist_v)-1,len(bins_2dhist_h)-1)) #A seperate histogram for each roi.  The above is a total histogram.
        if show_only_60hz_bg:
            print('ONLY SHOWING EVENTS THAT ARE EXPECTED TO BE 60HZ BACKGROUND SIGNALS')
        else:
            plot_background_60hz_TS_hist = False            

        if plot_background_60hz_TS_hist:
            background_60hz_TS_counts = numpy.zeros((3,len(bins_background_60hz_TS_hist)-1))

        for run in runs:
            #Prepare to load correlation values.
            reader = Reader(datapath,run)
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
            
            #Load correlation values            
            if filename is not None:
                with h5py.File(filename, 'a') as file:
                    eventids = file['eventids'][...]
                    dsets = list(file.keys()) #Existing datasets
                    try:
                        this_dset = 'bi-delta-curve-choice-%i'%curve_choice

                        event_times = file['calibrated_trigtime'][...]
                        trigger_type = file['trigger_type'][...]
                        output_correlation_values = file['cr_template_search'][this_dset][...]
                        if plot_impulsivity:
                            # impulsivity_dsets = list(file['impulsivity'].keys())
                            # print(impulsivity_dsets)
                            hpol_output_impulsivity = file['impulsivity'][impulsivity_dset]['hpol'][...]
                            vpol_output_impulsivity = file['impulsivity'][impulsivity_dset]['vpol'][...]


                        #Apply cuts
                        #These will be applied to all further calculations and are convenient for investigating particular sources. 
                        if show_only_60hz_bg:
                            trig2_cut = file['trigger_type'][...] == 2 #60Hz algorithm should only run on RF events.  Set others to -1.
                            metric = numpy.ones(len(trigger_type))*-1.0
                            metric[trig2_cut] = diffFromPeriodic(event_times[trig2_cut],window_s=window_s, normalize_by_window_index=normalize_by_window_index, plot_sample_hist=False)
                            show_only_cut = metric > TS_cut_level #Great than means it IS a 60Hz likely
                        else:
                            show_only_cut = numpy.ones(len(event_times),dtype=bool)
                        file.close()
                    except Exception as e:
                        print('Error loading data.')
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)

            max_output_correlation_values = numpy.max(output_correlation_values,axis=1)
            
            if numpy.any([plot_1dhists,plot_2dhists,plot_impulsivity,plot_background_60hz_TS_hist]):
                for trig_index in range(3):
                    trigger_cut = trigger_type == trig_index+1
                    trigger_cut_indices = numpy.where(numpy.logical_and(trigger_cut,show_only_cut))[0]
                    trigger_cut_indices_raw = numpy.where(trigger_cut)[0] #Without the selected background cut

                    max_output_correlation_values_h = numpy.max(output_correlation_values[trigger_cut_indices][:,[0,2,4,6]],axis=1)
                    max_output_correlation_values_v = numpy.max(output_correlation_values[trigger_cut_indices][:,[1,3,5,7]],axis=1)
                    if plot_1dhists:
                        hpol_counts[trig_index] += numpy.histogram(max_output_correlation_values_h,bins=bins_1dhist)[0]
                        vpol_counts[trig_index] += numpy.histogram(max_output_correlation_values_v,bins=bins_1dhist)[0]
                    if plot_2dhists:
                        hv_counts[trig_index] += numpy.histogram2d(max_output_correlation_values_h, max_output_correlation_values_v, bins = [bins_2dhist_h,bins_2dhist_v])[0].T 
                    if plot_impulsivity:
                        impulsivity_hv_counts[trig_index] += numpy.histogram2d(hpol_output_impulsivity[trigger_cut_indices], vpol_output_impulsivity[trigger_cut_indices], bins = [impulsivity_bin_edges_h,impulsivity_bin_edges_v])[0].T 
                        if plot_roi:
                            for roi_index, roi_coords in enumerate(roi):
                                roi_cut = numpy.logical_and(numpy.logical_and(max_output_correlation_values_h >= roi_coords[0], max_output_correlation_values_h <= roi_coords[1]),numpy.logical_and(max_output_correlation_values_v >= roi_coords[2], max_output_correlation_values_v <= roi_coords[3]))
                                roi_cut_indices = trigger_cut_indices[roi_cut]
                                # if roi_index == 0:
                                #     import pdb; pdb.set_trace()
                                impulsivity_roi_counts[roi_index][trig_index] += numpy.histogram2d(hpol_output_impulsivity[roi_cut_indices], vpol_output_impulsivity[roi_cut_indices], bins = [impulsivity_bin_edges_h,impulsivity_bin_edges_v])[0].T 
                    if plot_background_60hz_TS_hist:
                        background_60hz_TS_counts[trig_index] += numpy.histogram(metric[trigger_cut_indices_raw],bins=bins_background_60hz_TS_hist)[0]

        if plot_1dhists:
            summed_counts = hpol_counts + vpol_counts

            fig1, ax1 = plt.subplots()
            plt.title('Runs = %s'%str(runs))
            if show_only_60hz_bg == True:
                plt.title('Max Correlation Values\nBoth Polarizations\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]')
            else:
                plt.title('Max Correlation Values\nBoth Polarizations')
            ax1.bar(bins_1dhist[:-1], numpy.sum(summed_counts,axis=0), width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='All Triggers')
            plt.ylabel('Counts')
            plt.xlabel('Correlation Value with bi-delta CR Template')
            plt.legend(loc='upper left')

            fig2, ax2 = plt.subplots()
            if show_only_60hz_bg == True:
                plt.title('Runs = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%str(runs))
            else:
                plt.title('Runs = %s'%str(runs))
            if 1 in trigger_types:
                ax2.bar(bins_1dhist[:-1], summed_counts[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
            if 2 in trigger_types:
                ax2.bar(bins_1dhist[:-1], summed_counts[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
            if 3 in trigger_types:
                ax2.bar(bins_1dhist[:-1], summed_counts[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
            plt.legend(loc='upper left')
            plt.ylabel('Counts')
            plt.xlabel('Correlation Value with bi-delta CR Template')
            
            fig3 = plt.figure()
            if show_only_60hz_bg == True:
                plt.title('Runs = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%str(runs))
            else:
                plt.title('Runs = %s'%str(runs))
            ax_a = plt.subplot(2,1,1)
            for pol_index, pol in enumerate(['hpol','vpol']):
                ax_b = plt.subplot(2,1,pol_index+1,sharex=ax_a,sharey=ax_a)
                if pol == 'hpol':
                    max_output_correlation_values = hpol_counts
                else:
                    max_output_correlation_values = vpol_counts

                if plot_roi:
                    for roi_index, roi_coords in enumerate(roi): 
                        ax_b.axvspan(roi_coords[0+2*pol_index],roi_coords[1+2*pol_index],alpha=0.4, color=roi_colors[roi_index],label='roi %i'%roi_index)
                if 1 in trigger_types:
                    ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[0], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
                if 2 in trigger_types:
                    ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[1], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
                if 3 in trigger_types:
                    ax_b.bar(bins_1dhist[:-1], max_output_correlation_values[2], width=numpy.diff(bins_1dhist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)
                plt.ylabel('%s Counts'%pol.title())
                plt.xlabel('Correlation Value with bi-delta CR Template')

                plt.legend(loc='upper left')



        if plot_2dhists:
            for trig_index in range(3):
                if trig_index + 1 in trigger_types:
                    fig4, ax4 = plt.subplots()
                    if show_only_60hz_bg == True:
                        plt.title('bi-delta CR Template Correlations, Runs = %s\nTrigger = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%(str(runs),['Software','RF','GPS'][trig_index]))
                    else:
                        plt.title('bi-delta CR Template Correlations, Runs = %s\nTrigger = %s'%(str(runs),['Software','RF','GPS'][trig_index]))
                    im = ax4.pcolormesh(bin_centers_mesh_h, bin_centers_mesh_v, hv_counts[trig_index],norm=colors.LogNorm(vmin=0.5, vmax=hv_counts[trig_index].max()))#cmap=plt.cm.coolwarm
                    plt.xlabel('HPol Correlation Values')
                    plt.xlim(0,1)
                    plt.ylabel('VPol Correlation Values')
                    plt.ylim(0,1)
                    try:
                        cbar = fig4.colorbar(im)
                        cbar.set_label('Counts')
                    except Exception as e:
                        print('Error in colorbar, often caused by no events.')
                        print(e)
                    if plot_roi:
                        for roi_index, roi_coords in enumerate(roi): 
                            ax4.add_patch(Rectangle((roi_coords[0], roi_coords[2]), roi_coords[1] - roi_coords[0], roi_coords[3] - roi_coords[2],fill=False, edgecolor=roi_colors[roi_index]))
                            plt.text((roi_coords[1]+roi_coords[0])/2, roi_coords[3]+0.02,'roi %i'%roi_index,color=roi_colors[roi_index],fontweight='bold')

        if plot_impulsivity:
            for trig_index in range(3):
                if trig_index + 1 in trigger_types:
                    fig5, ax5 = plt.subplots()
                    if show_only_60hz_bg == True:
                        plt.title('bi-delta CR Impulsivity Values, Runs = %s\nTrigger = %s\n[ONLY SUSPECTED 60Hz BACKGROUND EVENTS]'%(str(runs),['Software','RF','GPS'][trig_index]))
                    else:
                        plt.title('bi-delta CR Impulsivity Values, Runs = %s\nTrigger = %s'%(str(runs),['Software','RF','GPS'][trig_index]))

                    im = ax5.pcolormesh(impulsivity_bin_centers_mesh_h, impulsivity_bin_centers_mesh_v, impulsivity_hv_counts[trig_index],norm=colors.LogNorm(vmin=0.5, vmax=impulsivity_hv_counts[trig_index].max()),cmap='Greys')#cmap=plt.cm.coolwarm
                    plt.xlabel('HPol Impulsivity Values')
                    #plt.xlim(0,1)
                    plt.ylabel('VPol Impulsivity Values')
                    #plt.ylim(0,1)
                    try:
                        cbar = fig5.colorbar(im)
                        cbar.set_label('Counts')
                    except Exception as e:
                        print('Error in colorbar, often caused by no events.')
                        print(e)
                    if plot_roi:
                        legend_properties = []
                        legend_labels = []
                        for roi_index, roi_coords in enumerate(roi):
                            levels = numpy.linspace(0,numpy.max(impulsivity_roi_counts[roi_index][trig_index]),6)[1:7] #Not plotting bottom contour because it is often background and clutters plot.
                            cs = ax5.contour(bin_centers_mesh_h, bin_centers_mesh_v, impulsivity_roi_counts[roi_index][trig_index], colors=[roi_colors[roi_index]],levels=levels)#,label='roi %i'%roi_index)
                            legend_properties.append(cs.legend_elements()[0][0])
                            legend_labels.append('roi %i'%roi_index)

                        plt.legend(legend_properties,legend_labels)

        if plot_background_60hz_TS_hist:
            fig_bg_60, ax_bg_60 = plt.subplots()
            plt.title('60 Hz Background Event Cut')
            for trig_index in range(3):
                if trig_index + 1 in trigger_types:
                    if 1 in trigger_types:
                        ax_bg_60.bar(bins_background_60hz_TS_hist[:-1], background_60hz_TS_counts[trig_index], width=numpy.diff(bins_background_60hz_TS_hist), edgecolor='black', align='edge',label='Trigger Type = Software',alpha=0.7,)
                    if 2 in trigger_types:
                        ax_bg_60.bar(bins_background_60hz_TS_hist[:-1], background_60hz_TS_counts[trig_index], width=numpy.diff(bins_background_60hz_TS_hist), edgecolor='black', align='edge',label='Trigger Type = RF',alpha=0.7,)
                    if 3 in trigger_types:
                        ax_bg_60.bar(bins_background_60hz_TS_hist[:-1], background_60hz_TS_counts[trig_index], width=numpy.diff(bins_background_60hz_TS_hist), edgecolor='black', align='edge',label='Trigger Type = GPS',alpha=0.7,)

            ax_bg_60.axvspan(TS_cut_level,max(bins_background_60hz_TS_hist[:-1]),color='g',label='Events Plotted',alpha=0.5)
            plt.legend()
            plt.xlabel('Test Statistic\n(Higher = More Likely 60 Hz)')
            plt.ylabel('Counts')
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    plt.close('all')
    #runs = [1642,1643,1644,1645,1646,1647]
    runs = [1644]
    for run in runs:
        reader = Reader(datapath,run)
        impulsivity_dset_key = 'LPf_None-LPo_None-HPf_None-HPo_None-Phase_0-Hilb_0-corlen_262144-align_0'
        print("Preparing dataSlicerSingleRun")
        ds = dataSlicerSingleRun(reader, impulsivity_dset_key, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False)

        print('Adding ROI to dataSlicerSingleRun')

        #ds.addROI('corr A',{'cr_template_search_h':[0.575,0.665],'cr_template_search_v':[0.385,0.46]})
        #ds.addROI('high v imp',{'impulsivity_h':[0.479,0.585],'impulsivity_h':[0.633,0.7]})
        #ds.addROI('small h.4 v.4 imp',{'impulsivity_h':[0.34,0.45],'impulsivity_h':[0.42,0.5]})

        ds.addROI('imp cluster',{'impulsivity_h':[0.35,0.46],'impulsivity_v':[0.45,0.50]})

        print('Generating plots:')
        for key_x, key_y in [['p2p_h','p2p_v'],['impulsivity_h','impulsivity_v'],['p2p_h','impulsivity_v']]:#ds.known_param_keys:
            print('Generating %s plot'%(key_x + ' vs ' + key_y))
            ds.plotROI2dHist(key_x, key_y, cmap='coolwarm', include_roi=True)
    
    #main() #This is the analysis before it was turned into a class.
    
  