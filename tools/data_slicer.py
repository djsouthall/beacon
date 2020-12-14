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

import numpy
import scipy
import scipy.signal
import scipy.signal

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile
from tools.fftmath import TemplateCompareTool
from tools.fftmath import FFTPrepper

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

datapath = os.environ['BEACON_DATA']

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
    def __init__(self,  reader, impulsivity_dset_key, time_delays_dset_key, \
                        curve_choice=0, trigger_types=[1,2,3],included_antennas=[0,1,2,3,4,5,6,7],include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=500,time_delays_n_bins_v=500,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=None,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=None):
        try:
            self.included_antennas = included_antennas#[0,1,2,3,4,5,6,7]
            self.included_hpol_antennas = numpy.array([0,2,4,6])[numpy.isin([0,2,4,6],self.included_antennas)]
            self.included_vpol_antennas = numpy.array([1,3,5,7])[numpy.isin([1,3,5,7],self.included_antennas)]

            #I want to work on adding: 'std', 'p2p', and 'snr', where snr is p2p/std.  I think these could be interesting, and are already available by default per signal. 
            #self.known_param_keys = ['impulsivity_hv', 'cr_template_search', 'std', 'p2p', 'snr'] #If it is not listed in here then it cannot be used.
            self.known_param_keys = [   'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
                                        'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
                                        'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
                                        'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish']
            self.updateReader(reader)
            self.tct = None #This will be defined when necessary by functions below. 



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

            self.trigger_types = trigger_types

            self.eventids_matching_trig = self.getEventidsFromTriggerType() #Opens file, so has to be called outside of with statement.
            
            #In progress.
            #ROI  List
            #This should be a list with coords x1,x2, y1, y2 for each row.  These will be used to produce plots pertaining to 
            #these specific regions of interest.  x and y should be correlation values, and ordered.
            self.roi = {}
            if include_test_roi:
                sample_ROI = self.printSampleROI(verbose=False)
                self.roi['test'] = sample_ROI

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
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

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
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getEventidsFromTriggerType(self):
        '''
        This will get the eventids that match one of the specified trigger types in self.trigger_types
        '''
        try:
            with h5py.File(self.analysis_filename, 'r') as file:
                #Initial cut based on trigger type.
                eventids = numpy.where(numpy.isin(file['trigger_type'][...],self.trigger_types))[0]
                file.close()
            return eventids
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getDataFromParam(self, eventids, param_key):
        '''
        Given eventids, this will load and return array for parameters associated with string param.
        '''
        try:
            if param_key in self.known_param_keys:
                with h5py.File(self.analysis_filename, 'r') as file:

                    if param_key == 'impulsivity_h':
                        param = file['impulsivity'][self.impulsivity_dset_key]['hpol'][...][eventids]
                    elif param_key == 'impulsivity_v':
                        param = file['impulsivity'][self.impulsivity_dset_key]['vpol'][...][eventids]
                    elif param_key == 'cr_template_search_h':
                        this_dset = 'bi-delta-curve-choice-%i'%self.cr_template_curve_choice
                        output_correlation_values = file['cr_template_search'][this_dset][...][eventids]
                        param = numpy.max(output_correlation_values[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD 
                    elif param_key == 'cr_template_search_v':
                        this_dset = 'bi-delta-curve-choice-%i'%self.cr_template_curve_choice
                        output_correlation_values = file['cr_template_search'][this_dset][...][eventids]
                        param = numpy.max(output_correlation_values[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                    elif param_key == 'std_h':
                        std = file['std'][...][eventids]
                        param = numpy.mean(std[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                    elif param_key == 'std_v':
                        std = file['std'][...][eventids]
                        param = numpy.mean(std[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                    elif param_key == 'p2p_h': 
                        p2p = file['p2p'][...][eventids]
                        param = numpy.max(p2p[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                    elif param_key == 'p2p_v': 
                        p2p = file['p2p'][...][eventids]
                        param = numpy.max(p2p[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                    elif param_key == 'snr_h':
                        std = file['std'][...][eventids]
                        param_1 = numpy.mean(std[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                        p2p = file['p2p'][...][eventids]
                        param_2 = numpy.max(p2p[:,self.included_hpol_antennas],axis=1) #hpol correlation values # SHOULD CONSIDER ADDING ANTENNA DEPENDANT CUTS TO THIS FOR TIMES WHEN ANTENNAS ARE DEAD
                        param = numpy.divide(param_2, param_1)
                    elif param_key == 'snr_v':
                        std = file['std'][...][eventids]
                        param_1 = numpy.mean(std[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                        p2p = file['p2p'][...][eventids]
                        param_2 = numpy.max(p2p[:,self.included_vpol_antennas],axis=1) #vpol_correlation values
                        param = numpy.divide(param_2, param_1)
                    elif 'time_delay_' in param_key:
                        split_param_key = param_key.split('_')
                        dset = '%spol_t_%ssubtract%s'%(split_param_key[3],split_param_key[2].split('subtract')[0],split_param_key[2].split('subtract')[1]) #Rewriting internal key name to time delay formatting.
                        param = file['time_delays'][self.time_delays_dset_key][dset][...][eventids]
                    elif 'cw_present' == param_key:
                        param = file['cw']['has_cw'][...][eventids].astype(int)
                    elif 'cw_freq_Mhz' == param_key:
                        param = file['cw']['freq_hz'][...][eventids]/1e6 #MHz
                    elif 'cw_linear_magnitude' == param_key:
                        param = file['cw']['linear_magnitude'][...][eventids]
                    elif 'cw_dbish' == param_key:
                        cw_dsets = list(file['cw'].keys())
                        if not numpy.isin('dbish',cw_dsets):
                            print('No stored dbish data from cw dataset, attempting to calculate from linear magnitude.')
                            if not hasattr(self, 'cw_prep'):
                                self.cw_prep = FFTPrepper(self.reader, final_corr_length=int(file['cw'].attrs['final_corr_length']), crit_freq_low_pass_MHz=float(file['cw'].attrs['crit_freq_low_pass_MHz']), crit_freq_high_pass_MHz=float(file['cw'].attrs['crit_freq_high_pass_MHz']), low_pass_filter_order=float(file['cw'].attrs['low_pass_filter_order']), high_pass_filter_order=float(file['cw'].attrs['high_pass_filter_order']), waveform_index_range=(None,None), plot_filters=False)
                                self.cw_prep.addSineSubtract(file['cw'].attrs['sine_subtract_min_freq_GHz'], file['cw'].attrs['sine_subtract_max_freq_GHz'], file['cw'].attrs['sine_subtract_percent'], max_failed_iterations=3, verbose=False, plot=False)
                            linear_magnitude = file['cw']['linear_magnitude'][...][eventids]
                            param = 10.0*numpy.log10( linear_magnitude**2 / len(self.cw_prep.t()) )
                        else:
                            param = file['cw']['dbish'][...][eventids]


                    file.close()
            else:
                print('\nWARNING!!!\nOther parameters have not been accounted for yet.\n%s'%(param_key))
            return param
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def getCutsFromROI(self,roi_key,load=False,save=False):
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
            if load == True:
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
                if roi_key in list(self.roi.keys()):
                    eventids = self.eventids_matching_trig.copy()
                    for param_key in list(self.roi[roi_key].keys()):
                        param = self.getDataFromParam(eventids, param_key)
                        #Reduce eventids by box cut
                        eventids = eventids[numpy.logical_and(param >= self.roi[roi_key][param_key][0], param < self.roi[roi_key][param_key][1])]  #Should get smaller/faster with every param cut.
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
                        print(e)
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        print(exc_type, fname, exc_tb.tb_lineno)
                    file.close() #want file to close where exception met or not.  
                print('Saving the cuts to the analysis file.')


            return eventids
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def get2dHistCounts(self, main_param_key_x, main_param_key_y, eventids, set_bins=True, load=False):
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
                        self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v = numpy.meshgrid((self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2, (self.current_bin_edges_y[:-1] + self.current_bin_edges_y[1:]) / 2)

                    counts = numpy.zeros(self.current_bin_centers_mesh_h.shape, dtype=int)
                    #Add counts based on stored indices.
                    numpy.add.at(counts,(bin_indices_y,bin_indices_x),1) #Need to double check this works as I expect, and is not transposed.  
                except Exception as e:
                    load = False
                    print('\nError in %s'%inspect.stack()[0][3])
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

            return counts

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getSingleParamPlotBins(self, param_key, eventids):
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
            if param_key in self.known_param_keys:
                #Append snr to the lists below only if max value isn't present.
                if self.max_snr_val is None:
                    std_requiring_params = ['std_h','std_v','snr_h','snr_v'] #List all params that might require max snr to be calculated if hard limit not given.
                    p2p_requiring_params = ['p2p_h','p2p_v','snr_h','snr_v'] #List all params that might require max p2p to be calculated if hard limit not given.
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

                    
            if param_key not in self.known_param_keys:
                print('WARNING!!!')
                print('Given key [%s] is not listed in known_param_keys:\n%s'%(param_key,str(self.known_param_keys)))
                return
            else:
                print('\tPreparing to get counts for %s'%param_key)

                calculate_bins_from_min_max = True #By default will be calculated at bottom of conditional list, unless a specific condition overrides.

                if param_key == 'impulsivity_h':
                    label = 'Impulsivity (hpol)'
                    x_n_bins = self.impulsivity_n_bins_h
                    x_max_val = 1
                    x_min_val = 0
                elif param_key == 'impulsivity_v':
                    label = 'Impulsivity (vpol)'
                    x_n_bins = self.impulsivity_n_bins_v
                    x_max_val = 1
                    x_min_val = 0
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
                    label = 'Mean SNR (hpol)\n P2P/STD'
                    x_n_bins = self.snr_n_bins_h
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif param_key == 'snr_v':
                    label = 'Mean SNR (vpol)\n P2P/STD'
                    x_n_bins = self.snr_n_bins_v
                    x_max_val = self.max_snr_val
                    x_min_val = 0
                elif 'time_delay_' in param_key:
                    split_param_key = param_key.split('_')
                    label = '%spol Time Delay\n Ant %s - Ant %s'%(split_param_key[3].title(),split_param_key[2].split('subtract')[0],split_param_key[2].split('subtract')[1])
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
                        if not numpy.isin('dbish',cw_dsets):
                            print('Creating FFTPrepper class to prepare CW bins.')
                            if not hasattr(self, 'cw_prep'):
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

            if calculate_bins_from_min_max:
                current_bin_edges = numpy.linspace(x_min_val,x_max_val,x_n_bins + 1) #These are bin edges

            return current_bin_edges, label
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

        TODO: Will need to add a load feature later. 
        '''
        self.current_bin_edges_x, self.current_label_x = self.getSingleParamPlotBins(main_param_key_x, eventids)
        self.current_bin_edges_y, self.current_label_y = self.getSingleParamPlotBins(main_param_key_y, eventids)

        self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v = numpy.meshgrid((self.current_bin_edges_x[:-1] + self.current_bin_edges_x[1:]) / 2, (self.current_bin_edges_y[:-1] + self.current_bin_edges_y[1:]) / 2)
        

    def plot2dHist(self, main_param_key_x,  main_param_key_y, eventids, title=None,cmap='coolwarm',load=False):
        '''
        This is meant to be a function the plot corresponding to the main parameter, and will plot the same quantity 
        (corresponding to main_param_key) with just events corresponding to the cut being used.  This subset will show
        up as a contour on the plot.  
        '''
        #Should make eventids a self.eventids so I don't need to call this every time.
        counts = self.get2dHistCounts(main_param_key_x,main_param_key_y,eventids,load=load,set_bins=True) #set_bins should only be called on first call, not on contours.
        
        _fig, _ax = plt.subplots()
        if title is not None:
            plt.title(title)
        else:
            plt.title('%s, Run = %i\nIncluded Triggers = %s'%(main_param_key_x + ' vs ' + main_param_key_y,int(self.reader.run),str(self.trigger_types)))

        _im = _ax.pcolormesh(self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v, counts,norm=colors.LogNorm(vmin=0.5, vmax=counts.max()),cmap=cmap)#cmap=plt.cm.coolwarm

        plt.xlabel(self.current_label_x)
        plt.ylabel(self.current_label_y)

        try:
            cbar = _fig.colorbar(_im)
            cbar.set_label('Counts')
        except Exception as e:
            print('Error in colorbar, often caused by no events.')
            print(e)

        return _fig, _ax

    def addContour(self, ax, main_param_key_x, main_param_key_y, contour_eventids, contour_color, load=False, n_contour=6, alpha=0.75):
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
            counts = self.get2dHistCounts(main_param_key_x, main_param_key_y, contour_eventids, load=load, set_bins=False)
            levels = numpy.linspace(0,numpy.max(counts),n_contour)[1:n_contour+1] #Not plotting bottom contour because it is often background and clutters plot.
            cs = ax.contour(self.current_bin_centers_mesh_h, self.current_bin_centers_mesh_v, counts, colors=[contour_color],levels=levels,alpha=alpha)
            return ax, cs
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            

    def plotROI2dHist(self, main_param_key_x, main_param_key_y, eventids=None, cmap='coolwarm', include_roi=True, load=False):
        '''
        This is the "do it all" function.  Given the parameter it will plot the 2dhist of the corresponding param by
        calling plot2dHist.  It will then plot the contours for each ROI on top.  It will do so assuming that each 
        ROI has a box cut in self.roi for EACH parameter.  I.e. it expects the # of listed entries in each ROI to be
        the same, and it will add a contour for the eventids that pass ALL cuts for that specific roi. 

        If eventids is given then then those events will be used to create the plot, and the trigger type cut will be ignored.
        '''
        if eventids is None:
            eventids = self.getEventidsFromTriggerType()
        fig, ax = self.plot2dHist(main_param_key_x, main_param_key_y, eventids, title=None, cmap=cmap) #prepares binning, must be called early (before addContour)

        #these few lines below this should be used for adding contours to the map. 
        if include_roi:
            legend_properties = []
            legend_labels = []

            for roi_index, roi_key in enumerate(list(self.roi.keys())):
                contour_eventids = self.getCutsFromROI(roi_key, load=load)

                ax, cs = self.addContour(ax, main_param_key_x, main_param_key_y, contour_eventids, self.roi_colors[roi_index], n_contour=6)
                legend_properties.append(cs.legend_elements()[0][0])
                legend_labels.append('roi %i: %s'%(roi_index, roi_key))

            plt.legend(legend_properties,legend_labels)

        return fig, ax


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

if __name__ == '__main__':
    plt.close('all')
    
    #runs = [1642,1643,1644,1645,1646,1647]
    runs = [1644]
    for run in runs:
        reader = Reader(datapath,run)
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

        # ds.saveHistogramData('p2p_h')
        # ds.saveHistogramData('p2p_v')
        # eventids = numpy.arange(ds.reader.N())
        # bin_indices_raw, bin_edges, label = ds.loadHistogramData('p2p_h',eventids)
        # bin_indices = ds.cleanHistogramData(bin_indices_raw, bin_edges)
    #main() #This is the analysis before it was turned into a class.
    
    
    #Sample data to test making 2dhists from stored bin numbers of 1dhists.
    # data_A = numpy.random.uniform(0,1,100)
    # data_B = numpy.random.uniform(0,1,100)**2 + 2
    # data_C = (numpy.random.uniform(0,1,100) - numpy.random.uniform(0,1,100))**2

    # bin_edges_A = numpy.linspace(0,1,101)
    # bin_edges_B = numpy.linspace(1,2,201)
    # bin_edges_C = numpy.linspace(0,1,51)

    # #This step is done at calculation phase
    # index_A = numpy.digitize(data_A,bins=bin_edges_A)
    # index_B = numpy.digitize(data_B,bins=bin_edges_B)
    # index_C = numpy.digitize(data_C,bins=bin_edges_C)

    # #This is done after loading to get back to histogram counts.  Below works for 1D hist.  Must develop and test for 2D hist. 
    # #https://stackoverflow.com/questions/56509600/can-numpy-add-at-be-used-with-2d-indices
    # masked_A = numpy.ma.masked_array(index_A,mask=numpy.logical_or(index_A == 0, index_A == len(bin_edges_A))) - 1 #Mask to ignore underflow/overflow bins.  Indices are base 1 here it seems.  
    # counts_A = numpy.zeros(len(bin_edges_A)-1,dtype=int); numpy.add.at(counts_A,masked_A,1)

    # hist_A = numpy.histogram(data_A,bin_edges_A)[0]