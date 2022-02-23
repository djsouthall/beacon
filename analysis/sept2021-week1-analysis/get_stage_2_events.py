#!/usr/bin/env python3
'''
This script is the same as inspect_above_horizon_events.py but specifically designed to reduce the clutter and just 
apply the already selected cuts on the data and save the eventids. 

This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import inspect
import h5py
import copy
from pprint import pprint
import time

import numpy
import scipy
import scipy.signal
import scipy.signal

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource
from beacon.tools.flipbook_reader import flipbookToDict
from beacon.tools.write_event_dict_to_excel import writeEventDictionaryToExcel

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)

def saveprint(*args, outfile=None):
    print(*args) #Print once to command line
    if outfile is not None:
        print(*args, file=open(outfile,'a')) #Print once to file

def savefig(fig, path, name, savefig_size_inches=(24,12.5), savefig_size_dpi=300, print_outfile=None):
    '''
    quick wrapper for saving figures.
    '''
    try:
        fig.set_size_inches(savefig_size_inches[0],savefig_size_inches[1])
        fig.savefig(os.path.join(path,'%s.png'%(name.replace('.png',''))), dpi=savefig_size_dpi, bbox_inches='tight')
    except Exception as e:
        if print_outfile is not None:
            saveprint('Failed to save ', os.path.join(path,'%s.png'%(name.replace('.png',''))), outfile=print_outfile)
        else:
            print('Failed to save ', os.path.join(path,'%s.png'%(name.replace('.png',''))))
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)


if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

    run_batches = {}

    run_batches['batch_test'] = numpy.arange(5733,5740)
    run_batches['batch_0'] = numpy.arange(5733,5974) # September data
    run_batches['batch_1'] = numpy.arange(5974,6073)
    run_batches['batch_2'] = numpy.arange(6074,6173)
    run_batches['batch_3'] = numpy.arange(6174,6273)
    run_batches['batch_4'] = numpy.arange(6274,6373)
    run_batches['batch_5'] = numpy.arange(6374,6473)
    run_batches['batch_6'] = numpy.arange(6474,6573)
    run_batches['batch_7'] = numpy.arange(6574,6673)

    ########
    batch_key = 'batch_1'
    ########


    out_path = os.path.join( os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'sept2021-week1-analysis', '%s_%i'%(batch_key, time.time())))
    output_text_file = os.path.join(out_path, 'output_%s.txt'%batch_key)

    os.mkdir(out_path)

    # This is a list of runs that were cancelled or ran out of time for whatever reason.  They should eventually be good 
    # to work with, but in the mean time they are to be ignored so they are not accessed by calculations happen.
    flawed_runs = numpy.array([])#numpy.array([6126,6277,6285])#numpy.array([5775,5981,5993,6033,6090,6520,6537,6538,6539]) 

    # These are runs that were processed correctly, but I choosing to ignore due to abnormal run behaviour.
    ignored_runs = numpy.array([])#numpy.array([6062,6063,6064]) 

    runs = run_batches[batch_key]
    runs = runs[numpy.logical_and(~numpy.isin(runs,flawed_runs),~numpy.isin(runs,ignored_runs))]

    saveprint("Preparing dataSlicer", outfile=output_text_file)
    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath, verbose_setup=False)
    saveprint('ds.data_slicers[0].constant_avg_rms_adu', outfile=output_text_file)
    saveprint(ds.data_slicers[0].constant_avg_rms_adu, outfile=output_text_file)

    #This one cuts out ALL events
    ds.addROI('above horizon only',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90]})
    ds.addROI('stage 1 cuts',{'elevation_best_choice':[10,90],'phi_best_choice':[-90,90],'similarity_count_h':[-0.1,10],'similarity_count_v':[-0.1,10],'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,10000],'impulsivity_hSLICERADDimpulsivity_v':[0.3,100],'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,100]})
    ds.addROI('stage 2 cuts',{'p2p_gap_h':[-1, 95], 'above_normalized_map_max_line':[0,10], 'above_snr_line':[0,10000]}) # Plus all cuts from stage 1
    ds.addROI('stage 2 partial cuts map',{'p2p_gap_h':[-1, 95], 'above_normalized_map_max_line':[0,10]}) # Events that pass only the map max cut
    ds.addROI('stage 2 partial cuts snr',{'p2p_gap_h':[-1, 95], 'above_snr_line':[0,10000]}) # Events that pass only the snr cut

    saveprint('ROI Dictionary definitions:', outfile=output_text_file)
    for key in list(ds.data_slicers[0].roi.keys()):
        saveprint('roi["%s"] = '%key, outfile=output_text_file)
        saveprint(str(ds.data_slicers[0].roi[key]), outfile=output_text_file)
    
    saveprint('Custom parameter functions defined in data slicers have the following descriptions:', outfile=output_text_file)
    for key in list(ds.data_slicers[0].parameter_functions.keys()):
        saveprint('parameter_functions["%s"]._description = '%key, outfile=output_text_file)
        saveprint(ds.data_slicers[0].parameter_functions[key]._description, outfile=output_text_file)

    # elevation best choice         :   [10,90]
    # phi best choice               :   [-90,90]
    # similarity count h            :   [-0.1,10]
    # similarity count v            :   [-0.1,10]
    # max(hpol peak to sidelobe , vpol peak to sidelobe)  :   [1.2,10000]
    # impulsivity h + impulsivity v                       :   [0.3,100]
    # max(cr template search h + cr template search v)    :   [0.4,100]

    return_successive_cut_counts = True
    return_total_cut_counts = True

    if return_successive_cut_counts and return_total_cut_counts:
        stage_1_eventids_dict, stage_1_successive_cut_counts, stage_1_total_cut_counts = ds.getCutsFromROI('stage 1 cuts',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    elif return_successive_cut_counts:
        stage_1_eventids_dict, stage_1_successive_cut_counts = ds.getCutsFromROI('stage 1 cuts',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    else:
        stage_1_eventids_dict = ds.getCutsFromROI('stage 1 cuts',load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    above_horizon_only_eventids_dict = ds.getCutsFromROI('above horizon only',load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)

    remove_from_stage_1_eventids_dict = {} #This will contain events from all clusters that are flagged and need to be removed.
    exclude_below_horizon_clusters = ( [[44.5,50], [-6,0]] , [[-11,-8], [-7,-3]] , [[-3,-1], [-12,0]] , [[24.5,28], [-7.75,-1]] , [[30.75,30], [-6,-1]] , [[6,8.5], [-12,-4]] )

    saveprint('exclude_below_horizon_clusters', outfile=output_text_file)
    saveprint(str(exclude_below_horizon_clusters), outfile=output_text_file)

    for remove_box_az, remove_el in exclude_below_horizon_clusters:
        cluster_cut_dict = {}#copy.deepcopy(ds.roi['stage 1 cuts'])
        cluster_cut_dict['phi_best_all_belowhorizon'] = remove_box_az
        cluster_cut_dict['elevation_best_all_belowhorizon'] = remove_el
        ds.addROI('below horizon cluster',cluster_cut_dict)

        # Cut on just direction box for cluster cut, rather than full box cut so less calculations performed.
        # Then get the evetns from that which are also in stage_1_eventids_dict
        single_cluster_eventids_dict = ds.getCutsFromROI('below horizon cluster', eventids_dict=copy.deepcopy(stage_1_eventids_dict), load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)

        # Add these events to the "to be removed" dict. returnUniqueEvents is similar to appending
        remove_from_stage_1_eventids_dict = ds.returnUniqueEvents(
            copy.deepcopy(remove_from_stage_1_eventids_dict),
            copy.deepcopy(single_cluster_eventids_dict)
            )

    #Make remove_from_stage_1_eventids_dict only events that will be removed
    remove_from_stage_1_eventids_dict = ds.returnCommonEvents(
        copy.deepcopy(stage_1_eventids_dict),
        copy.deepcopy(remove_from_stage_1_eventids_dict)
        )
    
    #Actually remove those events
    stage_1_eventids_dict = ds.returnEventsAWithoutB(
        copy.deepcopy(stage_1_eventids_dict),
        copy.deepcopy(remove_from_stage_1_eventids_dict)
        )

    # At this point the stage 1 cuts are finalized
    stage_1_eventids_array = ds.organizeEventDict(copy.deepcopy(stage_1_eventids_dict))
    remove_from_above_horizon_array = ds.organizeEventDict(copy.deepcopy(remove_from_stage_1_eventids_dict))

    # Get stage 2 cuts
    if return_successive_cut_counts and return_total_cut_counts:
        stage_2_eventids_dict, stage_2_successive_cut_counts, stage_2_total_cut_counts = ds.getCutsFromROI('stage 2 cuts',eventids_dict=copy.deepcopy(stage_1_eventids_dict), load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    elif return_successive_cut_counts:
        stage_2_eventids_dict, stage_2_successive_cut_counts = ds.getCutsFromROI('stage 2 cuts',eventids_dict=copy.deepcopy(stage_1_eventids_dict), load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)
    else:
        stage_2_eventids_dict = ds.getCutsFromROI('stage 2 cuts',eventids_dict=copy.deepcopy(stage_1_eventids_dict), load=False,save=False,verbose=False, return_successive_cut_counts=return_successive_cut_counts, return_total_cut_counts=return_total_cut_counts)

    # stage_2_eventids_dict is events that pass both cuts
    # partial_pass_stage_2_eventids_dict is events that pass only the map cut, but not both.  
    partial_pass_stage_2_eventids_dict = ds.getCutsFromROI('stage 2 partial cuts map',eventids_dict=copy.deepcopy(stage_1_eventids_dict), load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
    partial_pass_stage_2_eventids_dict = ds.returnEventsAWithoutB(copy.deepcopy(partial_pass_stage_2_eventids_dict), stage_2_eventids_dict) #Only want events that pass just the one cut
    
    # partial_fail_stage_2_eventids_dict is events that pass only the SNR cut, but not both.
    partial_fail_stage_2_eventids_dict = ds.getCutsFromROI('stage 2 partial cuts snr',eventids_dict=copy.deepcopy(stage_1_eventids_dict), load=False,save=False,verbose=False, return_successive_cut_counts=False, return_total_cut_counts=False)
    partial_fail_stage_2_eventids_dict = ds.returnEventsAWithoutB(copy.deepcopy(partial_fail_stage_2_eventids_dict), copy.deepcopy(stage_2_eventids_dict)) #Only want events that pass just the one cut

    failed_stage_2_eventids_dict = ds.returnEventsAWithoutB(copy.deepcopy(stage_1_eventids_dict), copy.deepcopy(stage_2_eventids_dict)) #Stage 1 passing minus passed events
    failed_stage_2_eventids_dict = ds.returnEventsAWithoutB(copy.deepcopy(failed_stage_2_eventids_dict), copy.deepcopy(partial_pass_stage_2_eventids_dict)) #Stage 1 passing minus passed events AND partial passed events

    stage_2_eventids_array = ds.organizeEventDict(stage_2_eventids_dict)

    if True:

        # ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=None, include_roi=False)
        fig_all_event_map_h, ax_all_event_map_h = ds.plotROI2dHist('phi_best_h','elevation_best_h', cmap=cmap, eventids_dict=None, include_roi=False)
        savefig(fig_all_event_map_h, out_path, 'map_h_all_events')

        fig_all_event_map_v, ax_all_event_map_v = ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=None, include_roi=False)
        savefig(fig_all_event_map_v, out_path, 'map_v_all_events')

        fig_direction_box_map_h, ax_direction_box_map_h = ds.plotROI2dHist('phi_best_h','elevation_best_h', cmap=cmap, eventids_dict=copy.deepcopy(above_horizon_only_eventids_dict), include_roi=False)
        savefig(fig_direction_box_map_h, out_path, 'map_h_direction_box_events')

        fig_direction_box_map_v, ax_direction_box_map_v = ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=copy.deepcopy(above_horizon_only_eventids_dict), include_roi=False)
        savefig(fig_direction_box_map_v, out_path, 'map_v_direction_box_events')

        #Plot event directions
        fig_all_events, ax_all_events = ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=None, include_roi=False)
        if fig_all_events._suptitle is not None:
            fig_all_events.suptitle('all_events\n'.replace('_',' ').title()+fig_all_events._suptitle.get_text())
        else:
            fig_all_events.suptitle('all_events\n'.replace('_',' ').title())
        savefig(fig_all_events, out_path, 'map_choice_all_events')    

        fig_direction_box, ax_direction_box = ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=copy.deepcopy(above_horizon_only_eventids_dict), include_roi=False)
        if fig_direction_box._suptitle is not None:
            fig_direction_box.suptitle('direction_box\n'.replace('_',' ').title()+fig_direction_box._suptitle.get_text())
        else:
            fig_direction_box.suptitle('direction_box\n'.replace('_',' ').title())
        savefig(fig_direction_box, out_path, 'map_choice_only_direction_box')

        fig_stage_1, ax_stage_1 = ds.plotROI2dHist('phi_best_choice','elevation_best_choice', cmap=cmap, eventids_dict=copy.deepcopy(stage_1_eventids_dict), include_roi=False)
        if fig_stage_1._suptitle is not None:
            fig_stage_1.suptitle('stage_1\n'.replace('_',' ').title()+fig_stage_1._suptitle.get_text())
        else:
            fig_stage_1.suptitle('stage_1\n'.replace('_',' ').title())
        savefig(fig_stage_1, out_path, 'map_choice_stage_1')

        fig_box_cut, ax_box_cut = ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=copy.deepcopy(remove_from_stage_1_eventids_dict), include_roi=False)
        if fig_box_cut._suptitle is not None:
            fig_box_cut.suptitle('box_cut\n'.replace('_',' ').title()+fig_box_cut._suptitle.get_text())
        else:
            fig_box_cut.suptitle('box_cut\n'.replace('_',' ').title())
        savefig(fig_box_cut, out_path, 'map_choice_excluded_by_belowhorizon_boxes')


        # Stage 2 plots
        fig_stage_2_pass_cut, ax_stage_2_pass_cut = ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=copy.deepcopy(stage_2_eventids_dict), include_roi=False)
        if fig_stage_2_pass_cut._suptitle is not None:
            fig_stage_2_pass_cut.suptitle('stage_2_pass\n'.replace('_',' ').title()+fig_stage_2_pass_cut._suptitle.get_text())
        else:
            fig_stage_2_pass_cut.suptitle('stage_2_pass\n'.replace('_',' ').title())
        savefig(fig_stage_2_pass_cut, out_path, 'map_choice_pass_stage_2')

        fig_stage_2_partial_pass_cut, ax_stage_2_partial_pass_cut = ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=copy.deepcopy(partial_pass_stage_2_eventids_dict), include_roi=False)
        if fig_stage_2_partial_pass_cut._suptitle is not None:
            fig_stage_2_partial_pass_cut.suptitle('stage_2_partial_pass\n'.replace('_',' ').title()+fig_stage_2_partial_pass_cut._suptitle.get_text())
        else:
            fig_stage_2_partial_pass_cut.suptitle('stage_2_partial_pass\n'.replace('_',' ').title())
        savefig(fig_stage_2_partial_pass_cut, out_path, 'map_choice_partial_pass_stage_2')

        fig_stage_2_failed_cut, ax_stage_2_failed_cut = ds.plotROI2dHist('phi_best_v','elevation_best_v', cmap=cmap, eventids_dict=copy.deepcopy(failed_stage_2_eventids_dict), include_roi=False)
        if fig_stage_2_failed_cut._suptitle is not None:
            fig_stage_2_failed_cut.suptitle('stage_2_failed\n'.replace('_',' ').title()+fig_stage_2_failed_cut._suptitle.get_text())
        else:
            fig_stage_2_failed_cut.suptitle('stage_2_failed\n'.replace('_',' ').title())
        savefig(fig_stage_2_failed_cut, out_path, 'map_choice_fail_stage_2')


        
        if len(stage_1_eventids_array) > 0 or False:
            plot_params = [['std_h', 'std_v'], ['p2p_h', 'p2p_v'], ['snr_h', 'snr_v'], ['csnr_h', 'csnr_v'], ['hpol_peak_to_sidelobe','vpol_peak_to_sidelobe'], ['cr_template_search_h', 'cr_template_search_v'], ['impulsivity_h','impulsivity_v'], ['hpol_normalized_map_value_abovehorizon','vpol_normalized_map_value_abovehorizon']]
            
            saveprint('Generating plots:', outfile=output_text_file)

            for key_x, key_y in plot_params:
                saveprint('Generating stage 1 %s plot'%(key_x + ' vs ' + key_y), outfile=output_text_file)
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=copy.deepcopy(stage_1_eventids_dict), include_roi=False)
                savefig(fig, out_path, '%s_vs_%s_stage_1'%(key_x, key_y))
                plt.close(fig)

                saveprint('Generating stage 2 pass %s plot'%(key_x + ' vs ' + key_y), outfile=output_text_file)
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=copy.deepcopy(stage_2_eventids_dict), include_roi=False)
                savefig(fig, out_path, '%s_vs_%s_stage_2_pass'%(key_x, key_y))
                plt.close(fig)

                saveprint('Generating stage 2 partial pass %s plot'%(key_x + ' vs ' + key_y), outfile=output_text_file)
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=copy.deepcopy(partial_pass_stage_2_eventids_dict), include_roi=False)
                savefig(fig, out_path, '%s_vs_%s_stage_2_partial_pass'%(key_x, key_y))
                plt.close(fig)

                saveprint('Generating stage 2 fail %s plot'%(key_x + ' vs ' + key_y), outfile=output_text_file)
                fig, ax = ds.plotROI2dHist(key_x, key_y, cmap=cmap, eventids_dict=copy.deepcopy(failed_stage_2_eventids_dict), include_roi=False)
                savefig(fig, out_path, '%s_vs_%s_stage_2_fail'%(key_x, key_y))
                plt.close(fig)


    if return_successive_cut_counts:
        roi_key = 'stage 1 cuts'
        for key in list(stage_1_successive_cut_counts.keys()):
            if key == 'initial':
                saveprint('Initial Event Count is %i'%(stage_1_successive_cut_counts[key]), outfile=output_text_file)
            else:
                saveprint('%0.3f%% events then cut by %s with bounds %s'%(100*(previous_count-stage_1_successive_cut_counts[key])/previous_count , key, str(ds.roi[roi_key][key])), outfile=output_text_file)
                saveprint('%0.3f%% of initial events would be cut by %s with bounds %s'%(100*(stage_1_total_cut_counts['initial']-stage_1_total_cut_counts[key])/stage_1_total_cut_counts['initial'] , key, str(ds.roi[roi_key][key])), outfile=output_text_file)
                saveprint('\nRemaining Events After Step %s is %i'%(key, stage_1_successive_cut_counts[key]), outfile=output_text_file)
            previous_count = stage_1_successive_cut_counts[key]

        saveprint('', outfile=output_text_file)
        saveprint('%0.3f%% events then cut by targeted below horizon box cuts'%(100*(previous_count-len(stage_1_eventids_array))/previous_count), outfile=output_text_file)
        saveprint('Double checking above math, %i events cut by targeted below horizon box cuts'%(len(remove_from_above_horizon_array)), outfile=output_text_file)
        saveprint('Final number of events remaining is %i'%len(stage_1_eventids_array), outfile=output_text_file)


        roi_key = 'stage 2 cuts'
        for key in list(stage_2_successive_cut_counts.keys()):
            if key == 'initial':
                saveprint('Initial Event Count is %i'%(stage_2_successive_cut_counts[key]), outfile=output_text_file)
            else:
                saveprint('%0.3f%% events then cut by %s with bounds %s'%(100*(previous_count-stage_2_successive_cut_counts[key])/previous_count , key, str(ds.roi[roi_key][key])), outfile=output_text_file)
                saveprint('%0.3f%% of initial events would be cut by %s with bounds %s'%(100*(stage_2_total_cut_counts['initial']-stage_2_total_cut_counts[key])/stage_2_total_cut_counts['initial'] , key, str(ds.roi[roi_key][key])), outfile=output_text_file)
                saveprint('\nRemaining Events After Step %s is %i'%(key, stage_2_successive_cut_counts[key]), outfile=output_text_file)
            previous_count = stage_2_successive_cut_counts[key]

        saveprint('', outfile=output_text_file)
        saveprint('%0.3f%% events then cut by targeted below horizon box cuts'%(100*(previous_count-len(stage_2_eventids_array))/previous_count), outfile=output_text_file)
        saveprint('Double checking above math, %i events cut by targeted below horizon box cuts'%(len(remove_from_above_horizon_array)), outfile=output_text_file)
        saveprint('Final number of events remaining is %i'%len(stage_2_eventids_array), outfile=output_text_file)



    # impulsivity_h = ds.getDataArrayFromParam('impulsivity_h', trigger_types=None, eventids_dict=stage_1_eventids_dict)
    # impulsivity_v = ds.getDataArrayFromParam('impulsivity_v', trigger_types=None, eventids_dict=stage_1_eventids_dict)
    # impulsivity = impulsivity_h + impulsivity_v
    # most_impulsive_events = numpy.lib.recfunctions.rec_append_fields(stage_1_eventids_array, ['impulsivity_h', 'impulsivity_v'], [impulsivity_h, impulsivity_v])[[numpy.argsort(impulsivity)[::-1]]]

    #Save the eventids dict of stage 1 passing events.

    outfile_name = os.path.join(out_path, 'stage_1_eventids_dict_%s.npy'%batch_key)
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving stage_1_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, stage_1_eventids_dict, allow_pickle=True)



    #This should include the information of which ranges of parameters are included and excluded, defining the box used.
    above_horizon_description = {}
    above_horizon_description['include'] = copy.deepcopy(ds.roi['stage 1 cuts'])
    above_horizon_description['exclude'] = {}

    for cluster_index, (remove_box_az, remove_el) in enumerate(exclude_below_horizon_clusters):
        cluster_cut_dict = {}
        cluster_cut_dict['phi_best_all_belowhorizon'] = remove_box_az
        cluster_cut_dict['elevation_best_all_belowhorizon'] = remove_el
        above_horizon_description['exclude']['belowhorizon_cluster_%i'%cluster_index] = copy.deepcopy(cluster_cut_dict)

    outfile_name = os.path.join(out_path, 'stage_1_cut_definition_dict_%s.npy'%batch_key)
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving stage_1_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, above_horizon_description, allow_pickle=True)



    #Save the eventids dict of stage 2 passing events.

    outfile_name = os.path.join(out_path, 'stage_2_eventids_dict_%s.npy'%batch_key)
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving stage_2_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, stage_2_eventids_dict, allow_pickle=True)

    #Save the eventids dict of stage 2 partial passing events.

    outfile_name = os.path.join(out_path, 'partial_pass_stage_2_eventids_dict_%s.npy'%batch_key)
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving partial_pass_stage_2_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, partial_pass_stage_2_eventids_dict, allow_pickle=True)

    #Save the eventids dict of stage 2 failed events.

    outfile_name = os.path.join(out_path, 'failed_stage_2_eventids_dict_%s.npy'%batch_key)
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving failed_stage_2_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, failed_stage_2_eventids_dict, allow_pickle=True)



    #This should include the information of which ranges of parameters are included and excluded, defining the box used.
    above_horizon_description = {}
    above_horizon_description['include'] = copy.deepcopy(ds.roi['stage 2 cuts'])
    above_horizon_description['exclude'] = {}

    for cluster_index, (remove_box_az, remove_el) in enumerate(exclude_below_horizon_clusters):
        cluster_cut_dict = {}
        cluster_cut_dict['phi_best_all_belowhorizon'] = remove_box_az
        cluster_cut_dict['elevation_best_all_belowhorizon'] = remove_el
        above_horizon_description['exclude']['belowhorizon_cluster_%i'%cluster_index] = copy.deepcopy(cluster_cut_dict)

    outfile_name = os.path.join(out_path, 'stage_2_cut_definition_dict_%s.npy'%batch_key)
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving stage_2_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, above_horizon_description, allow_pickle=True)


    if True:
        excel_filename = os.path.join(out_path, 'event_info_%s.xlsx'%batch_key)
        saveprint('Generating excel file %s'%excel_filename, outfile=output_text_file)
        writeEventDictionaryToExcel(stage_2_eventids_dict, excel_filename, 'stage_2_pass', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(partial_pass_stage_2_eventids_dict, excel_filename, 'stage_2_partial', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(failed_stage_2_eventids_dict, excel_filename, 'stage_2_fail', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(stage_1_eventids_dict, excel_filename, 'stage_1_eventids_dict', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(remove_from_stage_1_eventids_dict, excel_filename, 'excluded_by_belowhorizon_boxes', ds=ds, include_airplanes=False)

        
        master_excel_filename = os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'sept2021-week1-analysis', 'event_info_all_batches.xlsx')
        writeEventDictionaryToExcel(stage_2_eventids_dict, master_excel_filename, batch_key + '_stage_2_pass', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(partial_pass_stage_2_eventids_dict, master_excel_filename, batch_key + '_stage_2_partial', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(failed_stage_2_eventids_dict, master_excel_filename, batch_key + '_stage 2 fail', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(stage_1_eventids_dict, master_excel_filename, batch_key + '_stage_1_eventids_dict', ds=ds, include_airplanes=False)
        writeEventDictionaryToExcel(remove_from_stage_1_eventids_dict, master_excel_filename, batch_key + '_excluded_by_belowhorizon_boxes', ds=ds, include_airplanes=False)


    # saveprint('Generating event inspector and saving event plots.', outfile=output_text_file)
    # ds.eventInspector(stage_1_eventids_dict, savedir=out_path, savename_prepend='stage_1_')
    # ds.inspector_tm.trigger_tool('Save Book')
    # ds.delInspector()

    if True:
        saveprint('Generating event inspector and saving event plots.', outfile=output_text_file)
        ds.eventInspector(stage_2_eventids_dict, savedir=out_path, savename_prepend='stage_2_pass_')
        ds.inspector_tm.trigger_tool('Save Book')
        ds.delInspector()

        saveprint('Generating event inspector and saving event plots.', outfile=output_text_file)
        ds.eventInspector(partial_pass_stage_2_eventids_dict, savedir=out_path, savename_prepend='stage_2_partial_pass_')
        ds.inspector_tm.trigger_tool('Save Book')
        ds.delInspector()

        saveprint('Generating event inspector and saving event plots.', outfile=output_text_file)
        ds.eventInspector(failed_stage_2_eventids_dict, savedir=out_path, savename_prepend='stage_2_fail_')
        ds.inspector_tm.trigger_tool('Save Book')
        ds.delInspector()

    #Add in further sorting of events and automated map generation for each event. 

    saveprint('EOF', outfile=output_text_file)

