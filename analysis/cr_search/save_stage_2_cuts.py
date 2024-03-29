#!/usr/bin/env python3
'''
This script is intended to look at the events that construct best above horizon in allsky maps.  I want to view the
peak to sidelobe values and other parameters for the belowhorizon and abovehorizon maps and determine if there is
an obvious cut for which sidelobed above horizon events can be discriminated. 
'''

import sys
import os
import gc
import inspect
import h5py
import copy
from pprint import pprint
import textwrap
from multiprocessing import Process

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
from beacon.tools.get_sun_coords_from_timestamp import getSunElWeightsFromRunDict

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, ScalarFormatter, FuncFormatter
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz

plt.ion()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)

gc.enable()

'''
# cut_name = cut_dict_key + '-' + param_key

# ds.setCurrentPlotBins(param_key, param_key, None)
# current_label = ds.current_label_x.replace('\n','  ')
# sub_eventids_dict = ds.getCutsFromROI(cut_name,load=False,save=False,verbose=True, return_successive_cut_counts=False, return_total_cut_counts=False) #Events passing all cuts BUT current key

# _runs = numpy.array(list(sub_eventids_dict.keys()))
# for key in _runs:
#     sub_eventids_dict[str(key)] = sub_eventids_dict.pop(key)

# counts, bin_centers, bin_width = ds.plot1dHist(param_key, None, return_only_count_details=True)
# sub_eventids_dict['bin_centers'] = bin_centers
# sub_eventids_dict['bin_width'] = bin_width
# sub_eventids_dict['counts'] = counts.astype(int)
# sub_eventids_dict['label'] = current_label
# sub_eventids_dict['cut_dict'] = str(ds.roi[cut_name])
# sub_eventids_dict['included_runs'] = runs

# numpy.savez_compressed(os.path.join(out_dir, '%s.npz'%cut_name), **sub_eventids_dict)


# #How to load the data back
# if False:
#     loaded_eventids_dict = {}
#     hist_data = {}
#     with numpy.load(os.path.join(out_dir, '%s.npz'%cut_name)) as data:
#         for key in list(data.keys()):
#             if key.isdigit():
#                 loaded_eventids_dict[int(key)] = data[key]
#             else:
#                 hist_data[key] = data[key]


# for key in numpy.array(list(sub_eventids_dict.keys())):
#     del sub_eventids_dict[key] 
# del sub_eventids_dict
# del counts
# del bin_centers
# del bin_width
# gc.collect()
'''

def saveprint(*args, outfile=None):
    print(*args) #Print once to command line
    if outfile is not None:
        print(*args, file=open(outfile,'a')) #Print once to file

if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'

    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    if True:
        #These batches skip ~1% of runs, but it is what I acutally analyzed so it is what I am presenting the stats for. 
        run_batches = {}
        run_batches['batch_0'] = numpy.arange(5733,5974) # September data, should setup to auto add info to the "notes" section based off of existing sorting, and run this one those events for consistency
        run_batches['batch_1'] = numpy.arange(5974,6073)
        run_batches['batch_2'] = numpy.arange(6074,6173)
        run_batches['batch_3'] = numpy.arange(6174,6273)
        run_batches['batch_4'] = numpy.arange(6274,6373)
        run_batches['batch_5'] = numpy.arange(6374,6473)
        run_batches['batch_6'] = numpy.arange(6474,6573)
        run_batches['batch_7'] = numpy.arange(6574,6641)

        batch = 0
        runs = run_batches['batch_%i'%batch]
    elif False:
        batch = 0
        batch_length = 150
        runs = 5733 + numpy.arange(batch_length) + batch_length*batch
        # runs = numpy.arange(5733,6641)
        runs = runs[runs<=6640]
    elif True:
        runs = numpy.arange(5733,6641)
    elif True:
        #Quicker for testing
        runs = numpy.arange(5910,5912)
    elif False:
        runs = numpy.arange(5733,5974)
        runs = runs[runs != 5864]
    else:
        runs = numpy.arange(5910,5934)#numpy.arange(5733,5974)#numpy.arange(5910,5912)

    print("Preparing dataSlicer")

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, \
                    low_ram_mode=False,\
                    analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    cut_dict_key = 'all_cuts'
    ds.addROI(cut_dict_key, {   'elevation_best_choice':[10,90],
                                'phi_best_choice':[-90,90],
                                'similarity_count_h':[0,10],
                                'similarity_count_v':[0,10],
                                'hpol_peak_to_sidelobeSLICERADDvpol_peak_to_sidelobe':[2.15,1e10],
                                'impulsivity_hSLICERADDimpulsivity_v':[0.4,1e10],
                                'cr_template_search_hSLICERADDcr_template_search_v':[0.8,1e10],
                                'p2p_gap_h':[-1e10, 95],
                                'above_normalized_map_max_line':[0,1e10],
                                'above_snr_line':[0,1e10],
                                'in_targeted_box':[-0.5, 0.5]})


    for param_key in list(ds.roi[cut_dict_key].keys()):
        copied_dict = copy.deepcopy(ds.roi[cut_dict_key])
        del copied_dict[param_key]
        ds.addROI(cut_dict_key + '-' + param_key, copy.deepcopy(copied_dict)) #Add an ROI containing all cuts BUT the current key (used as ROI label).


    out_dir = './cuts_run%i-run%i_%i'%(min(runs), max(runs), time.time())
    os.mkdir(out_dir)


    pass_all_cuts_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI(cut_dict_key,load=False,save=False,verbose=True, return_successive_cut_counts=True, return_total_cut_counts=True) #Events passing all cuts


    '''
    I want to add in saving the same histograms with NO cuts applied and ALL cuts applied, as well as sequentially in
    the order they are made.  Can I store the information for 2d histograms this way as well?  I much prefer them.
    '''

    # Save histograms with no cuts

    for param_key in list(ds.roi[cut_dict_key].keys()):
        if param_key in ['elevation_best_choice', 'phi_best_choice']:
            continue

        cut_name = cut_dict_key + '-' + param_key
        ds.setCurrentPlotBins(param_key, param_key, None)
        current_label = ds.current_label_x.replace('\n','  ')

        # Save histograms with all cuts BUT specified cut
        sub_eventids_dict = ds.getCutsFromROI(cut_name,load=False,save=False,verbose=True, return_successive_cut_counts=False, return_total_cut_counts=False) #Events passing all cuts BUT current key

        counts, bin_centers, bin_width = ds.plot1dHist(param_key, sub_eventids_dict, return_only_count_details=True)

        _runs = numpy.array(list(sub_eventids_dict.keys()))
        for key in _runs:
            sub_eventids_dict[str(key)] = sub_eventids_dict.pop(key)
        sub_eventids_dict['bin_centers'] = bin_centers
        sub_eventids_dict['bin_width'] = bin_width
        sub_eventids_dict['counts'] = counts.astype(int)
        sub_eventids_dict['label'] = current_label
        sub_eventids_dict['cut_dict'] = str(ds.roi[cut_name])
        sub_eventids_dict['included_runs'] = runs

        numpy.savez_compressed(os.path.join(out_dir, 'hist_for_%s_with_all_cuts_but_%s.npz'%(param_key, param_key)), **sub_eventids_dict)


        # Save histograms after all cuts
        for key in numpy.array(list(sub_eventids_dict.keys())):
            del sub_eventids_dict[key] 
        del sub_eventids_dict
        del counts
        del bin_centers
        del bin_width
        gc.collect()

        counts, bin_centers, bin_width = ds.plot1dHist(param_key, pass_all_cuts_eventids_dict, return_only_count_details=True)

        sub_eventids_dict = copy.deepcopy(pass_all_cuts_eventids_dict)
        _runs = numpy.array(list(sub_eventids_dict.keys()))

        for key in _runs:
            sub_eventids_dict[str(key)] = sub_eventids_dict.pop(key)

        sub_eventids_dict['bin_centers'] = bin_centers
        sub_eventids_dict['bin_width'] = bin_width
        sub_eventids_dict['counts'] = counts.astype(int)
        sub_eventids_dict['label'] = current_label
        sub_eventids_dict['cut_dict'] = str(ds.roi[cut_dict_key])
        sub_eventids_dict['included_runs'] = runs

        numpy.savez_compressed(os.path.join(out_dir, 'hist_for_%s_with_all_cuts.npz'%(param_key)), **sub_eventids_dict)

        # Save histograms after no cuts
        for key in numpy.array(list(sub_eventids_dict.keys())):
            del sub_eventids_dict[key] 
        del sub_eventids_dict
        del counts
        del bin_centers
        del bin_width
        gc.collect()

        counts, bin_centers, bin_width = ds.plot1dHist(param_key, None, return_only_count_details=True)

        sub_eventids_dict = {}
        sub_eventids_dict['bin_centers'] = bin_centers
        sub_eventids_dict['bin_width'] = bin_width
        sub_eventids_dict['counts'] = counts.astype(int)
        sub_eventids_dict['label'] = current_label
        sub_eventids_dict['cut_dict'] = str({})
        sub_eventids_dict['included_runs'] = runs

        numpy.savez_compressed(os.path.join(out_dir, 'hist_for_%s_with_no_cuts.npz'%(param_key)), **sub_eventids_dict)

        for key in numpy.array(list(sub_eventids_dict.keys())):
            del sub_eventids_dict[key] 
        del sub_eventids_dict
        del counts
        del bin_centers
        del bin_width
        gc.collect()



    output_text_file = os.path.join(out_dir, 'output_histogram_saving_%i.txt'%time.time())

    for key in list(successive_cut_counts.keys()):
        if key == 'initial':
            saveprint('Initial Event Count is %i'%(successive_cut_counts[key]), outfile=output_text_file)
        else:
            saveprint('%0.3f%% events then cut by %s with bounds %s'%(100*(previous_count-successive_cut_counts[key])/previous_count , key, str(ds.roi[cut_dict_key][key])), outfile=output_text_file)
            saveprint('%0.3f%% of initial events would be cut by %s with bounds %s'%(100*(total_cut_counts['initial']-total_cut_counts[key])/total_cut_counts['initial'] , key, str(ds.roi[cut_dict_key][key])), outfile=output_text_file)
            saveprint('\nRemaining Events After Step %s is %i'%(key, successive_cut_counts[key]), outfile=output_text_file)
        previous_count = successive_cut_counts[key]

    saveprint('', outfile=output_text_file)
    eventids_array = ds.organizeEventDict(pass_all_cuts_eventids_dict)
    saveprint('Final number of events remaining is %i'%len(eventids_array), outfile=output_text_file)






        

