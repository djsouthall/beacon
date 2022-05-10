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

def saveprint(*args, outfile=None):
    print(*args) #Print once to command line
    if outfile is not None:
        print(*args, file=open(outfile,'a')) #Print once to file


start_time = time.time()

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')

if False:
    processed_datapath = '/home/dsouthall/scratch-midway2/beacon/backup_mar31_2022'
elif True:
    processed_datapath = '/home/dsouthall/scratch-midway2/beacon/backup_feb28_2022'
else:
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



if __name__ == '__main__':
    plt.close('all')
    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_length = 16384
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_%i-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'%map_length
    if False:
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
    elif True:
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
        runs = numpy.array([])
        for k in (run_batches.keys()):
            runs = numpy.append(runs,run_batches[k])
    else:
        runs = numpy.arange(5733,5740)#numpy.arange(5974,6073)#numpy.arange(5733,5974)#numpy.arange(5910,5913)#numpy.arange(5733,5974)#numpy.arange(5910,5912)


    out_dir = './data/cuts_run%i-run%i_%i'%(min(runs), max(runs), start_time)
    os.mkdir(out_dir)
    os.mkdir(os.path.join(out_dir, 'all_cuts'))
    os.mkdir(os.path.join(out_dir, 'all_but_cuts'))
    os.mkdir(os.path.join(out_dir, 'no_cuts'))
    output_text_file = os.path.join(out_dir, 'output_histogram_saving_%i.txt'%start_time)
    saveprint('SETTING processed_datapath TO: ', processed_datapath, outfile=output_text_file)


    saveprint("Preparing dataSlicer", outfile=output_text_file)

    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.1 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    if 'backup_feb28_2022' in processed_datapath:
        low_ram_mode = False
        exclude_runs = numpy.array([    5981, 5993, 6033, 6090, 6500, 6500, 6501, 6502, 6503, 6504, 6505,
                                        6506, 6507, 6508, 6509, 6510, 6511, 6512, 6513, 6514, 6515, 6516,
                                        6517, 6518, 6519, 6520, 6521, 6522, 6523, 6524, 6525, 6526, 6527,
                                        6528, 6529, 6530, 6531, 6532, 6533, 6534, 6535, 6536, 6540, 6541,
                                        6542, 6543, 6544, 6545, 6546, 6547, 6548, 6549, 6550, 6551, 6552,
                                        6553, 6554, 6555, 6556, 6557, 6558, 6559, 6560, 6561, 6562, 6563,
                                        6564, 6565, 6566, 6567, 6568, 6569, 6570, 6571, 6572, 6573, 6574,
                                        6575, 6576, 6577, 6578, 6579, 6580, 6581, 6582, 6583, 6584, 6585,
                                        6586, 6587, 6588, 6589, 6590, 6591, 6592, 6593, 6594, 6595, 6596,
                                        6597, 6598, 6599, 6600, 6601, 6602, 6603, 6604, 6605, 6606, 6607,
                                        6608, 6609, 6610, 6611, 6612, 6613, 6614, 6615, 6616, 6617, 6618,
                                        6619, 6620, 6621, 6622, 6623, 6624, 6625, 6626, 6627, 6628, 6629,
                                        6630, 6631, 6632, 6633, 6634, 6635, 6636, 6637, 6638, 6639, 6640])

        runs = runs[~numpy.isin(runs, exclude_runs)]

    else:
        low_ram_mode = True


    ds = dataSlicer(runs, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, \
                    low_ram_mode=low_ram_mode,\
                    analysis_data_dir=processed_datapath, curve_choice=0, trigger_types=[2],included_antennas=[0,1,2,3,4,5,6,7],\
                    cr_template_n_bins_h=200,cr_template_n_bins_v=200,impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                    std_n_bins_h=200,std_n_bins_v=200,max_std_val=12,p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                    snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35,include_test_roi=False,\
                    n_phi=n_phi, range_phi_deg=(min_phi,max_phi), n_theta=n_theta, range_theta_deg=(min_theta,max_theta), remove_incomplete_runs=True)

    cut_dict_key = 'all_cuts'
    if False and 'backup_feb28_2022' in processed_datapath:
        ds.addROI(cut_dict_key, { 'p2p_gap_h':[-1e10, 95]})
    else:
        ds.addROI(cut_dict_key, {   'elevation_best_choice':[10,90],
                                    'phi_best_choice':[-90,90],
                                    'similarity_count_h':[-0.1,10],
                                    'similarity_count_v':[-0.1,10],
                                    'hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe':[1.2,1e10],
                                    'impulsivity_hSLICERADDimpulsivity_v':[0.3,1e10],
                                    'cr_template_search_hSLICERMAXcr_template_search_v':[0.4,1e10],
                                    'in_targeted_box':[-0.5, 0.5],
                                    'p2p_gap_h':[-1e10, 95],
                                    'above_normalized_map_max_line':[0,1e10],
                                    'above_snr_line':[0,1e10]})

        

    for param_key in list(ds.roi[cut_dict_key].keys()):
        copied_dict = copy.deepcopy(ds.roi[cut_dict_key])
        del copied_dict[param_key]
        ds.addROI(cut_dict_key + '-' + param_key, copy.deepcopy(copied_dict)) #Add an ROI containing all cuts BUT the current key (used as ROI label).

    saveprint("Getting initial cut statistics\n", outfile=output_text_file)
    pass_all_cuts_eventids_dict, successive_cut_counts, total_cut_counts = ds.getCutsFromROI(cut_dict_key,load=False,save=False,verbose=True, return_successive_cut_counts=True, return_total_cut_counts=True) #Events passing all cuts

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

    outfile_name = os.path.join(out_dir, 'pass_all_cuts_eventids_dict.npy')
    if os.path.exists(outfile_name):
        outfile_name.replace('.npy','_%i.npy'%time.time())

    saveprint('Saving pass_all_cuts_eventids_dict as %s'%outfile_name, outfile=output_text_file)
    numpy.save(outfile_name, pass_all_cuts_eventids_dict, allow_pickle=True)
    
    '''
    I want to add in saving the same histograms with NO cuts applied and ALL cuts applied, as well as sequentially in
    the order they are made.  Can I store the information for 2d histograms this way as well?  I much prefer them.
    '''

    saveprint("\nSaving Histograms\n", outfile=output_text_file)
    for param_key in list(ds.roi[cut_dict_key].keys()):
        saveprint(param_key, outfile=output_text_file)
        # Save histograms with no cuts
        # if param_key in ['elevation_best_choice', 'phi_best_choice']:
        #     continue

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

        numpy.savez_compressed(os.path.join(out_dir, 'all_but_cuts', 'hist_for_%s_with_all_cuts_but_%s.npz'%(param_key, param_key)), **sub_eventids_dict)
        saveprint(os.path.join(out_dir, 'all_but_cuts', 'hist_for_%s_with_all_cuts_but_%s.npz'%(param_key, param_key)), outfile=output_text_file)

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

        numpy.savez_compressed(os.path.join(out_dir, 'all_cuts', 'hist_for_%s_with_all_cuts.npz'%(param_key)), **sub_eventids_dict)
        saveprint(os.path.join(out_dir, 'all_cuts', 'hist_for_%s_with_all_cuts.npz'%(param_key)), outfile=output_text_file)

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

        numpy.savez_compressed(os.path.join(out_dir, 'no_cuts', 'hist_for_%s_with_no_cuts.npz'%(param_key)), **sub_eventids_dict)
        saveprint(os.path.join(out_dir, 'no_cuts', 'hist_for_%s_with_no_cuts.npz'%(param_key)), outfile=output_text_file)

        for key in numpy.array(list(sub_eventids_dict.keys())):
            del sub_eventids_dict[key] 
        del sub_eventids_dict
        del counts
        del bin_centers
        del bin_width
        gc.collect()







        

'''
Remaining Events After Step similarity_count_h is 1116064
1.081% events then cut by similarity_count_v with bounds [-0.1, 10]
85.258% of initial events would be cut by similarity_count_v with bounds [-0.1, 10]

Remaining Events After Step similarity_count_v is 1104002
81.710% events then cut by hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe with bounds [1.2, 10000000000.0]
6.496% of initial events would be cut by hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe with bounds [1.2, 10000000000.0]

Remaining Events After Step hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe is 201926
71.441% events then cut by impulsivity_hSLICERADDimpulsivity_v with bounds [0.3, 10000000000.0]
2.888% of initial events would be cut by impulsivity_hSLICERADDimpulsivity_v with bounds [0.3, 10000000000.0]

Remaining Events After Step impulsivity_hSLICERADDimpulsivity_v is 57669
26.852% events then cut by cr_template_search_hSLICERMAXcr_template_search_v with bounds [0.4, 10000000000.0]
2.755% of initial events would be cut by cr_template_search_hSLICERMAXcr_template_search_v with bounds [0.4, 10000000000.0]

Remaining Events After Step cr_template_search_hSLICERMAXcr_template_search_v is 42184
55.433% events then cut by p2p_gap_h with bounds [-10000000000.0, 95]
0.382% of initial events would be cut by p2p_gap_h with bounds [-10000000000.0, 95]

Remaining Events After Step p2p_gap_h is 18800
45.138% events then cut by above_normalized_map_max_line with bounds [0, 10000000000.0]
23.144% of initial events would be cut by above_normalized_map_max_line with bounds [0, 10000000000.0]

Remaining Events After Step above_normalized_map_max_line is 10314
28.679% events then cut by above_snr_line with bounds [0, 10000000000.0]
4.395% of initial events would be cut by above_snr_line with bounds [0, 10000000000.0]

Remaining Events After Step above_snr_line is 7356
0.272% events then cut by in_targeted_box with bounds [-0.5, 0.5]
0.017% of initial events would be cut by in_targeted_box with bounds [-0.5, 0.5]

Remaining Events After Step in_targeted_box is 7336

Final number of events remaining is 7336
        Getting counts from 1dhists for elevation_best_choice
        Getting counts from 1dhists for elevation_best_choice
        Getting counts from 1dhists for elevation_best_choice
        Getting counts from 1dhists for phi_best_choice
        Getting counts from 1dhists for phi_best_choice
        Getting counts from 1dhists for phi_best_choice
        Getting counts from 1dhists for similarity_count_h
        Getting counts from 1dhists for similarity_count_h
        Getting counts from 1dhists for similarity_count_h
        Getting counts from 1dhists for similarity_count_v
        Getting counts from 1dhists for similarity_count_v
        Getting counts from 1dhists for similarity_count_v
        Getting counts from 1dhists for hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe
        Getting counts from 1dhists for hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe
        Getting counts from 1dhists for hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe
        Getting counts from 1dhists for impulsivity_hSLICERADDimpulsivity_v
        Getting counts from 1dhists for impulsivity_hSLICERADDimpulsivity_v
        Getting counts from 1dhists for impulsivity_hSLICERADDimpulsivity_v
        Getting counts from 1dhists for cr_template_search_hSLICERMAXcr_template_search_v
        Getting counts from 1dhists for cr_template_search_hSLICERMAXcr_template_search_v
        Getting counts from 1dhists for cr_template_search_hSLICERMAXcr_template_search_v
        Getting counts from 1dhists for p2p_gap_h
        Getting counts from 1dhists for p2p_gap_h
        Getting counts from 1dhists for p2p_gap_h
        Getting counts from 1dhists for above_normalized_map_max_line
        Getting counts from 1dhists for above_normalized_map_max_line
        Getting counts from 1dhists for above_normalized_map_max_line
        Getting counts from 1dhists for above_snr_line
        Getting counts from 1dhists for above_snr_line
        Getting counts from 1dhists for above_snr_line
        Getting counts from 1dhists for in_targeted_box
        Getting counts from 1dhists for in_targeted_box
        Getting counts from 1dhists for in_targeted_box
'''