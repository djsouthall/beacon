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
import pandas as pd
from cycler import cycler


#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import FFTPrepper, TemplateCompareTool
# from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
# from beacon.tools.line_of_sight import circleSource
# from beacon.tools.flipbook_reader import flipbookToDict

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

impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

# cr_template_search_h
# cr_template_search_v
# cr_template_search_hSLICERMAXcr_template_search_v
# hpol_peak_to_sidelobe
# vpol_peak_to_sidelobe
# hpol_peak_to_sidelobeSLICERMAXvpol_peak_to_sidelobe
# hpol_normalized_map_value
# vpol_normalized_map_value
# above_normalized_map_max_line
# above_snr_line
# impulsivity_h
# impulsivity_v
# impulsivity_hSLICERADDimpulsivity_v
# similarity_count_h
# similarity_count_v
# p2p_gap_h
# p2p_gap_v
# csnr_h
# csnr_v
# snr_h
# snr_v
# p2p_h
# p2p_v
# std_h
# std_v

if __name__ == '__main__':
    plt.close('all')

    df = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'sept2021-week1-analysis', 'event_info_collated.xlsx'), sheet_name='good-with-airplane')
    fill_value = 'no assigned airplane'
    df['suspected_airplane_icao24'] = df['suspected_airplane_icao24'].fillna(value=fill_value)

    params = [['cr_template_search_h', 'cr_template_search_v'], ['csnr_h', 'csnr_v'], ['phi_best_choice', 'elevation_best_choice'], ['impulsivity_h', 'impulsivity_v'], ['hpol_peak_to_sidelobe', 'vpol_peak_to_sidelobe']]

    default_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']


    for param_x, param_y in params:
        cc = ['b', 'g', 'r', 'c', 'm', 'y']
        fig = plt.figure()
        fig.canvas.set_window_title('%s-%s'%(param_x,param_y))
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        for group_index, group in enumerate(numpy.unique(df['key'])):
            if group == 'bad':
                edge_color = 'tab:red'
                continue
            elif group == 'good':
                edge_color = 'tab:green'
            elif group == 'ambiguous':
                edge_color = 'tab:orange'


            for icao24_index, icao24 in enumerate(numpy.unique(df['suspected_airplane_icao24'])):
                if icao24 != 'no assigned airplane':
                    continue

                x = df.query('key == "%s" & suspected_airplane_icao24 == "%s"'%(group, icao24))[param_x]
                y = df.query('key == "%s" & suspected_airplane_icao24 == "%s"'%(group, icao24))[param_y]
                if icao24 == 'no assigned airplane':
                    plt.scatter(x, y, c='k', edgecolor=edge_color, linewidths=2, label='%s - Unassigned'%group)
                else:
                    plt.scatter(x, y, c=cc[icao24_index%len(cc)], edgecolor=edge_color, linewidths=2)#, label=group + '-' + icao24

        plt.legend()


    unassigned = df.query('key == "good" & suspected_airplane_icao24 == "%s"'%fill_value)

    #Cross correlate these events against the the template given by Andrew, as well as by 5911-73399