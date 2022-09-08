#!/usr/bin/env python3
'''
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
import textwrap
import pandas

import numpy
import scipy
import scipy.signal

from beacon.tools.data_slicer import dataSlicer
import  beacon.tools.get_plane_tracks as pt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import time
from datetime import datetime
import pytz

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)

def maximizeAllFigures():
    '''
    Maximizes all matplotlib plots.
    '''
    for i in plt.get_fignums():
        plt.figure(i)
        fm = plt.get_current_fig_manager()
        fm.resize(*fm.window.maxsize())

'''
'impulsivity_h','impulsivity_v', 'cr_template_search_h', 'cr_template_search_v', 'std_h', 'std_v', 'p2p_h', 'p2p_v', 'snr_h', 'snr_v',\
'time_delay_0subtract1_h','time_delay_0subtract2_h','time_delay_0subtract3_h','time_delay_1subtract2_h','time_delay_1subtract3_h','time_delay_2subtract3_h',\
'time_delay_0subtract1_v','time_delay_0subtract2_v','time_delay_0subtract3_v','time_delay_1subtract2_v','time_delay_1subtract3_v','time_delay_2subtract3_v',
'cw_present','cw_freq_Mhz','cw_linear_magnitude','cw_dbish','theta_best_h','theta_best_v','elevation_best_h','elevation_best_v','phi_best_h','phi_best_v',\
'calibrated_trigtime','triggered_beams','beam_power','hpol_peak_to_sidelobe','vpol_peak_to_sidelobe','hpol_max_possible_map_value','vpol_max_possible_map_value',\
'map_max_time_delay_0subtract1_h','map_max_time_delay_0subtract2_h','map_max_time_delay_0subtract3_h',\
'map_max_time_delay_1subtract2_h','map_max_time_delay_1subtract3_h','map_max_time_delay_2subtract3_h',\
'map_max_time_delay_0subtract1_v','map_max_time_delay_0subtract2_v','map_max_time_delay_0subtract3_v',\
'map_max_time_delay_1subtract2_v','map_max_time_delay_1subtract3_v','map_max_time_delay_2subtract3_v'
'''

#Include special conditions for certain events
special_conditions = {}
# Default Values
include_baselines = numpy.array([0,1,2,3,4,5])
append_notches = None

ignore_airplanes = []#['a73278']

if __name__ == '__main__':
    save = True


    if save == True:
        plt.ioff()
    else:
        plt.ion()

    run = 5911
    eventid = 73399


    cmap = 'cool'#'coolwarm'
    impulsivity_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    time_delays_dset_key = 'LPf_80.0-LPo_14-HPf_20.0-HPo_4-Phase_1-Hilb_0-corlen_131072-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
    map_direction_dset_key = 'LPf_85.0-LPo_6-HPf_25.0-HPo_8-Phase_1-Hilb_0-upsample_16384-maxmethod_0-sinesubtract_1-deploy_calibration_september_2021_minimized_calibration.json-n_phi_3600-min_phi_neg180-max_phi_180-n_theta_480-min_theta_0-max_theta_120-scope_allsky'

    ds = dataSlicer([run], impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key, analysis_data_dir=processed_datapath)
    ds.conference_mode = True

    ds.eventInspector({run:[eventid]}, show_all=False, include_time_delays=not ds.conference_mode,append_notches=append_notches,include_baselines=include_baselines, div128=True)
    ds.inspector_mpl['fig1'].set_size_inches(20,12)
    [ds.inspector_mpl[p].set_ylabel(apply_filter*'Filtered ' + 'PSD\n' + '(dB, arb)', fontsize=18) for p, apply_filter in [['fig1_spec_raw',False] ,['fig1_spec_filt', True]]]

    [ds.inspector_mpl[p].yaxis.set_label_coords(-0.05,0.5) for p in ['fig1_spec_raw' ,'fig1_spec_filt']]

    ds.inspector_mpl['fig1_wf_h'].text(0.985, 0.03, 'HPol', transform=ds.inspector_mpl['fig1_wf_h'].transAxes, fontsize=18, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="square", fc="w"))
    ds.inspector_mpl['fig1_wf_v'].text(0.985, 0.03, 'VPol', transform=ds.inspector_mpl['fig1_wf_v'].transAxes, fontsize=18, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle="square", fc="w"))

    plt.tight_layout()

    plt.subplots_adjust(left=0.05, bottom=0.1,right=0.95, top=0.95, wspace=0.33, hspace=0.4)

    if save == True:
        ds.inspector_mpl['fig1'].savefig('./figures/r%i_e%i_event_display.pdf'%(run,eventid), dpi=300)
    # plt.subplots_adjust(left=0.06, bottom=0.12,right=0.98, top=0.95, wspace=0.3, hspace=0.2)
