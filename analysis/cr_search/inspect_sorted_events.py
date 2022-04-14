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
import glob

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

def maximizeAllFigures():
    '''
    Maximizes all matplotlib plots.
    '''
    for i in plt.get_fignums():
        plt.figure(i)
        fm = plt.get_current_fig_manager()
        fm.resize(*fm.window.maxsize())


template_dict = {}
filtered_template_dict = {} #will be changed throughout the script

# Load cranberry waveforms
cranberry_waveform_paths = glob.glob(os.path.join(os.environ['BEACON_ANALYSIS_DIR'] , 'analysis', 'cr_search', 'cranberry_analysis', '2-11-2022') + '/*.npy')
for path in cranberry_waveform_paths:
    event_name = os.path.split(path)[-1].replace('.npy','')
    template_dict['cranberry_%s'%event_name] = {}
    with open(path, 'rb') as f:
        template_dict['cranberry_%s'%event_name]['args'] = numpy.load(f, allow_pickle=True)
        channel0 = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['t']   = 2.0*numpy.arange(len(channel0))
        template_dict['cranberry_%s'%event_name]['wfs'] = numpy.zeros((8, len(channel0)))
        template_dict['cranberry_%s'%event_name]['wfs'][0] = channel0
        template_dict['cranberry_%s'%event_name]['wfs'][1] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][2] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][3] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][4] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][5] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][6] = numpy.load(f)
        template_dict['cranberry_%s'%event_name]['wfs'][7] = numpy.load(f)

# Load the strong cr candidate 5911 73399
for run, eventid in [[5911,73399]]:
    template_dict['%i-%i'%(run,eventid)] = {}
    reader = Reader(raw_datapath, run)
    reader.setEntry(eventid)
    template_dict['%i-%i'%(run,eventid)]['t']   = reader.t()
    template_dict['%i-%i'%(run,eventid)]['wfs'] = numpy.zeros((8, len(template_dict['%i-%i'%(run,eventid)]['t'])))
    template_dict['%i-%i'%(run,eventid)]['wfs'][0] = reader.wf(0) - numpy.mean(reader.wf(0))
    template_dict['%i-%i'%(run,eventid)]['wfs'][1] = reader.wf(1) - numpy.mean(reader.wf(1))
    template_dict['%i-%i'%(run,eventid)]['wfs'][2] = reader.wf(2) - numpy.mean(reader.wf(2))
    template_dict['%i-%i'%(run,eventid)]['wfs'][3] = reader.wf(3) - numpy.mean(reader.wf(3))
    template_dict['%i-%i'%(run,eventid)]['wfs'][4] = reader.wf(4) - numpy.mean(reader.wf(4))
    template_dict['%i-%i'%(run,eventid)]['wfs'][5] = reader.wf(5) - numpy.mean(reader.wf(5))
    template_dict['%i-%i'%(run,eventid)]['wfs'][6] = reader.wf(6) - numpy.mean(reader.wf(6))
    template_dict['%i-%i'%(run,eventid)]['wfs'][7] = reader.wf(7) - numpy.mean(reader.wf(7))

if __name__ == '__main__':
    plt.close('all')

    df = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated.xlsx'), sheet_name='good-with-airplane')
    # df = pd.read_excel(os.path.join(os.environ['BEACON_ANALYSIS_DIR'], 'analysis', 'cr_search', 'event_info_collated_1647029589.xlsx'), sheet_name='good-with-cc')
    fill_value = 'no assigned airplane'
    df['suspected_airplane_icao24'] = df['suspected_airplane_icao24'].fillna(value=fill_value)

    params = [['calibrated_trigtime', 'phi_best_choice'],['calibrated_trigtime', 'elevation_best_choice'],['cr_template_search_h', 'cr_template_search_v'], ['csnr_h', 'csnr_v'], ['phi_best_choice', 'elevation_best_choice'], ['impulsivity_h', 'impulsivity_v'], ['hpol_peak_to_sidelobe', 'vpol_peak_to_sidelobe'], ['template_cranberry_event000421_hpol', 'template_cranberry_event000421_vpol'],['template_cranberry_event001941_hpol', 'template_cranberry_event001941_vpol'],['template_cranberry_event000088_hpol', 'template_cranberry_event000088_vpol'],['template_5911-73399_hpol', 'template_5911-73399_vpol']]

    default_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']



    if True:
        #Plot things based on how they match to the different templates
        template_roots = numpy.array(['template_cranberry_event000421','template_cranberry_event001941','template_cranberry_event000088'])
        template_averages = {}
        template_averages['best_template'] = numpy.zeros(df.shape[0],dtype=template_roots.dtype)
        template_averages['best'] = numpy.zeros(df.shape[0])
        for template_root in template_roots:
            template_averages[template_root] = {}
            template_averages[template_root]['hpol'] = numpy.zeros(df.shape[0])
            template_averages[template_root]['vpol'] = numpy.zeros(df.shape[0])
            template_averages[template_root]['all'] = numpy.zeros(df.shape[0])

        for row_index, row in df.iterrows():
            sys.stdout.write('(%i/%i)\t\t\t\r'%(row_index+1,df.shape[0]))
            max_mean = 0
            best = ''
            for template_root in template_roots:
                for channel in range(8):
                    if channel%2 == 0:
                        template_averages[template_root]['hpol'][row_index] += row[template_root+'_ch%i'%channel]/4
                    else:
                        template_averages[template_root]['vpol'][row_index] += row[template_root+'_ch%i'%channel]/4
                    template_averages[template_root]['all'][row_index] += row[template_root+'_ch%i'%channel]/8
                if template_averages[template_root]['all'][row_index] > max_mean:
                    max_mean = template_averages[template_root]['all'][row_index]
                    best = template_root
            template_averages['best_template'][row_index] = best
            template_averages['best'][row_index] = max_mean

        df['best_template'] = template_averages['best_template']
        df['best'] = template_averages['best']


        cmap = plt.cm.get_cmap('cool')#plt.cm.get_cmap('RdYlGn')
        for param_x, param_y in params:
            fig = plt.figure()
            ax = plt.gca()
            fig.canvas.set_window_title('B: %s-%s'%(param_x,param_y))
            if 'template_' in param_x and 'pol' in param_x:
                plt.xlabel('Max CC with %s'%(param_x.replace('template','').replace('_',' ').title()))
                plt.ylabel('Max CC with %s'%(param_y.replace('template','').replace('_',' ').title()))
            else:
                plt.xlabel(param_x)
                plt.ylabel(param_y)                

            if param_x == 'calibrated_trigtime':
                _x = df[param_x]
                _xmin = _x.min()
                _x = (_x - _xmin)/(3600*24)
                xlim = (min(_x), max(_x)*1.1)
            else:
                xlim = (min(df[param_x]), max(df[param_x])*1.1)

            ylim = (min(df[param_y]), max(df[param_y])*1.1)

            x = df.query('key == "good" & suspected_airplane_icao24 == "%s"'%fill_value)[param_x]
            y = df.query('key == "good" & suspected_airplane_icao24 == "%s"'%fill_value)[param_y]

            if param_x == 'calibrated_trigtime':
                xmin = x.min()
                x = (x - xmin)/(3600*24)
                plt.xlabel(param_x + '\n(days since %i)'%xmin)
                sc = plt.scatter(x, y, c=df.query('key == "good" & suspected_airplane_icao24 == "%s"'%fill_value)['run']%8, vmin=-0.5, vmax=7.5, cmap=plt.cm.get_cmap('Dark2'))
                cbar = plt.colorbar(sc)
                cbar.set_label('Run%8')

                plt.legend()
            else:
                sc = plt.scatter(x, y, c=df.query('key == "good" & suspected_airplane_icao24 == "%s"'%fill_value)['best'], vmin=0, vmax=1.0, cmap=cmap)
                cbar = plt.colorbar(sc)
                cbar.set_label('Mean Template Correlations\n(Best of %i Templates)'%len(template_roots))

            if 'template' in param_x:
                plt.ylim(0,1.1)
                plt.xlim(0,1.1)
                plt.legend(loc='upper left')
            else:
                plt.xlim(min(xlim),max(xlim))
                plt.ylim(min(ylim),max(ylim))
                plt.legend()

            # for template_root in template_roots:
            #     cut = template_averages['best_template'] == template_root
            #     cut = cut[x.index] #Only the ones in the query
            #     plt.scatter(x[cut], y[cut],label='Correlates best with\n%s'%template_root)


            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


            #NOw with only airplanes 
            fig = plt.figure()
            ax = plt.gca()
            fig.canvas.set_window_title('B: %s-%s\nKnown Airplanes'%(param_x,param_y))
            if 'template_' in param_x and 'pol' in param_x:
                plt.xlabel('Max CC with %s'%(param_x.replace('template','').replace('_',' ').title()))
                plt.ylabel('Max CC with %s'%(param_y.replace('template','').replace('_',' ').title()))
            else:
                plt.xlabel(param_x)
                plt.ylabel(param_y)                

            x = df.query('key == "good" & suspected_airplane_icao24 != "%s"'%fill_value)[param_x]
            y = df.query('key == "good" & suspected_airplane_icao24 != "%s"'%fill_value)[param_y]

            if param_x == 'calibrated_trigtime':
                xmin = x.min()
                x = (x - xmin)/(3600*24)
                plt.xlabel(param_x + '\n(days since %i)'%xmin)
                sc = plt.scatter(x, y, c=df.query('key == "good" & suspected_airplane_icao24 != "%s"'%fill_value)['run']%8, vmin=-0.5, vmax=7.5, cmap=plt.cm.get_cmap('Dark2'))
                cbar = plt.colorbar(sc)
                cbar.set_label('Run%8')
            else:
                sc = plt.scatter(x, y, c=df.query('key == "good" & suspected_airplane_icao24 != "%s"'%fill_value)['best'], vmin=0, vmax=1.0, cmap=cmap)
                cbar = plt.colorbar(sc)
                cbar.set_label('Mean Template Correlations\n(Best of %i Templates)'%len(template_roots))


            # for template_root in template_roots:
            #     cut = template_averages['best_template'] == template_root
            #     cut = cut[x.index] #Only the ones in the query
            #     plt.scatter(x[cut], y[cut],label='Correlates best with\n%s'%template_root)

            if 'template' in param_x:
                plt.ylim(0,1.1)
                plt.xlim(0,1.1)
            else:
                plt.xlim(min(xlim),max(xlim))
                plt.ylim(min(ylim),max(ylim))

            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)

    if False:

        for param_x, param_y in params:
            cc = ['b', 'g', 'r', 'c', 'm', 'y']
            fig = plt.figure()
            ax = plt.gca()
            fig.canvas.set_window_title('A: %s-%s'%(param_x,param_y))
            if 'template_' in param_x and 'pol' in param_x:
                plt.xlabel('Max CC with %s'%(param_x.replace('template','').replace('_',' ').title()))
                plt.ylabel('Max CC with %s'%(param_y.replace('template','').replace('_',' ').title()))
            else:
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
                        label=None
                        # continue
                    else:
                        label = '%s - Unassigned'%group

                    x = df.query('key == "%s" & suspected_airplane_icao24 == "%s"'%(group, icao24))[param_x]
                    y = df.query('key == "%s" & suspected_airplane_icao24 == "%s"'%(group, icao24))[param_y]
                    if icao24 == 'no assigned airplane':
                        plt.scatter(x, y, c='k', edgecolor=edge_color, linewidths=2, label=label)
                    else:
                        plt.scatter(x, y, c=cc[icao24_index%len(cc)], edgecolor=edge_color, linewidths=2)#, label=group + '-' + icao24

            if 'template' in param_x:
                plt.ylim(0,1.1)
                plt.xlim(0,1.1)
                plt.legend(loc='upper left')
            else:
                plt.legend()
            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)


        unassigned = df.query('key == "good" & suspected_airplane_icao24 == "%s"'%fill_value)



    maximizeAllFigures()
        #Cross correlate these events against the the template given by Andrew, as well as by 5911-73399