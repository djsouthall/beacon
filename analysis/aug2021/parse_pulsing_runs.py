#!/usr/bin/env python3
'''
This script will go through the known list of pulsing runs to help me determine which events correspond to which pulser
settings.  This will be used to inform info.py which events are expected from each direction and at what attenuation
the signals were emitted (of interest for SNR sweeps).
'''
import os
import sys
import numpy

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_slicer import dataSlicerSingleRun,dataSlicer
from beacon.tools.correlator import Correlator

import csv
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
from pprint import pprint
import itertools
import warnings
import h5py
import inspect
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector
import pandas as pd
numpy.set_printoptions(threshold=sys.maxsize)
def txtToClipboard(txt):
    '''
    Seems to bug out on midway.
    '''
    df=pd.DataFrame([txt])
    df.to_clipboard(index=False,header=False)
plt.ion()

# import beacon.tools.info as info
from beacon.tools.data_handler import createFile, getTimes, loadTriggerTypes, getEventTimes
# from beacon.tools.fftmath import FFTPrepper
# from beacon.analysis.background_identify_60hz import plotSubVSec, plotSubVSecHist, alg1, diffFromPeriodic, get60HzEvents2, get60HzEvents3, get60HzEvents
from beacon.tools.fftmath import TemplateCompareTool

from beacon.analysis.find_pulsing_signals_june2021_deployment import Selector


# pulsing_sites = {
#                     'd2sa':{
#                             'runs':{
#                                     5626
#                             }
#                     }
# }
# Day 2 Site A:
# HPol: 5626-5631
# VPol: 5632
# Day 3 Site A:
# HPol: 5639-5641
# VPol: 5642
# Day 3 Site B:
# HPol: 5644
# VPol: 5646
# Day 3 Site C:
# H/VPol: 5647-5649
# Day 5 Site A:
# HPol: 5656
# VPol: 5657
# Day 5 Site B:
# HPol: 5659
# VPol: 5660


# 5625
# 5626
# 5627
# 5628-5629
# 5630
# 5631
# 5632
# 5633
# 5639
# 5640
# 5641
# 5642
# 5643
# 5644
# 5645
# 5646
# 5647
# 5648
# 5649
# 5650-5654
# 5655
# 5656
# 5657
# 5658
# 5659
# 5660

def moving_average(x, w):
    return numpy.convolve(x, numpy.ones(w), 'valid') / w

class Selector(object):
    def __init__(self,ax,scatter_data, event_info):
        self.ax = ax
        self.xys = scatter_data.get_offsets()
        self.event_info = event_info
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.readers = {}
        self.cors = {}
        for run in numpy.unique(self.event_info['run']):
            self.readers[run] = Reader(os.environ['BEACON_DATA'],run)
            self.cors[run] = Correlator(self.readers[run],upsample=len(self.readers[run].t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1)

    def onselect(self,verts):
        #print(verts)
        path = Path(verts)
        ind = numpy.nonzero(path.contains_points(self.xys))[0]

        if type(self.event_info) == numpy.ma.core.MaskedArray:
            print('Only printing non-masked events:')
            eventids = self.event_info[ind][~self.event_info.mask[[ind]]]
            pprint(numpy.asarray(eventids[['run','eventid']]))
        else:
            eventids = self.event_info[ind]
            pprint(eventids[['run','eventid']])
        if len(eventids) == 1:
            eventid = eventids[0]['eventid']
            run = eventids[0]['run']
            fig = plt.figure()
            plt.title('Run %i, Eventid %i'%(run, eventid))
            self.readers[run].setEntry(eventid)
            ax = plt.subplot(2,1,1)
            plt.suptitle('HPol P2P = %i\nVPol P2P = %i'%(eventids['p2p_h'][0], eventids['p2p_v'][0]))
            for channel in [0,2,4,6]:
                plt.plot(self.readers[run].t(),self.readers[run].wf(channel))
            plt.ylabel('HPol Adu')
            plt.xlabel('t (ns)')
            plt.subplot(2,1,2,sharex=ax)
            plt.ylabel('VPol Adu')
            plt.xlabel('t (ns)')
            for channel in [1,3,5,7]:
                plt.plot(self.readers[run].t(),self.readers[run].wf(channel))
            self.cors[run].map(eventid,'both',interactive=True)

if __name__ == '__main__':
    plt.close('all')
    try:


        #Sort through trigtype == 3 events and plot p2p as a function of time in a large scatter plot.
        # Just a simple list of runs with pulses.  As yet unparsed.  

        day = 3
        if day == 2:
            pulsing_run_list = [5630, 5631,5632] #day 2

            #This first one is "raw" estimates
            #The second is the refined time windowing

            # times_of_interest = {
            #                     'HPol 20 dB starts' : 1629327563,\
            #                     'HPol 10 dB starts' : 1629328334,\
            #                     'HPol New Run, 13 dB': 1629329339,\
            #                     'HPol Start 30 dB': 1629330230,\
            #                     'VPol Start at 13 dB': 1629331740,\
            #                     'VPol Start 10 dB':1629332414,\
            #                     'VPol Start 20 dB':1629333157,\
            #                     'Stop Pulsing': 1629333653
            #                     }
            #30dB are estimates, basically no signal visible
            times_of_interest = {
                                'HPol 20 dB starts' :   1629327563.72,\
                                'HPol 20 dB stops' :    1629328274.30,\
                                'HPol 10 dB starts' :   1629328327.38,\
                                'HPol 10 dB stops' :    1629329217.156,\
                                'HPol 13 dB starts':    1629329339,\
                                'HPol 13 dB stops':     1629330143.1,\
                                'HPol 30 dB starts':    1629330230,\
                                'HPol 30 dB stops':     1629331227,\
                                'VPol 13 dB starts':    1629331623.13,\
                                'VPol 13 dB stops':     1629332321.5,\
                                'VPol 10 dB starts':    1629332399.6,\
                                'VPol 10 dB stops':     1629333067.11,\
                                'VPol 20 dB starts':    1629333124.4,\
                                'VPol 20 dB stops':     1629333653
                                }
        elif day == 3:
            pulsing_run_list = [5639,5640,5641,5642,5644,5646,5647,5648,5649] #day 3
            times_of_interest = {}
        else:
            pulsing_run_list = [5656,5657,5659,5660] #day 4
            times_of_interest = {}

        all_values_p2p = numpy.array([])
        all_values_p2p = numpy.array([])

        info_dtype = numpy.dtype([  
                                    ('run','i'),
                                    ('eventid','i'),
                                    ('p2p_h','i'),
                                    ('p2p_v','i'),
                                    ('calibrated_trig_time','float64')
                                ])

        info = numpy.array([],dtype=info_dtype)

        for run_index, run in enumerate(pulsing_run_list):
            sys.stdout.write('(%i/%i)\r'%(run_index + 1,len(pulsing_run_list)))
            sys.stdout.flush()
            reader = Reader(os.environ['BEACON_DATA'],run)
            #tct = TemplateCompareTool(reader,apply_phase_response=True)

            trigger_type = loadTriggerTypes(reader)
            eventids = numpy.arange(len(trigger_type))
            trigtype_cut = trigger_type == 3

            _info = numpy.ones(sum(trigtype_cut),dtype=info_dtype)
            _info['run'] = run
            _info['eventid'] = eventids[trigtype_cut]
            _info['calibrated_trig_time'] = getEventTimes(reader)[trigtype_cut]

            for event_index, eventid in enumerate(_info['eventid']):
                reader.setEntry(eventid)
                
                max_p2p_hpol = 0
                max_p2p_vpol = 0
                
                for ch in range(8):
                    wf = reader.wf(ch)
                    if ch%2 == 0:
                        max_p2p_hpol = max( max_p2p_hpol , max(wf) - min(wf) )                    
                    else:
                        max_p2p_vpol = max( max_p2p_vpol , max(wf) - min(wf) )
                
                _info['p2p_h'][event_index] = max_p2p_hpol
                _info['p2p_v'][event_index] = max_p2p_vpol
            info = numpy.append(info,_info)

        print('\nInfo preparation complete')

        lassos = []
        
        
        plt.figure()
        ax = plt.gca()
        w = 20 #Width of rolling average
        plt.title('Day %i\nPeak to Peak with %i Event Moving Window'%(day, w))
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_h'],w)
        plt.plot(x,y,alpha=0.7,c='#94D0CC',linestyle = '-')
        
        scatter = plt.scatter(x,y,alpha=0.7,c='#94D0CC',label='Hpol P2P',)        
        _s = Selector(ax,scatter,info)
        lassos.append(_s)
        
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_v'],w)
        plt.plot(x,y,alpha=0.7,c='#F29191',linestyle = '-.')
        
        scatter = plt.scatter(x,y,alpha=0.7,c='#F29191',label='Vpol P2P',)
        _s = Selector(ax,scatter,info)
        lassos.append(_s)

        plt.ylabel('p2p')
        plt.xlabel('Calibrated Trigger Time')

        for key, time in times_of_interest.items():
            plt.axvline(time, label=key, color = next(ax._get_lines.prop_cycler)['color'])
        plt.legend()


        plt.figure()
        ax = plt.gca()
        w = 20 #Width of rolling average
        plt.title('Day %i\nLine Represents Peak to Peak with %i Event Moving Window'%(day, w))
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_h'],w)
        plt.plot(x,y,alpha=0.7,c='#94D0CC',linestyle = '-')

        x = info['calibrated_trig_time']
        y = info['p2p_h']
        scatter = plt.scatter(x,y,alpha=0.7,c='#94D0CC',label='Hpol P2P',)        
        _s = Selector(ax,scatter,info)
        lassos.append(_s)
        
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_v'],w)
        plt.plot(x,y,alpha=0.7,c='#F29191',linestyle = '-.')

        x = info['calibrated_trig_time']
        y = info['p2p_v']
        scatter = plt.scatter(x,y,alpha=0.7,c='#F29191',label='Vpol P2P',)
        _s = Selector(ax,scatter,info)
        lassos.append(_s)

        plt.ylabel('p2p')
        plt.xlabel('Calibrated Trigger Time')

        for key, time in times_of_interest.items():
            plt.axvline(time, label=key, color = next(ax._get_lines.prop_cycler)['color'])
        plt.legend()



    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

