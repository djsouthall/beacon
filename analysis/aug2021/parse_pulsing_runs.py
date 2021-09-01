#!/usr/bin/env python3
'''
This script will go through the known list of pulsing runs to help me determine which events correspond to which pulser
settings.  This will be used to inform info.py which events are expected from each direction and at what attenuation
the signals were emitted (of interest for SNR sweeps).
'''
import os
import sys
import numpy
import pymap3d as pm

from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
import beacon.tools.interpret #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_slicer import dataSlicerSingleRun,dataSlicer
from beacon.tools.correlator import Correlator
from beacon.tools.fftmath import TimeDelayCalculator
import beacon.tools.info as info

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
    def __init__(self,ax,scatter_data, event_info, deploy_index):
        self.ax = ax
        self.xys = scatter_data.get_offsets()
        self.event_info = event_info
        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.readers = {}
        self.cors = {}
        for run in numpy.unique(self.event_info['run']):
            self.readers[run] = Reader(os.environ['BEACON_DATA'],run)
            self.cors[run] = Correlator(self.readers[run],upsample=len(self.readers[run].t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1,deploy_index=deploy_index)

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

def predictAlignment(phi_deg, theta_deg, cor, pol='hpol'):
    '''
    Given an expected azimuth, zenith, and correlator, this will use the correlators precalculated grid of expected
    time delays to determine how to shift the waveforms from that source direction.
    '''
    try:
        theta_index = numpy.argmin( abs(cor.thetas_deg - theta_deg) ) #Data plotted with elevation angles, not zenith.
        phi_index = numpy.argmin( abs(cor.phis_deg - phi_deg) ) 

        if pol == 'hpol':
            channels = numpy.array([0,2,4,6])

            t_best_0subtract1 = cor.t_hpol_0subtract1[theta_index,phi_index]
            t_best_0subtract2 = cor.t_hpol_0subtract2[theta_index,phi_index]
            t_best_0subtract3 = cor.t_hpol_0subtract3[theta_index,phi_index]
            t_best_1subtract2 = cor.t_hpol_1subtract2[theta_index,phi_index]
            t_best_1subtract3 = cor.t_hpol_1subtract3[theta_index,phi_index]
            t_best_2subtract3 = cor.t_hpol_2subtract3[theta_index,phi_index]
        elif pol == 'vpol':
            channels = numpy.array([1,3,5,7])
            t_best_0subtract1 = cor.t_vpol_0subtract1[theta_index,phi_index]
            t_best_0subtract2 = cor.t_vpol_0subtract2[theta_index,phi_index]
            t_best_0subtract3 = cor.t_vpol_0subtract3[theta_index,phi_index]
            t_best_1subtract2 = cor.t_vpol_1subtract2[theta_index,phi_index]
            t_best_1subtract3 = cor.t_vpol_1subtract3[theta_index,phi_index]
            t_best_2subtract3 = cor.t_vpol_2subtract3[theta_index,phi_index]

        return numpy.array([t_best_0subtract1,t_best_0subtract2,t_best_0subtract3,t_best_1subtract2,t_best_1subtract3,t_best_2subtract3])
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)




class PulserInfo():
    '''
    Contains info and fucntions to get eventids and site info of potential pulser events.  These are not yet
    certain to be pulsing events, but pass the temporal cuts. 
    '''
    def __init__(self):
        try:
            self.event_dtype = numpy.dtype([('run','i'),('eventid','i'),('calibrated_trig_time','float64'),('attenuation_dB','f')])
            self.event_dtype_reduced = numpy.dtype([('run','i'),('eventid','i'),('attenuation_dB','f')])

            self.timing_dict = {}

            # Day 2 

            self.timing_dict['d2sa'] = {}
            self.timing_dict['d2sa']['runs'] = [5630, 5631,5632] #day 2
            self.timing_dict['d2sa']['latlonel'] = (37.5859361, -118.233918056, 3762.9)

            self.timing_dict['d2sa']['reference_event'] = {}
            self.timing_dict['d2sa']['reference_event']['hpol'] = (5630 , 1585)
            self.timing_dict['d2sa']['reference_event']['vpol'] = (5632 , 2105)
            
            self.timing_dict['d2sa']['times_of_interest'] = {}
            self.timing_dict['d2sa']['times_of_interest']['hpol'] = {}
            self.timing_dict['d2sa']['times_of_interest']['vpol'] = {}

            self.timing_dict['d2sa']['times_of_interest']['hpol']['20dB'] = (1629327563.72, 1629328274.30)
            self.timing_dict['d2sa']['times_of_interest']['hpol']['10dB'] = (1629328327.38, 1629329217.156)
            self.timing_dict['d2sa']['times_of_interest']['hpol']['13dB'] = (1629329339, 1629330143.1)
            self.timing_dict['d2sa']['times_of_interest']['hpol']['30dB'] = (1629330230, 1629331227)

            self.timing_dict['d2sa']['times_of_interest']['vpol']['13dB'] = (1629331623.13, 1629332321.5)
            self.timing_dict['d2sa']['times_of_interest']['vpol']['10dB'] = (1629332399.6, 1629333067.11)
            self.timing_dict['d2sa']['times_of_interest']['vpol']['20dB'] = (1629333124.4, 1629333653)


            # Day 3
            # Site a
            self.timing_dict['d3sa'] = {}
            self.timing_dict['d3sa']['runs'] = [5638,5639,5640,5641,5642] #day 3
            self.timing_dict['d3sa']['latlonel'] = ( 37.58575767,-118.22592267, 3697.4)

            self.timing_dict['d3sa']['reference_event'] = {}
            self.timing_dict['d3sa']['reference_event']['hpol'] = (5641 , 1527)
            self.timing_dict['d3sa']['reference_event']['vpol'] = (5642 , 2918)

            self.timing_dict['d3sa']['times_of_interest'] = {}
            self.timing_dict['d3sa']['times_of_interest']['hpol'] = {}
            self.timing_dict['d3sa']['times_of_interest']['vpol'] = {}

            self.timing_dict['d3sa']['times_of_interest']['hpol']['0dB'] = (1629393054.5 , 1629394031.53)
            self.timing_dict['d3sa']['times_of_interest']['hpol']['3dB'] = (1629394089.98 , 1629395178.25)
            self.timing_dict['d3sa']['times_of_interest']['hpol']['6dB'] = (1629395221.716 , 1629396041.16)
            self.timing_dict['d3sa']['times_of_interest']['hpol']['10dB'] = (1629396104.76 , 1629396686.18)
            self.timing_dict['d3sa']['times_of_interest']['hpol']['20dB'] = (1629396732.53 , 1629397478.1)
            self.timing_dict['d3sa']['times_of_interest']['hpol']['13dB'] = (1629397606.5 , 1629398545.31)

            self.timing_dict['d3sa']['times_of_interest']['vpol']['13dB'] = (1629398876.5 , 1629399598.5)
            self.timing_dict['d3sa']['times_of_interest']['vpol']['10dB'] = (1629399685.9 , 1629400519.2)
            self.timing_dict['d3sa']['times_of_interest']['vpol']['6dB'] = (1629400584.7 , 1629401182.2)
            self.timing_dict['d3sa']['times_of_interest']['vpol']['20dB'] = (1629401233.8 , 1629402310.5)

            # Site b
            self.timing_dict['d3sb'] = {}
            self.timing_dict['d3sb']['runs'] = [5643,5644,5645,5646,5647] #day 3
            self.timing_dict['d3sb']['latlonel'] = (37.58779650,-118.22452000,3619.0)

            #BAD REFERENCE EVENT NOT YET SET
            self.timing_dict['d3sb']['reference_event'] = {}
            self.timing_dict['d3sb']['reference_event']['hpol'] = (5644 , 6278)
            self.timing_dict['d3sb']['reference_event']['vpol'] = (5646 , 4500)

            self.timing_dict['d3sb']['times_of_interest'] = {}
            self.timing_dict['d3sb']['times_of_interest']['hpol'] = {}
            self.timing_dict['d3sb']['times_of_interest']['vpol'] = {}

            self.timing_dict['d3sb']['times_of_interest']['hpol']['0dB'] = (1629404874.67 , 1629405556.23)
            self.timing_dict['d3sb']['times_of_interest']['hpol']['3dB'] = (1629405587.5 , 1629406305.5)
            self.timing_dict['d3sb']['times_of_interest']['hpol']['6dB'] = (1629406318.5 , 1629406943.5)
            self.timing_dict['d3sb']['times_of_interest']['hpol']['10dB'] = (1629407005.5 , 1629407592.5)
            self.timing_dict['d3sb']['times_of_interest']['hpol']['13dB'] = (1629407645.0 , 1629408245.5)
            self.timing_dict['d3sb']['times_of_interest']['hpol']['20dB'] = (1629408283.5 , 1629408866.5)

            self.timing_dict['d3sb']['times_of_interest']['vpol']['0dB'] = (1629409096.5 , 1629409738.5)
            self.timing_dict['d3sb']['times_of_interest']['vpol']['3dB'] = (1629409778.5 , 1629410419.5)
            self.timing_dict['d3sb']['times_of_interest']['vpol']['6dB'] = (1629410439.5 , 1629411250.5)
            self.timing_dict['d3sb']['times_of_interest']['vpol']['10dB'] = (1629411273.5 , 1629411870.5)
            self.timing_dict['d3sb']['times_of_interest']['vpol']['13dB'] = (1629411890.5 , 1629412556.5)
            self.timing_dict['d3sb']['times_of_interest']['vpol']['20dB'] = (1629412580.5 , 1629413165.5)

            # Site c
            self.timing_dict['d3sc'] = {}
            self.timing_dict['d3sc']['runs'] = [5648,5649] #day 3
            self.timing_dict['d3sc']['latlonel'] = (37.58885717,-118.22786317,3605.9)

            self.timing_dict['d3sc']['reference_event'] = {}
            self.timing_dict['d3sc']['reference_event']['hpol'] = (5648 , 2419) #This is a peaking event
            self.timing_dict['d3sc']['reference_event']['vpol'] = (5648 , 14683)

            self.timing_dict['d3sc']['times_of_interest'] = {}
            self.timing_dict['d3sc']['times_of_interest']['hpol'] = {}
            self.timing_dict['d3sc']['times_of_interest']['vpol'] = {}        

            self.timing_dict['d3sc']['times_of_interest']['hpol']['0dB'] = (1629414990.5 , 1629415324.5)
            self.timing_dict['d3sc']['times_of_interest']['hpol']['3dB'] = (1629415642.5 , 1629416225.5)
            self.timing_dict['d3sc']['times_of_interest']['hpol']['6dB'] = (1629416355.5 , 1629416856.5)
            self.timing_dict['d3sc']['times_of_interest']['hpol']['10dB'] = (1629417014.5 , 1629417606.5)
            self.timing_dict['d3sc']['times_of_interest']['hpol']['13dB'] = (1629417625.5 , 1629418274.5)
            self.timing_dict['d3sc']['times_of_interest']['hpol']['20dB'] = (1629418320.0 , 1629418920.0)

            self.timing_dict['d3sc']['times_of_interest']['vpol']['0dB'] = (1629419128.5 , 1629419714.5)
            self.timing_dict['d3sc']['times_of_interest']['vpol']['3dB'] = (1629419753.5 , 1629420363.5)
            self.timing_dict['d3sc']['times_of_interest']['vpol']['6dB'] = (1629420372.5 , 1629420960.5)
            self.timing_dict['d3sc']['times_of_interest']['vpol']['10dB'] = (1629420986.5 , 1629421567.5)
            self.timing_dict['d3sc']['times_of_interest']['vpol']['13dB'] = (1629421587.5 , 1629422212.5)
            self.timing_dict['d3sc']['times_of_interest']['vpol']['20dB'] = (1629422262.5 , 1629422851.5)

            # Day 4
            # Site a
            self.timing_dict['d4sa'] = {}
            self.timing_dict['d4sa']['runs'] = [5655, 5656, 5657, 5659, 5660] #day 4
            self.timing_dict['d4sa']['latlonel'] = (37.59264500,-118.22765817,3741.7)
            self.timing_dict['d4sa']['reference_event'] = {}
            self.timing_dict['d4sa']['reference_event']['hpol'] = (5656 , 3021)
            self.timing_dict['d4sa']['reference_event']['vpol'] = (5657 , 4411)

            self.timing_dict['d4sa']['times_of_interest'] = {}
            self.timing_dict['d4sa']['times_of_interest']['hpol'] = {}
            self.timing_dict['d4sa']['times_of_interest']['vpol'] = {}

            self.timing_dict['d4sa']['times_of_interest']['hpol']['0dB'] = (1629481652.5 , 1629482323.5)
            self.timing_dict['d4sa']['times_of_interest']['hpol']['3dB'] = (1629482356.5 , 1629482986.5)
            self.timing_dict['d4sa']['times_of_interest']['hpol']['6dB'] = (1629483010.5 , 1629483595.5)
            self.timing_dict['d4sa']['times_of_interest']['hpol']['10dB'] = (1629483650.5 , 1629484296.5)
            self.timing_dict['d4sa']['times_of_interest']['hpol']['13dB'] = (1629484320.5 , 1629485010.5)
            self.timing_dict['d4sa']['times_of_interest']['hpol']['20dB'] = (1629485067.5 , 1629485703.5)
            
            self.timing_dict['d4sa']['times_of_interest']['vpol']['0dB'] = (1629485943.5 , 1629486590.5)
            self.timing_dict['d4sa']['times_of_interest']['vpol']['3dB'] = (1629486621.5 , 1629487202.5)
            self.timing_dict['d4sa']['times_of_interest']['vpol']['6dB'] = (1629487281.5 , 1629487886.5)
            self.timing_dict['d4sa']['times_of_interest']['vpol']['10dB'] = (1629487939.5 , 1629488473.5)
            self.timing_dict['d4sa']['times_of_interest']['vpol']['13dB'] = (1629488503.5 , 1629489200.5)
            self.timing_dict['d4sa']['times_of_interest']['vpol']['20dB'] = (1629489296.5 , 1629489857.5)

            # Site b
            self.timing_dict['d4sb'] = {}
            self.timing_dict['d4sb']['runs'] = [5658,5659,5660] #day 4
            self.timing_dict['d4sb']['latlonel'] = (37.59208167,-118.23553200,3804.9)
            self.timing_dict['d4sb']['reference_event'] = {}
            self.timing_dict['d4sb']['reference_event']['hpol'] = (5659 , 3247)
            self.timing_dict['d4sb']['reference_event']['vpol'] = (5660 , 2869)

            self.timing_dict['d4sb']['times_of_interest'] = {}
            self.timing_dict['d4sb']['times_of_interest']['hpol'] = {}
            self.timing_dict['d4sb']['times_of_interest']['vpol'] = {}

            self.timing_dict['d4sb']['times_of_interest']['hpol']['0dB'] = (1629494132.5 , 1629494733.5)
            self.timing_dict['d4sb']['times_of_interest']['hpol']['3dB'] = (1629494784.5 , 1629495411.5)
            self.timing_dict['d4sb']['times_of_interest']['hpol']['6dB'] = (1629495444.5 , 1629495965.5)
            self.timing_dict['d4sb']['times_of_interest']['hpol']['10dB'] = (1629496030.5 , 1629496607.5)
            self.timing_dict['d4sb']['times_of_interest']['hpol']['13dB'] = (1629496711.5 , 1629497449.5)
            self.timing_dict['d4sb']['times_of_interest']['hpol']['20dB'] = (1629497394.5 , 1629498072.5)
            
            self.timing_dict['d4sb']['times_of_interest']['vpol']['0dB'] = (1629498221.5 , 1629498789.5)
            self.timing_dict['d4sb']['times_of_interest']['vpol']['3dB'] = (1629498866.5 , 1629499478.5)
            self.timing_dict['d4sb']['times_of_interest']['vpol']['6dB'] = (1629499517.5 , 1629500090.5)
            self.timing_dict['d4sb']['times_of_interest']['vpol']['10dB'] = (1629500125.5 , 1629500756.5)
            self.timing_dict['d4sb']['times_of_interest']['vpol']['13dB'] = (1629500793.5 , 1629501320.5)
            self.timing_dict['d4sb']['times_of_interest']['vpol']['20dB'] = (1629501355.5 , 1629501977.5)
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getPulserCandidates(self, site, pol, attenuation=None):
        '''
        Returns the run and eventids of events that are suspected to be coming from this particular site

        If attenuation is given then this will attempt to find the events corresponding to that specific Tx attenuation.

        Attenuation should be given as a string such as "0dB".
        '''
        # Prepare arrays for output data
        try:
            info = numpy.array([],dtype=self.event_dtype)
            self.timing_dict[site]['runs']

            for run_index, run in enumerate(self.timing_dict[site]['runs']):
                sys.stdout.write('(%i/%i)\r'%(run_index + 1,len(self.timing_dict[site]['runs'])))
                sys.stdout.flush()
                reader = Reader(os.environ['BEACON_DATA'],run)
                trigger_type = loadTriggerTypes(reader)
                eventids = numpy.arange(len(trigger_type))
                trigtype_cut = trigger_type == 3

                _info = numpy.ones(sum(trigtype_cut),dtype=self.event_dtype)
                _info['run'] = run
                _info['eventid'] = eventids[trigtype_cut]
                _info['calibrated_trig_time'] = getEventTimes(reader)[trigtype_cut]

                info = numpy.append(info,_info)

            for att in list(self.timing_dict[site]['times_of_interest'][pol].keys()):
                cut = numpy.logical_and(info['calibrated_trig_time'] >= self.timing_dict[site]['times_of_interest'][pol][att][0], info['calibrated_trig_time'] <= self.timing_dict[site]['times_of_interest'][pol][att][1])
                info['attenuation_dB'][cut] = float(att.replace('dB',''))

            if attenuation is not None:
                if attenuation in list(self.timing_dict[site]['times_of_interest'][pol].keys()):
                    cut = info['attenuation_dB'] == float(att.replace('dB',''))
                    return info[cut]
                else:
                    print('Given attenuation not in dict, returning all.')
                    return info
            else:
                return info
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getPulserCandidates(self, site, pol, attenuation=None):
        '''
        This will grab the run and eventid of each event that has been selected to be a likely pulsing event.  This
        selection was done using the correlate_against_template.py script by circling events in the dominent cluster
        of the cross correlation and map max value plot.
        '''
        # Prepare arrays for output data
        try:
            info = numpy.array([],dtype=self.event_dtype)
            self.timing_dict[site]['runs']

            for run_index, run in enumerate(self.timing_dict[site]['runs']):
                sys.stdout.write('(%i/%i)\r'%(run_index + 1,len(self.timing_dict[site]['runs'])))
                sys.stdout.flush()
                reader = Reader(os.environ['BEACON_DATA'],run)
                trigger_type = loadTriggerTypes(reader)
                eventids = numpy.arange(len(trigger_type))
                trigtype_cut = trigger_type == 3

                _info = numpy.ones(sum(trigtype_cut),dtype=self.event_dtype)
                _info['run'] = run
                _info['eventid'] = eventids[trigtype_cut]
                _info['calibrated_trig_time'] = getEventTimes(reader)[trigtype_cut]

                info = numpy.append(info,_info)

            for att in list(self.timing_dict[site]['times_of_interest'][pol].keys()):
                cut = numpy.logical_and(info['calibrated_trig_time'] >= self.timing_dict[site]['times_of_interest'][pol][att][0], info['calibrated_trig_time'] <= self.timing_dict[site]['times_of_interest'][pol][att][1])
                info['attenuation_dB'][cut] = float(att.replace('dB',''))

            if attenuation is not None:
                if attenuation in list(self.timing_dict[site]['times_of_interest'][pol].keys()):
                    cut = info['attenuation_dB'] == float(att.replace('dB',''))
                    return info[cut]
                else:
                    print('Given attenuation not in dict, returning all.')
                    return info
            else:
                return info
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def returnRuns(self,site):
        return self.timing_dict[site]['runs']

    def getPulserReferenceEvent(self, site, pol):
        '''
        Returns the run and eventid of the "model" event for that site.
        '''
        try:
            reference_event = numpy.zeros(1,dtype=self.event_dtype)
            run = self.timing_dict[site]['reference_event'][pol][0]
            reader = Reader(os.environ['BEACON_DATA'],run)
            eventid = self.timing_dict[site]['reference_event'][pol][1]
            calibrated_trig_time = getEventTimes(reader)[eventid]

            db = 0
            for att in list(self.timing_dict[site]['times_of_interest'][pol].keys()):
                if numpy.logical_and(calibrated_trig_time >= self.timing_dict[site]['times_of_interest'][pol][att][0], calibrated_trig_time <= self.timing_dict[site]['times_of_interest'][pol][att][1]):
                    db = float(att.replace('dB',''))
            reference_event = numpy.array([ (run , eventid, calibrated_trig_time, db)], dtype=self.event_dtype)

            return reference_event
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def getPulserLatLonEl(self, site):
        '''
        Returns the lat lon el information from a particular site.
        '''
        try:
            return self.timing_dict[site]['latlonel']
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

if __name__ == '__main__':
    plt.close('all')
    try:
        deploy_index = info.returnDefaultDeploy()

        #Filter settings
        final_corr_length = 2**15 #Should be a factor of 2 for fastest performance
        apply_phase_response = True
        
        # crit_freq_low_pass_MHz = [100,75,75,75,75,75,75,75]
        # low_pass_filter_order = [8,12,14,14,14,14,14,14]
        
        # crit_freq_high_pass_MHz = None#30#None#50
        # high_pass_filter_order = None#5#None#8

        crit_freq_low_pass_MHz = None#[80,70,70,70,70,70,60,70] #Filters here are attempting to correct for differences in signals from pulsers.
        low_pass_filter_order = None#[0,8,8,8,10,8,3,8]

        crit_freq_high_pass_MHz = None#65
        high_pass_filter_order = None#12

        sine_subtract = True
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03
        
        plot_filters = False
        plot_multiple = False

        hilbert = True #Apply hilbert envelope to wf before correlating
        align_method = 0

        shorten_signals = True
        shorten_thresh = 0.7
        shorten_delay = 10.0
        shorten_length = 90.0

        #Sort through trigtype == 3 events and plot p2p as a function of time in a large scatter plot.
        # Just a simple list of runs with pulses.  As yet unparsed.  

        day = 3
        site = 'filler'
        if day == 2:
            pulsing_run_list = [5630, 5631,5632] #day 2
            source_latlonel = (37.5859361, -118.233918056, 3762.9)
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
            site = ['d3sa','d3sb','d3sc'][0]
            if site == 'd3sa':
                source_latlonel = ( 37.58575767,-118.22592267, 3697.4)
                pulsing_run_list = [5638,5639,5640,5641,5642]
                # 5639    8/19/2021   Pulsing SE Hpol 1 Hz no attenuation at Site 2
                # 5640    8/19/2021   Pulsing SE Site 2 Hpol 1 Hz 3dB attenuation; 10:47 AM 6 dB 1 Hz Hpol; 11:02 pulsing at 10db HPol
                # 5641    8/19/2021   11:02 pulsing at 10db Hpol; 11:12 AM 20 dB attenaution hpol ; 11:26 pulsing with 13 dB attenuation
                # 5642    8/19/2021   11:40 switchiing into Vpol; 11:48 AM Pulsing at 1 Hz with 13 dB attenuation; 12:01 PM 10 dB; 12:16 PM 6 dB 
                times_of_interest = {   'HPol 0 dB starts' :    1629393793,\
                                        'HPol 0 dB stops' :     1629394031.53,\
                                        'HPol 3 dB starts' :    1629394089.98,\
                                        'HPol 3 dB stops' :     1629395178.25,\
                                        'HPol 6 dB starts' :    1629395221.716,\
                                        'HPol 6 dB stops' :     1629396041.16,\
                                        'HPol 10 dB starts' :   1629396104.76,\
                                        'HPol 10 dB stops' :    1629396686.18,\
                                        'HPol 20 dB starts' :   1629396732.53,\
                                        'HPol 20 dB stops' :    1629397478.1,\
                                        'HPol 13 dB starts' :   1629397606.5,\
                                        'HPol 13 dB stops' :    1629398545.31,\
                                        'VPol 13 dB starts' :   1629398876.5,\
                                        'VPol 13 dB stops' :    1629399598.5,\
                                        'VPol 10 dB starts' :   1629399685.9,\
                                        'VPol 10 dB stops' :    1629400519.2,\
                                        'VPol 6 dB starts' :    1629400584.7,\
                                        'VPol 6 dB stops' :     1629401182.2,\
                                        'VPol 20 dB starts' :   1629401233.8,\
                                        'VPol 20 dB stops' :    1629401321.18,\
                                        }
            elif site == 'd3sb':
                source_latlonel = (37.58779650,-118.22452000,3619.0)
                pulsing_run_list = [5643,5644,5645,5646,5647]
                # 5643    8/19/2021   New run for time between sites (moving from site 2 to site 3); 1:29 Pulsing 1 Hz with 0dB attenuation in Hpol
                # 5644    8/19/2021   Hpol run for site 3; Starting at 1:35 PM  at Site 2 1 Hz with 0 dB attenuation in Hpol; 1:40 PM 3 dB; 1:50 PM 6 dB; 2:00 PM 10 dB; 2:10 13 dB; 2:20 20 dB
                # 5645    8/19/2021   "run between Hpol and Vpol. May have some Hpol at beginning or Vpol at the end."
                # 5646    8/19/2021   Vpol run for Site 3; Starting at 2:40 PM 1 Hz with 0 dB attenuation in Hpol; 2:50 PM 3 dB; 3:00 PM 6 dB; 3:10 PM 10 dB; 3:20PM 13 dB; 3:30 20 dB
                # 5647    8/19/2021   Likely moving to a new site (site 3 to site 4) so starting a new run
                # 5648    8/19/2021   New site (site 4); new run; Followind same pattern as before: Hpol, 1 Hz, 0, 3, 6, 10, 13, 20 dB, then Vpol 1Hz, 0, 3, 6, 10, 13, 20 dB
                # 5649    8/19/2021   Started a new run because it's been awhile. This run ran for a while after we stopped pulsing
                times_of_interest = {   'HPol 0 dB starts' :    1629404874.67,\
                                        'HPol 0 dB stops' :     1629405556.23,\
                                        'HPol 3 dB starts' :    1629405587.5,\
                                        'HPol 3 dB stops' :     1629406305.5,\
                                        'HPol 6 dB starts' :    1629406318.5,\
                                        'HPol 6 dB stops' :     1629406943.5,\
                                        'HPol 10 dB starts' :   1629407005.5,\
                                        'HPol 10 dB stops' :    1629407592.5,\
                                        'HPol 13 dB starts' :   1629407645.0,\
                                        'HPol 13 dB stops' :    1629408245.5,\
                                        'HPol 20 dB starts' :   1629408283.5,\
                                        'HPol 20 dB stops' :    1629408866.5,\
                                        'VPol 0 dB starts' :    1629409096.5,\
                                        'VPol 0 dB stops' :     1629409738.5,\
                                        'VPol 3 dB starts' :    1629409778.5,\
                                        'VPol 3 dB stops' :     1629410419.5,\
                                        'VPol 6 dB starts' :    1629410439.5,\
                                        'VPol 6 dB stops' :     1629411250.5,\
                                        'VPol 10 dB starts' :   1629411273.5,\
                                        'VPol 10 dB stops' :    1629411870.5,\
                                        'VPol 13 dB starts' :   1629411890.5,\
                                        'VPol 13 dB stops' :    1629412556.5,\
                                        'VPol 20 dB starts' :   1629412580.5,\
                                        'VPol 20 dB stops' :    1629413165.5,\
                                        }
            elif site == 'd3sc':
                source_latlonel = (37.58885717,-118.22786317,3605.9)
                pulsing_run_list = [5648,5649]
                # 5648    8/19/2021   New site (site 4); new run; Followind same pattern as before: Hpol, 1 Hz, 0, 3, 6, 10, 13, 20 dB, then Vpol 1Hz, 0, 3, 6, 10, 13, 20 dB
                # 5649    8/19/2021   Started a new run because it's been awhile. This run ran for a while after we stopped pulsing
                times_of_interest = {   'HPol 0 dB starts' :    1629414990.5,\
                                        'HPol 0 dB stops' :     1629415324.5,\
                                        'HPol 3 dB starts' :    1629415642.5,\
                                        'HPol 3 dB stops' :     1629416225.5,\
                                        'HPol 6 dB starts' :    1629416355.5,\
                                        'HPol 6 dB stops' :     1629416856.5,\
                                        'HPol 10 dB starts' :   1629417014.5,\
                                        'HPol 10 dB stops' :    1629417606.5,\
                                        'HPol 13 dB starts' :   1629417625.5,\
                                        'HPol 13 dB stops' :    1629418274.5,\
                                        'HPol 20 dB starts' :   1629418320.0,\
                                        'HPol 20 dB stops' :    1629418920.0,\
                                        'VPol 0 dB starts' :    1629419128.5,\
                                        'VPol 0 dB stops' :     1629419714.5,\
                                        'VPol 3 dB starts' :    1629419753.5,\
                                        'VPol 3 dB stops' :     1629420363.5,\
                                        'VPol 6 dB starts' :    1629420372.5,\
                                        'VPol 6 dB stops' :     1629420960.5,\
                                        'VPol 10 dB starts' :   1629420986.5,\
                                        'VPol 10 dB stops' :    1629421567.5,\
                                        'VPol 13 dB starts' :   1629421587.5,\
                                        'VPol 13 dB stops' :    1629422212.5,\
                                        'VPol 20 dB starts' :   1629422262.5,\
                                        'VPol 20 dB stops' :    1629422851.5,\
                                        }

        elif day == 4:
            site = ['d4sa','d4sb'][0]
            pulsing_run_list = [5655, 5656,5657,5659,5660] #day 4
            if site == 'd4sa':
                source_latlonel = (37.59264500,-118.22765817,3741.7)

                reference_event_hpol = (5656 , 3021)
                reference_event_vpol = (5657 , 4411)

                pulsing_run_list = [5655, 5656,5657] #day 4
                times_of_interest = {   'HPol 0 dB starts' :    1629481652.5,\
                                        'HPol 0 dB stops' :     1629482323.5,\
                                        'HPol 3 dB starts' :    1629482356.5,\
                                        'HPol 3 dB stops' :     1629482986.5,\
                                        'HPol 6 dB starts' :    1629483010.5,\
                                        'HPol 6 dB stops' :     1629483595.5,\
                                        'HPol 10 dB starts' :   1629483650.5,\
                                        'HPol 10 dB stops' :    1629484296.5,\
                                        'HPol 13 dB starts' :   1629484320.5,\
                                        'HPol 13 dB stops' :    1629485010.5,\
                                        'HPol 20 dB starts' :   1629485067.5,\
                                        'HPol 20 dB stops' :    1629485703.5,\
                                        'VPol 0 dB starts' :    1629485943.5,\
                                        'VPol 0 dB stops' :     1629486590.5,\
                                        'VPol 3 dB starts' :    1629486621.5,\
                                        'VPol 3 dB stops' :     1629487202.5,\
                                        'VPol 6 dB starts' :    1629487281.5,\
                                        'VPol 6 dB stops' :     1629487886.5,\
                                        'VPol 10 dB starts' :   1629487939.5,\
                                        'VPol 10 dB stops' :    1629488473.5,\
                                        'VPol 13 dB starts' :   1629488503.5,\
                                        'VPol 13 dB stops' :    1629489200.5,\
                                        'VPol 20 dB starts' :   1629489296.5,\
                                        'VPol 20 dB stops' :    1629489857.5,\
                                        }
            elif site == 'd4sb':
                source_latlonel = (37.59208167,-118.23553200,3804.9)

                reference_event_hpol = (5659 , 3247)
                reference_event_vpol = (5660 , 2869)

                pulsing_run_list = [5658,5659,5660] #day 4
                times_of_interest = {   'HPol 0 dB starts' :    1629494132.5,\
                                        'HPol 0 dB stops' :     1629494733.5,\
                                        'HPol 3 dB starts' :    1629494784.5,\
                                        'HPol 3 dB stops' :     1629495411.5,\
                                        'HPol 6 dB starts' :    1629495444.5,\
                                        'HPol 6 dB stops' :     1629495965.5,\
                                        'HPol 10 dB starts' :   1629496030.5,\
                                        'HPol 10 dB stops' :    1629496607.5,\
                                        'HPol 13 dB starts' :   1629496711.5,\
                                        'HPol 13 dB stops' :    1629497449.5,\
                                        'HPol 20 dB starts' :   1629497394.5,\
                                        'HPol 20 dB stops' :    1629498072.5,\
                                        'VPol 0 dB starts' :    1629498221.5,\
                                        'VPol 0 dB stops' :     1629498789.5,\
                                        'VPol 3 dB starts' :    1629498866.5,\
                                        'VPol 3 dB stops' :     1629499478.5,\
                                        'VPol 6 dB starts' :    1629499517.5,\
                                        'VPol 6 dB stops' :     1629500090.5,\
                                        'VPol 10 dB starts' :   1629500125.5,\
                                        'VPol 10 dB stops' :    1629500756.5,\
                                        'VPol 13 dB starts' :   1629500793.5,\
                                        'VPol 13 dB stops' :    1629501320.5,\
                                        'VPol 20 dB starts' :   1629501355.5,\
                                        'VPol 20 dB stops' :    1629501977.5,\
                                        }

        # Prepare expected angle and arrival times
        origin = info.loadAntennaZeroLocation()
        enu = pm.geodetic2enu(source_latlonel[0],source_latlonel[1],source_latlonel[2],origin[0],origin[1],origin[2])
        source_distance_m = numpy.sqrt(enu[0]**2 + enu[1]**2 + enu[2]**2)
        azimuth_deg = numpy.rad2deg(numpy.arctan2(enu[1],enu[0]))
        zenith_deg = numpy.rad2deg(numpy.arccos(enu[2]/source_distance_m))

        cor_reader = Reader(os.environ['BEACON_DATA'],pulsing_run_list[0])
        cor = Correlator(cor_reader,upsample=len(cor_reader.t())*8, n_phi=180*4 + 1, n_theta=360*4 + 1,deploy_index=deploy_index)
        
        expected_time_delays = -predictAlignment(azimuth_deg, zenith_deg, cor, pol='hpol')

        

        # Prepare arrays for output data
        all_values_p2p = numpy.array([])
        all_values_p2p = numpy.array([])

        info_dtype = numpy.dtype([  
                                    ('run','i'),
                                    ('eventid','i'),
                                    ('p2p_h','i'),
                                    ('p2p_v','i'),
                                    ('calibrated_trig_time','float64'),
                                    ('simple_powersum_peak','float64')
                                ])

        info = numpy.array([],dtype=info_dtype)

        for run_index, run in enumerate(pulsing_run_list):


            sys.stdout.write('(%i/%i)\r'%(run_index + 1,len(pulsing_run_list)))
            sys.stdout.flush()
            reader = Reader(os.environ['BEACON_DATA'],run)

            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=(None,None),plot_filters=False,apply_phase_response=apply_phase_response)
            

            sample_indices = numpy.array([numpy.argmin(numpy.abs(tdc.corr_time_shifts - _etd)) for _etd in expected_time_delays])
            
            #tct = TemplateCompareTool(reader,apply_phase_response=True)

            trigger_type = loadTriggerTypes(reader)
            eventids = numpy.arange(len(trigger_type))
            trigtype_cut = trigger_type == 3

            _info = numpy.ones(sum(trigtype_cut),dtype=info_dtype)
            _info['run'] = run
            _info['eventid'] = eventids[trigtype_cut]
            _info['calibrated_trig_time'] = getEventTimes(reader)[trigtype_cut]

            if site == 'd3sb':
                t_cut = numpy.logical_and(reader.t() > 5500, reader.t() < 6000)
                # elif site == 'd3sc':
                #t_cut = numpy.logical_and(reader.t() > 4400, reader.t() < 5400)
                #t_cut = numpy.logical_and(reader.t() > 900, reader.t() < 1200)
            else:
                t_cut = numpy.ones_like(reader.t(),dtype=bool)
            for event_index, eventid in enumerate(_info['eventid']):
                reader.setEntry(eventid)

                max_p2p_hpol = 0
                max_p2p_vpol = 0
                for ch in range(8):
                    wf = reader.wf(ch)[t_cut]
                    if ch%2 == 0:
                        max_p2p_hpol = max( max_p2p_hpol , max(wf) - min(wf) )                    
                    else:
                        max_p2p_vpol = max( max_p2p_vpol , max(wf) - min(wf) )
                
                _info['p2p_h'][event_index] = max_p2p_hpol
                _info['p2p_v'][event_index] = max_p2p_vpol

                if False and max_p2p_hpol > 60:
                    indices, time_shift, corr_value, _pairs, corrs = tdc.calculateTimeDelaysFromEvent(eventid,hilbert=hilbert,return_full_corrs=True,align_method_10_estimate=tdc.corr_time_shifts[sample_indices],align_method_10_window_ns=10,shorten_signals=shorten_signals, shorten_thresh=shorten_thresh, shorten_delay=shorten_delay, shorten_length=shorten_length,sine_subtract=sine_subtract, crosspol_delays=None) #Using default of the other function
                    tdc.plotEvent(eventid, channels=[0,1,2,3,4,5,6,7], apply_filter=False, hilbert=False, sine_subtract=False, apply_tukey=None, additional_title_text=None, time_delays=None, verbose=False)
                    for corr_index, corr in enumerate(corrs):
                        plt.figure()
                        plt.plot(tdc.corr_time_shifts, corr, label=_pairs[corr_index])
                        plt.axvline(tdc.corr_time_shifts[sample_indices[corr_index]], c='k')
                        plt.legend()
                        plt.xlim(-200,200)
                    import pdb; pdb.set_trace()
            info = numpy.append(info,_info)

        print('\nInfo preparation complete')

        lassos = []
        
        
        plt.figure()
        ax = plt.gca()
        w = 20 #Width of rolling average
        plt.title('Day %i %s\nPeak to Peak with %i Event Moving Window'%(day, site, w))
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_h'],w)
        plt.plot(x,y,alpha=0.7,c='#94D0CC',linestyle = '-')
        
        scatter = plt.scatter(x,y,alpha=0.7,c='#94D0CC',label='Hpol P2P',)        
        _s = Selector(ax,scatter,info,deploy_index)
        lassos.append(_s)
        
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_v'],w)
        plt.plot(x,y,alpha=0.7,c='#F29191',linestyle = '-.')
        
        scatter = plt.scatter(x,y,alpha=0.7,c='#F29191',label='Vpol P2P',)
        _s = Selector(ax,scatter,info,deploy_index)
        lassos.append(_s)

        plt.ylabel('p2p')
        plt.xlabel('Calibrated Trigger Time')

        for key, time in times_of_interest.items():
            plt.axvline(time, label=key, color = next(ax._get_lines.prop_cycler)['color'])
        plt.legend(fontsize=8)


        plt.figure()
        ax2 = plt.gca(sharex=ax)
        w = 20 #Width of rolling average
        plt.title('Day %i %s\nLine Represents Peak to Peak with %i Event Moving Window'%(day, site, w))
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_h'],w)
        plt.plot(x,y,alpha=0.7,c='#94D0CC',linestyle = '-')

        x = info['calibrated_trig_time']
        y = info['p2p_h']
        scatter = plt.scatter(x,y,alpha=0.7,c='#94D0CC',label='Hpol P2P',)        
        _s = Selector(ax2,scatter,info,deploy_index)
        lassos.append(_s)
        
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_v'],w)
        plt.plot(x,y,alpha=0.7,c='#F29191',linestyle = '-.')

        x = info['calibrated_trig_time']
        y = info['p2p_v']
        scatter = plt.scatter(x,y,alpha=0.7,c='#F29191',label='Vpol P2P',)
        _s = Selector(ax2,scatter,info,deploy_index)
        lassos.append(_s)

        plt.ylabel('p2p')
        plt.xlabel('Calibrated Trigger Time')

        for key, time in times_of_interest.items():
            plt.axvline(time, label=key, color = next(ax2._get_lines.prop_cycler)['color'])
        plt.legend(fontsize=8)


        plt.figure()
        ax2 = plt.gca(sharex=ax)
        w = 20 #Width of rolling average
        plt.title('Day %i %s\nLine Represents Peak to Peak with %i Event Moving Window'%(day, site, w))
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_h'],w)
        plt.plot(x,y,alpha=0.7,c='#94D0CC',linestyle = '-')

        x = info['calibrated_trig_time']
        y = info['p2p_h']
        scatter = plt.scatter(x,y,alpha=0.7,c='#94D0CC',label='Hpol P2P',)        
        _s = Selector(ax2,scatter,info,deploy_index)
        lassos.append(_s)
        
        plt.ylabel('p2p')
        plt.xlabel('Calibrated Trigger Time')

        for key, time in times_of_interest.items():
            if 'vpol' in key:
                continue
            plt.axvline(time, label=key, color = next(ax2._get_lines.prop_cycler)['color'])
        plt.legend(fontsize=8)


        plt.figure()
        ax2 = plt.gca(sharex=ax)
        w = 20 #Width of rolling average
        plt.title('Day %i %s\nLine Represents Peak to Peak with %i Event Moving Window'%(day, site, w))
        
        x = moving_average(info['calibrated_trig_time'],w)
        y = moving_average(info['p2p_v'],w)
        plt.plot(x,y,alpha=0.7,c='#F29191',linestyle = '-.')

        x = info['calibrated_trig_time']
        y = info['p2p_v']
        scatter = plt.scatter(x,y,alpha=0.7,c='#F29191',label='Vpol P2P',)
        _s = Selector(ax2,scatter,info,deploy_index)
        lassos.append(_s)
        
        plt.ylabel('p2p')
        plt.xlabel('Calibrated Trigger Time')

        for key, time in times_of_interest.items():
            if 'hpol' in key:
                continue
            plt.axvline(time, label=key, color = next(ax2._get_lines.prop_cycler)['color'])
        plt.legend(fontsize=8)


    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

