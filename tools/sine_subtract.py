'''
This script contains the SineSubtract class.  This code was originally written by KA Hughes as a wrapper for ROOT code
written by C Deaconu.  

KA Hughes Original:
https://github.com/vPhase/PA_Analysis/blob/master/src/analysis/EventFilterClass.py

C Deaconu Original:
https://github.com/nichol77/libRootFftwWrapper/blob/master/src/SineSubtract.cxx
'''

import os
from os import path
import ROOT
#import ctypes

sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])


import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import hilbert,butter,lfilter
import scipy.optimize as opt
from collections import defaultdict
from scipy.interpolate import interp1d, Akima1DInterpolator
import numpy
import sys
import math
import matplotlib
import itertools

from scipy.fftpack import fft

import time
import pandas as pd
from RZ_CorrelatorClass import RZ_Correlate

from utils.FFT_Tools import ReturnFFTForPlot

ROOT.gInterpreter.ProcessLine('#include "%s"'%(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'include/FFTtools.h'))
ROOT.gSystem.Load(os.environ['LIB_ROOT_FFTW_WRAPPER_DIR'] + 'build/libRootFftwWrapper.so.3')

from ROOT import FFTtools
#import h5py

font = {'weight' : 'bold',
        'size'   : 18}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 26})
plt.style.use('seaborn-dark-palette')

class SineSubtract:
    '''
    This is a wrapper class written by KA Hughes for the Sine Subtraction ROOT code written by C Deaconu.  This class
    will attempt time domain subtraction of sine waves from signals, with the goal of removing CW from signals.  It
    is done in the time domain to account for the fact the a time windowed sine wave does not result in a single freq
    in the frequency domain.  

    This code was originally used for ARA phased array analysis, for which reasonable values seemed to be:
    min_power_ratio = 0.1 (originally 0.05)
    min_freq = 0.2 (in GHz)
    max_freq = 0.7 (in GHz)


    Parameters
    ----------
    reader : examples.beacon_data_reader.reader
            The run reader you wish to load signals from.
    min_freq : float
        The minium frequency to be considered part of the same known CW source.  This should be given in GHz.  
    max_freq : float
        The maximum frequency to be considered part of the same known CW source.  This should be given in GHz.  
    min_power_ratio : float
        This is the threshold for a prominence to be considered CW.  If the power in the band defined by min_freq and
        max_freq contains more than min_power_ratio percent (where min_power_ratio <= 1.0) of the total signal power,
        then it is considered a CW source, and will be removed. 
    max_failed_iterations : int
        This sets a limiter on the number of attempts to make when removing signals, before exiting.
    '''
    def __init__(self, reader, min_freq, max_freq, min_power_ratio, max_failed_iterations=3, verbose=False):
        self.reader = reader
        self.sine_subtract = FFTtools.SineSubtract(max_failed_iterations, min_power_ratio,False)
        self.sine_subtract.setVerbose(False) #Don't print a bunch to the screen
        self.sine_subtract.setFreqLimits(min_freq, max_freq)

    def CWFilter(self):
        '''        
        '''
        #create time array and calculate dt
        t_array = self.reader.t()
        dt = t_array[1]-t_array[0]

        #self.freqs saves all frequencies found. self.saved_freqs saves the first frequency found (not all frequencies found) in a dictionary
        self.freqs = numpy.zeros([8,4])
        self.saved_freqs = {}
        #loop over channels
        for ch in range(0,8):
            #load in voltages and create an output array
            v_array = self.reader.wfs[ch]
            v_out = numpy.zeros(len(v_array),dtype=numpy.double)

            #Do the sine subtraction
            self.sine_subtract.subtractCW(len(v_array),v_array.data,dt,v_out)

            #Check how many solutions were found
            n_fit = self.sine_subtract.getNSines()

            #Save all frequencies in array
            self.freqs[ch,:n_fit]=(numpy.frombuffer(self.sine_subtract.getFreqs(),dtype=numpy.float64,count=n_fit))

            #if at least one soluion was found, update waveforms to match v_out
            if(n_fit>0):
                self.reader.wfs[ch+100]=v_out #regular waveform
                self.reader.wfs_up[ch+100]=signal.resample(v_out,len(v_out)*d.upsample) #resampled waveform
                self.saved_freqs[ch]=self.freqs[ch,0] #first frequency removed
            else:
                #if no solutions are found, no need to update waveforms
                self.saved_freqs[ch]=0.0 


Filter = EventFilter()
CWsub_narrow = SineSubtract(0.4,0.41,0.05)
CWsub_broad = SineSubtract(0.2,0.7,0.10)



remake_timemaps = 0
kRef = 1
upsample = 5
resolution = 'middlesurface'
#head_val = 'Decimated'
Correlator = RZ_Correlate(remake_timemaps,resolution)

if(len(sys.argv)==1 or sys.argv[1]=='sim'):
    sim=1
    
    d = nuphase_sim_reader.SimReader(AraFile,AraNoise,1,upsample)
    kCab = 0
    prob_array = numpy.zeros(8)
    PA_run = 'sim'
    kPlotter = 0

    firstEvent = 0 if len(sys.argv)==1 else int(sys.argv[2])
    lastEvent = d.N() if len(sys.argv)==1 else int(sys.argv[3])
    if(lastEvent>d.N()):
        lastEvent=d.N()
    print('Running over simulated events.', PA_run, firstEvent,lastEvent)

if(len(sys.argv)==2):
    sim=0
    PA_run = int(sys.argv[1])
    
    if(PA_run<1500):
        therm = 1
        d = nuphase_data_reader.Reader(PA_run,upsample=upsample,thermal=therm)

    else:
        therm=0
        if(PA_run in [2053,2054,2055,2056,2057,2058,2059,2060,2268]):
            head_type=None
        else:
            head_type = 'decimated'
            d_full = nuphase_data_reader.Reader(PA_run)
            header_full =d_full.header()
        d = nuphase_data_reader.Reader(PA_run,upsample=upsample,thermal=therm,header_type=head_type)



    kCab = 1
    d.CorrectRMS()
    print('rms is ', d.rms_ch)
    prob_array =Filter.FixAlignment(PA_run,thermal=therm)
    kPlotter = 0

    firstEvent = 0 if len(sys.argv)==2 else int(sys.argv[2])
    lastEvent = d.N() if len(sys.argv)==2 else int(sys.argv[3])
    print('Running over data events.', PA_run, firstEvent,lastEvent)

"""
for ev in range(0,d.N()):
    #print(ev)
    d.setEntry(ev)
    d.load_wfs()
    print(ev,numpy.max(d.wf(0)))
    h = d.header()

    print(ev,h.event_number,numpy.max(d.wf(0)))
    
    
    if(numpy.max(d.wf(0))<0):
        plt.plot(d.t(),d.wf(0))
        plt.show()
"""
"""
for ev in range(0,200):
    #time0 = time.time()
    
    #h = d.header()
    d.setEntry(ev)
    h = d.header()
    #d.load_wfs()
    #plt.plot(d.t(),d.wf(0))
    #plt.show()
    
    print(ev,numpy.max(d.wf(0)))
    if(numpy.max(d.wf(0))>100):
        plt.plot(d.t(),d.wf(0))
        plt.show()

    #CWsub_narrow.CWFilter(d)
    #CWsub_broad.CWFilter(d)

#130,2135
#131,4862
"""
"""
if (len(sys.argv) > 1):
    sim = 0
else:
    sim = 1


if(sim==1):
    d = nuphase_sim_reader.SimReader(AraFile,AraNoise,1,upsample)
    kCab = 0
    prob_array = numpy.zeros(8)
    PA_run = 'sim'
    kPlotter = 0


else:
    PA_run = int(sys.argv[1])
    if(PA_run<1500):
        therm = 1
        d = nuphase_data_reader.Reader(PA_run,upsample=upsample,thermal=therm)

    else:
        therm=0
        d = nuphase_data_reader.Reader(PA_run,upsample=upsample,thermal=therm)
    kCab = 1
    d.CorrectRMS()
    prob_array =Filter.FixAlignment(PA_run,thermal=therm)
    kPlotter = 0
"""
#totalEvents = d.N()
#CheckRunValues(d)
#Lists for dataframe:
trig_type = []
read_time = []
avg_SNR = []
max_cor = []
max_cord = []
max_corr = []
best_r = []
best_z = []
best_r_r = []
best_z_r = []
best_r_d = []
best_z_d = []

ev_num = []


is_calpulse = []
weights = []
true_SNR = []


imp_vals_d = []
lin_rvals_d = []
lin_slopes_d = []
lin_ints_d = []
power_spot_d = []
c5_cor_d = []

imp_vals_r = []
lin_rvals_r = []
lin_slopes_r = []
lin_ints_r = []
power_spot_r = []
c5_cor_r = []

imp_vals_b = []
lin_rvals_b = []
lin_slopes_b = []
lin_ints_b = []
power_spot_b = []
c5_cor_b = [] 

energy = []
ks_test_d = []
ks_test_r = []
ks_test_b = []

refracted = []

pos_x = []
pos_y = []
pos_z = []

surface_cor_d = []
surface_r_d = []
surface_z_d = []


freqs_n0 = []
freqs_n1 = []
freqs_n2 = []
freqs_n3 = []
freqs_n4 = []
freqs_n5 = []
freqs_n6 = []
freqs_n7 = []

freqs_b0 = []
freqs_b1 = []
freqs_b2 = []
freqs_b3 = []
freqs_b4 = []
freqs_b5 = []
freqs_b6 = []
freqs_b7 = []

hilbert_peak = []
#beam_num = []


print('starting to loop over events in run : ', PA_run)
"""
rms_vals = defaultdict(list)
for ev in range(0,totalEvents):
    d.setEntry(ev)
    h = d.header()
    if(h.isRFTrigger()==True):
        d.load_wfs()
        for ch in range(100,108):
            rms_vals[ch].append(numpy.std(d.wfs[ch]))

counter = 1
for i in range(0,8):
    plt.subplot(4,2,counter)
    plt.hist(rms_vals[i+100],bins=25)
    counter = counter+1
    print(i,numpy.mean(rms_vals[i+100]))
plt.show()
"""
#time0 = time.time()
#event 33739
for ev in range(33739,lastEvent):
    #time0 = time.time()
    #print(ev)
    if(ev%100==0):
        print(ev/(lastEvent-firstEvent)*100)
    if(ev%100==0):
        df = pd.DataFrame({'Trig_Type':trig_type,'Event_time':read_time,'SNR':avg_SNR,'MaxCor':max_cor,'MaxCor_d':max_cord,'MaxCor_r':max_corr,'BestR':best_r,'BestZ':best_z,'BestR_R':best_r_r,"BestZ_R":best_z_r,'BestR_D':best_r_d,'BestZ_D':best_z_d,
            'EventNumber':ev_num, 'Is_CalPulser':is_calpulse,'Weight':weights, 'Energy': energy, 'Refracted':refracted, 'TrueSNR':true_SNR,
            'PosX':pos_x,'PosY':pos_y,'PosZ':pos_z, 'HilbertPeak':hilbert_peak, 'SurfaceCor': surface_cor_d, 'SurfaceR':surface_r_d,'SurfaceZ':surface_z_d,
            'NarrowFreq0': freqs_n0,'NarrowFreq1': freqs_n1,'NarrowFreq2': freqs_n2,'NarrowFreq3': freqs_n3,'NarrowFreq4': freqs_n4,'NarrowFreq5': freqs_n5,'NarrowFreq6': freqs_n6,'NarrowFreq7': freqs_n7,
            'BroadFreq0': freqs_b0,'BroadFreq1': freqs_b1,'BroadFreq2': freqs_b2,'BroadFreq3': freqs_b3,'BroadFreq4': freqs_b4,'BroadFreq5': freqs_b5,'BroadFreq6': freqs_b6,'BroadFreq7': freqs_b7,
            'Impulsivity_d':imp_vals_d,'rval_d':lin_rvals_d,'slopes_d': lin_slopes_d, 'intercepts_d':lin_ints_d,'PowerSpot_d':power_spot_d, 'A5_cor_d':c5_cor_d, 'ks_d':ks_test_d,
            'Impulsivity_r':imp_vals_r,'rval_r':lin_rvals_r,'slopes_r': lin_slopes_r, 'intercepts_r':lin_ints_r,'PowerSpot_r':power_spot_r,'A5_cor_r':c5_cor_r,'ks_r':ks_test_r,
            'Impulsivity_b':imp_vals_b,'rval_b':lin_rvals_b,'slopes_b': lin_slopes_b, 'intercepts_b':lin_ints_b,'PowerSpot_b':power_spot_b,'A5_cor_b':c5_cor_b,'ks_b':ks_test_b})
        df = df.set_index('EventNumber')
        if(sim==1):
            df.to_pickle('/scratch/midway2/kahughes/burn_output/run_'+str(PA_run)+'_'+str(firstEvent)+'dataframe.pickle')
        else:
            df.to_pickle('/scratch/midway2/kahughes/burn_output/run_'+str(PA_run)+'_dataframe.pickle')
    d.setEntry(ev)
    h = d.header()
    d.load_wfs()
    #print(d.event_entry)
    #plt.plot(d.t(),d.wf(0))
    #plt.show()
    
    #print(h.event_number,numpy.max(d.wf(0)))
    #print(ev,d.event_entry)
    if(sim==0 and h.isSurfaceTrigger()==1):
        #print('surface trigger!!',ev,d.event_entry)
        continue
    if(PA_run==2103 and h.isGated()==0):
        continue
    if(sim==0):
        trig_type.append(h.isRFTrigger())
        read_time.append(h.getReadoutTimeFloat())
        is_calpulse.append(h.isGated())
        #if(is_calpulse[-1]==True):
        #    print('yes')
        weights.append(1)
        energy.append(0.0)
        refracted.append(0.0)

        pos_x.append(0.0)
        pos_y.append(0.0)
        pos_z.append(0.0)

        true_SNR.append(0.0)
        if(head_type=='decimated'):
            #print('here')
            ev_num.append(d_full.head_tree.GetEntryNumberWithIndex(int(h.event_number%1e9),h.isSurfaceTrigger()))
        else:
            ev_num.append(d.head_tree.GetEntryNumberWithIndex(int(h.event_number%1e9),h.isSurfaceTrigger()))
        #print(is_calpulse[-1])
    elif(sim==1):
        trig_type.append(True)
        read_time.append(0.0)
        is_calpulse.append(False)
        weights.append(d.getWeight())
        energy.append(d.getEnergy())
        refracted.append(d.getEventType())

        true_SNR.append(d.getTrueSNR())
        #print('true SNR: ', true_SNR[-1])
        pos_x.append(d.pos_x)
        pos_y.append(d.pos_y)
        pos_z.append(d.pos_z)
        ev_num.append(ev)
    CWsub_narrow.CWFilter(d)
    CWsub_broad.CWFilter(d)

    #print(ev_num[-1],numpy.max(d.wf(0)))

    freqs_n0.append(CWsub_narrow.saved_freqs[0])
    freqs_n1.append(CWsub_narrow.saved_freqs[1])
    freqs_n2.append(CWsub_narrow.saved_freqs[2])
    freqs_n3.append(CWsub_narrow.saved_freqs[3])
    freqs_n4.append(CWsub_narrow.saved_freqs[4])
    freqs_n5.append(CWsub_narrow.saved_freqs[5])
    freqs_n6.append(CWsub_narrow.saved_freqs[6])
    freqs_n7.append(CWsub_narrow.saved_freqs[7])

    freqs_b0.append(CWsub_broad.saved_freqs[0])
    freqs_b1.append(CWsub_broad.saved_freqs[1])
    freqs_b2.append(CWsub_broad.saved_freqs[2])
    freqs_b3.append(CWsub_broad.saved_freqs[3])
    freqs_b4.append(CWsub_broad.saved_freqs[4])
    freqs_b5.append(CWsub_broad.saved_freqs[5])
    freqs_b6.append(CWsub_broad.saved_freqs[6])
    freqs_b7.append(CWsub_broad.saved_freqs[7])

    Correlator.GenerateCorMap(d,ev,kPlotter,kRef,kCab,prob_array,sim)
    
    

    dd_wf_d,dd_wf_r,dd_wf_b, dd_t = Filter.MakeDeDispersed(d,Correlator.best_ts_d,Correlator.best_ts_r,Correlator.best_ts_b,upsample)
    
    imp_vals_d.append(Filter.impulsive_value(dd_wf_d,dd_t))
    
    hilbert_peak.append(Filter.HilbertPeak)
    #print(hilbert_peak[-1])

    lin_rvals_d.append(Filter.r_value)
    lin_slopes_d.append(Filter.slope)
    lin_ints_d.append(Filter.intercept)
    power_spot_d.append(Filter.maxspot)
    c5_cor_d.append(Filter.a5_cormax_d)
    ks_test_d.append(Filter.ks)

    imp_vals_r.append(Filter.impulsive_value(dd_wf_r,dd_t))
    lin_rvals_r.append(Filter.r_value)
    lin_slopes_r.append(Filter.slope)
    lin_ints_r.append(Filter.intercept)
    power_spot_r.append(Filter.maxspot)
    c5_cor_r.append(Filter.a5_cormax_r)
    ks_test_r.append(Filter.ks)

    imp_vals_b.append(Filter.impulsive_value(dd_wf_b,dd_t))
    lin_rvals_b.append(Filter.r_value)
    lin_slopes_b.append(Filter.slope)
    lin_ints_b.append(Filter.intercept)
    power_spot_b.append(Filter.maxspot)
    c5_cor_b.append(Filter.a5_cormax_b)
    ks_test_b.append(Filter.ks)

    avg_SNR.append(Correlator.SNR)
    max_cor.append(Correlator.max_cor)
    max_cord.append(Correlator.max_cor_d)
    max_corr.append(Correlator.max_cor_r)
    best_r.append(Correlator.best_r)
    best_z.append(Correlator.best_z)
    best_r_r.append(Correlator.best_r_r)
    best_z_r.append(Correlator.best_z_r)
    best_r_d.append(Correlator.best_r_d)
    best_z_d.append(Correlator.best_z_d)
    
    surface_cor_d.append(Correlator.max_surface_d)
    surface_r_d.append(Correlator.surface_r_d)
    surface_z_d.append(Correlator.surface_z_d)
    #print(time.time()-time0)
#time1 = time.time()
#print('elapsed time: ', time1-time0)
#pd.DataFrame({'A': a_list, 'B': b_list})
df = pd.DataFrame({'Trig_Type':trig_type,'Event_time':read_time,'SNR':avg_SNR,'MaxCor':max_cor,'MaxCor_d':max_cord,'MaxCor_r':max_corr,'BestR':best_r,'BestZ':best_z,'BestR_R':best_r_r,"BestZ_R":best_z_r,'BestR_D':best_r_d,'BestZ_D':best_z_d,
            'EventNumber':ev_num, 'Is_CalPulser':is_calpulse,'Weight':weights, 'Energy': energy, 'Refracted':refracted, 'TrueSNR':true_SNR,
            'PosX':pos_x,'PosY':pos_y,'PosZ':pos_z, 'HilbertPeak':hilbert_peak, 'SurfaceCor': surface_cor_d, 'SurfaceR':surface_r_d,'SurfaceZ':surface_z_d,
            'NarrowFreq0': freqs_n0,'NarrowFreq1': freqs_n1,'NarrowFreq2': freqs_n2,'NarrowFreq3': freqs_n3,'NarrowFreq4': freqs_n4,'NarrowFreq5': freqs_n5,'NarrowFreq6': freqs_n6,'NarrowFreq7': freqs_n7,
            'BroadFreq0': freqs_b0,'BroadFreq1': freqs_b1,'BroadFreq2': freqs_b2,'BroadFreq3': freqs_b3,'BroadFreq4': freqs_b4,'BroadFreq5': freqs_b5,'BroadFreq6': freqs_b6,'BroadFreq7': freqs_b7,
            'Impulsivity_d':imp_vals_d,'rval_d':lin_rvals_d,'slopes_d': lin_slopes_d, 'intercepts_d':lin_ints_d,'PowerSpot_d':power_spot_d, 'A5_cor_d':c5_cor_d, 'ks_d':ks_test_d,
            'Impulsivity_r':imp_vals_r,'rval_r':lin_rvals_r,'slopes_r': lin_slopes_r, 'intercepts_r':lin_ints_r,'PowerSpot_r':power_spot_r,'A5_cor_r':c5_cor_r,'ks_r':ks_test_r,
            'Impulsivity_b':imp_vals_b,'rval_b':lin_rvals_b,'slopes_b': lin_slopes_b, 'intercepts_b':lin_ints_b,'PowerSpot_b':power_spot_b,'A5_cor_b':c5_cor_b,'ks_b':ks_test_b})
df = df.set_index('EventNumber')
if(sim==1):
    df.to_pickle('/scratch/midway2/kahughes/burn_output/run_'+str(PA_run)+'_'+str(firstEvent)+'dataframe.pickle')
else:
    df.to_pickle('/scratch/midway2/kahughes/burn_output/run_'+str(PA_run)+'_dataframe.pickle')
#df = pd.DataFrame({'Trig_Type':trig_type,'Event_time':read_time,'SNR':avg_SNR,'Impulsivity':imp_vals,'MaxCor':max_cor,'MaxCor_d':max_cord,'MoxCor_r':max_corr})
#df.to_pickle('run_'+str(PA_run)+'_dataframe.pickle')
#print(df)
#print(df['Impulsivity'])
#plt.hist(df['Impulsivity'])
#plt.show()
"""
if(sim==0 and totalEvents==d.N()):
    numpy.save('impulsivity_run_'+str(PA_run)+'.npy',numpy.asarray(impulsivity_vals))

plt.hist(imp_vals,bins=25)
plt.xlabel('Impulsivity')
if(sim==1):
    plt.title("1000 Simulated Events")
else:
    plt.title("Run 2054")
#plt.xlim([])
plt.grid()
plt.show()
"""
del CWsub_narrow
del CWsub_broad