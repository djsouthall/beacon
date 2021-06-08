#!/usr/bin/env python3
'''
This is intended to plot s21 and s11 data from measurements made on 6/8/2021.  
'''

import numpy
import scipy.spatial
import scipy.signal
from scipy.optimize import curve_fit
import os
import sys
import csv
import inspect
import glob

import beacon.tools.field_fox as ff

import matplotlib.pyplot as plt
from matplotlib import lines


params = {'legend.fontsize': 20,
          'figure.figsize': (14, 6.5),
         'axes.labelsize': 24,
         'axes.titlesize':24,
         'xtick.labelsize':24,
         'ytick.labelsize':24}
plt.rcParams.update(params)


def gainFromS21LikeLike(filename_logmag, distance_m,header=17):
    '''
    Calculates the gain from the S21 following the source:
    https://www.pasternack.com/t-calculator-fspl.aspx

    This assumes the two antennas have the same gain. 
    '''
    try:
        freqs_Hz, logmag_vals = ff.readerFieldFox(filename_logmag,header=header)
        fspl = numpy.abs(logmag_vals)
        p1 = 20*numpy.log10(distance_m)
        p2 = 20*numpy.log10(freqs_Hz)
        p3 = 20*numpy.log10(4*numpy.pi/(299792458))
        gain = (p1 + p2 + p3 - fspl)/2.0
        #import pdb; pdb.set_trace()

        return freqs_Hz, gain,
    except Exception as e:
            print(e)

def plotGainLikeLike(distance_m,datapath):
    '''
    This will plot the gain assuming that both ends of the s21 measurement are similar antennas.
    '''
    plt.close('all')
    infiles = numpy.array(glob.glob(os.path.join(datapath , '*txtx*s21*logmag*.csv')))

    plt.rcParams['axes.prop_cycle'].by_key()['color']

    all_linestyles = numpy.array(list(lines.lineStyles.keys()))
    linestyles = numpy.array(['-', '--',':', '-.', ' ', ''], dtype='<U4')#all_linestyles[~numpy.isin(all_linestyles,['None'])]

    #PLOT PREPPING
    fontsize=20
    leg_fontsize=14
    alpha = 0.8
    thickness = 4
    #PLOT Gain
    gain_plot = plt.figure()
    gain_ax = plt.subplot(1,1,1)
    plt.ylabel('dBi')
    plt.xlabel('MHz')
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='k', linestyle='-')
    plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    for infile in infiles:
        try:
            label = os.path.split(infile)[-1].replace('.csv','').replace('_phase','').replace('_logmag','').replace('HOURGLASS','HOURGLASS ').replace('_',' ').title().replace('Pf',' pF').replace('Nh',' nH').replace('Ohm',' $\\Omega$').replace('Noshield','No Shield')
            print(label)
            print()
            label = 'Hpol Gain with 2.7 pF Shunt \nCapacitors and 100 nH Inductors'
            freqs, gain = gainFromS21LikeLike(infile,distance_m,header=17)
            
            plot_cut_ul = 100            
            plot_cut = freqs/1e6 < plot_cut_ul

            linestyle = '-'

            gain_ax.plot(freqs[plot_cut]/1e6, gain[plot_cut],linewidth=thickness,label=label,alpha=alpha,linestyle=linestyle)#,color=color)


        except Exception as e:
            print(e)

    plt.axhline(2.0,linewidth=thickness,linestyle='--',c=(112/256,173/256,71/256),label='2 dBi Line')
    gain_ax.legend()
    gain_ax.set_xlim([0,plot_cut_ul])

if __name__ == '__main__':
    datapath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'data','beacon_s21_june6_2021')
    plotGainLikeLike(0.3048*50,datapath)