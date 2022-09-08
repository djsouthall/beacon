#!/usr/bin/env python3
'''
Sent from Kaeli to create spectra comparison plot.
'''
import os
import matplotlib.pyplot as plt
from scipy import signal
import numpy
import sys
import math
from scipy.fftpack import fft
import matplotlib

plt.ion()

data_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/spectrum_comparison'

if __name__ == '__main__':
    plt.close('all')
    freqs = numpy.load(os.path.join(data_path, 'all_freqs.npy'))

    all_db_v_2021 = numpy.load(os.path.join(data_path, 'all_db_v_2021.npy')) #runs 6000 to 6050
    all_db_v_2019 = numpy.load(os.path.join(data_path, 'all_db_v_2019.npy')) #runs 1600 to 1650
    all_db_v_2018 = numpy.load(os.path.join(data_path, 'all_db_v_2018.npy')) #runs 200 to 250

    all_db_v = {2018:all_db_v_2018,2019:all_db_v_2019,2021:all_db_v_2021}

    all_db_h_2021 = numpy.load(os.path.join(data_path, 'all_db_h_2021.npy')) #runs 6000 to 6050
    all_db_h_2019 = numpy.load(os.path.join(data_path, 'all_db_h_2019.npy')) #runs 1600 to 1650
    all_db_h_2018 = numpy.load(os.path.join(data_path, 'all_db_h_2018.npy')) #runs 200 to 250

    all_db_h = {2018:all_db_h_2018,2019:all_db_h_2019,2021:all_db_h_2021}
    colors={2018:'black',2019:'red',2021:'dodgerblue'}


    # matplotlib.rcParams['figure.figsize'] = [10, 20]
    # params = {'axes.labelsize': 30,'axes.titlesize':30, 'font.size': 30, 'legend.fontsize': 20, 'xtick.labelsize': 20, 'ytick.labelsize': 20,'font.family':'serif'}
    # matplotlib.rcParams.update(params)

    plt.figure(figsize=(8,6))
    major_fontsize = 18
    minor_fontsize = 14
    plt.rc('xtick',labelsize=minor_fontsize)
    plt.rc('ytick',labelsize=minor_fontsize)

    for year in [2018,2019,2021]:
        print(all_db_v[year])
        
        ax2 = plt.subplot(2,1,1)
        plt.plot(freqs[numpy.isfinite(all_db_h[year])]*1000,all_db_h[year][numpy.isfinite(all_db_h[year])],lw=2.5,color=colors[year],label=str(year))
        plt.grid()
        plt.xlim([0,150])

        ax3 = plt.subplot(2,1,2, sharex=ax2, sharey=ax2)
        plt.plot(freqs[numpy.isfinite(all_db_v[year])]*1000,all_db_v[year][numpy.isfinite(all_db_v[year])],lw=2.5,color=colors[year],label=str(year))
        plt.grid()
        plt.xlim([0,150])

    ax2.set_xlim(0,150) #All others will follow
    ax2.set_xticks([0,30,80,150])
    ax2.set_ylabel('PSD (dB, arb)', fontsize=major_fontsize)

    align = 'right'

    if align == 'right':

        ax2.text(1 - 0.025, 0.95, 'HPol',
            size=major_fontsize,
            transform=ax2.transAxes, va='top', ha='right',
            bbox=dict(boxstyle="square", fc="w"))


        ax3.set_xlabel('Frequency (MHz)', fontsize=major_fontsize)
        ax3.set_ylabel('PSD (dB, arb)', fontsize=major_fontsize)
        ax3.text(1 - 0.025, 0.95, 'VPol',
            size=major_fontsize,
            transform=ax3.transAxes, va='top', ha='right',
            bbox=dict(boxstyle="square", fc="w"))
        ax3.legend(fontsize=minor_fontsize, loc='center right')
    else:
        ax2.text(0.025, 0.95, 'HPol',
            size=major_fontsize,
            transform=ax2.transAxes, va='top', ha='left',
            bbox=dict(boxstyle="square", fc="w"))


        ax3.set_xlabel('Frequency (MHz)', fontsize=major_fontsize)
        ax3.set_ylabel('PSD (dB, arb)', fontsize=major_fontsize)
        ax3.text(0.025, 0.95, 'VPol',
            size=major_fontsize,
            transform=ax3.transAxes, va='top', ha='left',
            bbox=dict(boxstyle="square", fc="w"))
        ax3.legend(fontsize=minor_fontsize, loc='center left')

    plt.tight_layout()
    plt.savefig('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/figures/spectra_comparison_v2.pdf')
