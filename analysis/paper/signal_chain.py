#!/usr/bin/env python3
# coding: utf-8

import os
import matplotlib.pyplot as plt
import numpy
import numpy.fft as fft
import csv
import scipy.interpolate as interp
import pandas as pd
import scipy
import scipy.fftpack
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1 import make_axes_locatable


load_dir = os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data' , 'system_response', 'interpolated' )


# read channel mapping

channel_map = pd.read_excel(os.path.join(load_dir, 'BEACON Channel Mapping.xlsx'), 
                            names=['channel', 'first_stage_board_ind', 'second_stage_board_ind', 
                                  'antenna_mast', 'mast_position', 'pol', 'antchan'], engine='openpyxl')
#channel_map.head(10)


ch = 0
ind1 = channel_map[channel_map.channel==ch].first_stage_board_ind.values[0]
ind2 = channel_map[channel_map.channel==ch].second_stage_board_ind.values[0]
antchan = channel_map[channel_map.channel==ch].antchan.values[0]
#print(ch, ind1, ind2, antchan)


# read in the first stage files

first_stage = pd.read_csv(os.path.join(load_dir, "first_stage_%d_interpolate.csv"%ind1), usecols=[1,2,3,4,5,6])
first_stage['gain_complex'] = first_stage.real + 1j* first_stage.imag
#first_stage.head(10)


# read in the cable files
cables = pd.read_csv(os.path.join(load_dir, "cables.csv"), usecols=[1,2,3,4,5,6,7,8,9])
#cables.head(10)


# read in the second stage files
second_stage = pd.read_csv(os.path.join(load_dir, "second_stage_%d_interpolate.csv"%ind2), usecols=[1,2,3,4,5,6])
second_stage['gain_complex'] = second_stage.real + 1j* second_stage.imag
#second_stage.head(10)


def add_cable_loss(component, cable, channel):
    
    # copy the signal chain
    with_cable_loss = component.copy()
    
    # loss due to cables
    loss_lin = pow(10., -cable['%d'%channel]/20.)
    
    # reset all the derived parameters
    with_cable_loss['real'] = with_cable_loss.gain_complex.values.real
    with_cable_loss['imag'] = with_cable_loss.gain_complex.values.imag
    with_cable_loss['mag_dB'] = 20.*numpy.log10(abs(with_cable_loss.gain_complex))
    with_cable_loss['phase_rad'] = numpy.arctan2(with_cable_loss.gain_complex.values.imag, with_cable_loss.gain_complex.values.real)
    with_cable_loss['unwrap_phase_rad'] = numpy.unwrap(with_cable_loss.phase_rad)
    
    return with_cable_loss


def convolve(comp1, comp2):
    # convolve two responses together, assuming that they in the same frequency range and sampled the same

    # copy the first one
    components = comp1.copy()
     
    # multiply them together
    components['gain_complex'] *= comp2.gain_complex
    
    # reset all the derived parameters
    components['real'] = components.gain_complex.values.real
    components['imag'] = components.gain_complex.values.imag
    components['mag_dB'] = 20.*numpy.log10(abs(components.gain_complex))
    components['phase_rad'] = numpy.arctan2(components.gain_complex.values.imag, components.gain_complex.values.real)
    components['unwrap_phase_rad'] = numpy.unwrap(components.phase_rad)
    
    return components


signal_chain = first_stage.copy()
signal_chain = add_cable_loss(signal_chain, cables, ch)
signal_chain = convolve(signal_chain, second_stage)


if __name__ == '__main__':

    major_fontsize = 24
    minor_fontsize = 18
    fig, ax = plt.subplots(figsize=(14,10))

    for ch in range(8):
        signal_chain = pd.read_csv(os.path.join(load_dir, 'signalchain_ch%d_interpolate_fd.csv'%ch))
        ax.plot(signal_chain.freq_Hz/1e6, signal_chain.mag_dB, linewidth=2, label='%d'%ch)

        #axins.plot(signal_chain.freq_Hz/1e6, signal_chain.mag_dB)

    #axins = zoomed_inset_axes(ax, 3, loc=1)

    #axins.set_xticks([-0.1, 0, 0.1])
    #axins.set_yticks([-0.1, 0, 0.1])
    #axins.tick_params(axis='x', colors='white')
    #axins.tick_params(axis='y', colors='white')

    axins = inset_axes(ax, 7,2.5, loc=1)
    axins.set_xlim(30,80)
    axins.set_ylim(77,83)
    axins.grid()
    axins.tick_params(labelsize=minor_fontsize)
    for ch in range(8):
        signal_chain = pd.read_csv(os.path.join(load_dir, 'signalchain_ch%d_interpolate_fd.csv'%ch))
        axins.plot(signal_chain.freq_Hz/1e6, signal_chain.mag_dB)
        
    ax.set_ylim(-10,120)
    ax.set_xlim(min(signal_chain.freq_Hz/1e6), 240)
    ax.grid()
    ax.set_ylabel("Gain (dB)", fontsize=major_fontsize)
    ax.set_xlabel("Frequency (MHz)", fontsize=major_fontsize)
    ax.set_yticks(numpy.arange(-10,120,10))
    ax.set_xticks(numpy.arange(min(signal_chain.freq_Hz/1e6), max(signal_chain.freq_Hz/1e6),20))
    ax.tick_params(axis='both', labelsize=minor_fontsize)
    ax.legend(title="Channel", title_fontsize=major_fontsize, fontsize=major_fontsize, loc=4)



    mark_inset(ax, axins, loc1=2, loc2=4, lw=1.5, alpha=0.5)

    plt.tight_layout()
    # plt.savefig("/home/avz5228/Pictures/signal_chain.pdf")




