#!/usr/bin/env python3
# coding: utf-8

# In[3]:

import sys
import os
import inspect
import h5py
import copy

import numpy
import scipy
import scipy.signal
import pymap3d as pm

#from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.sine_subtract_cache import sineSubtractedReader as Reader
from beacon.tools.data_handler import loadTriggerTypes
from beacon.tools.data_handler import createFile
from beacon.tools.fftmath import TemplateCompareTool, TimeDelayCalculator
from beacon.tools.fftmath import FFTPrepper
from beacon.tools.correlator import Correlator
from beacon.tools.data_slicer import dataSlicer
from beacon.tools.line_of_sight import circleSource
import beacon.tools.info as info
from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm, ticker
from matplotlib.patches import Rectangle
import time
import textwrap

import matplotlib.pyplot as plt

plt.ion()


raw_datapath = os.environ['BEACON_DATA']

plot = True



if False:
    #Filters used in dataslicer correlator
    crit_freq_low_pass_MHz = 85
    low_pass_filter_order = 6

    crit_freq_high_pass_MHz = 25
    high_pass_filter_order = 8

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03
    max_failed_iterations = 5

    apply_phase_response = True

    shorten_signals = False
    shorten_thresh = 0.7
    shorten_delay = 10.0
    shorten_length = 90.0

    upsample = 2**14 #Just upsample in this case, Reduced to 2**14 when the waveform length was reduced, to maintain same time precision with faster execution.
    notch_tv = True
    misc_notches = True
else:
    #Standard values used for waveform analysis
    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03
    max_failed_iterations = 5

    apply_phase_response = True

    shorten_signals = False
    shorten_thresh = 0.7
    shorten_delay = 10.0
    shorten_length = 90.0

    upsample = 2**17
    notch_tv = True
    misc_notches = True

# , notch_tv=notch_tv, misc_notches=misc_notches

align_method = None

hilbert=False
final_corr_length = upsample


mpl_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']




run = 5911
eventid = 73399
#Create initial tdc
reader = Reader(raw_datapath, run)
polarization_degs = []
for notch_tv in [False, True]:
    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=(None,None),plot_filters=False,apply_phase_response=apply_phase_response, notch_tv=notch_tv, misc_notches=misc_notches)
    if sine_subtract:
        tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=max_failed_iterations, verbose=False, plot=False)

    polarization_deg = tdc.calculatePolarizationFromTimeDelays(eventid, apply_filter=True, waveforms=None, plot=True, sine_subtract=True)
    polarization_degs.append(polarization_deg)
polarization_deg = tdc.calculatePolarizationFromTimeDelays(eventid, apply_filter=False, waveforms=None, plot=True, sine_subtract=True)
polarization_degs.append(polarization_deg)

cr_az = -33.55932236
cr_el = 18.60124969
    


if __name__ == '__main__':
    plt.close('all')
    fontsize = 18
    data_dir = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/cranberry_results'

    # In[4]:


    energy = numpy.round(numpy.arange(16.0, 19.05, 0.1),1)
    e_array = []
    acc = []
    antenna_thz_list_of_arrays = []
    antenna_az_list_of_arrays = []
    pol_list_of_arrays = []


    # In[6]:


    for en in energy:
        os.path.join(data_dir,f'hpol_{en}.npz')
        data = numpy.load(os.path.join(data_dir,f'hpol_{en}.npz'), allow_pickle=True)
        e_array.append(float(data['energy']))
        acc.append(float(data['acceptance']))
        antenna_thz_list_of_arrays.append(data['antenna_zenith'])
        antenna_az_list_of_arrays.append(data['antenna_azimuth'])
        pol_list_of_arrays.append(data['pol'])
        
    acc = numpy.array(acc)
    e_array = numpy.array(e_array)


    # In[7]:

    fig, ax = plt.subplots()

    # plot the acceptance
    plt.semilogy(e_array, acc, lw=3, label=r'$5\sigma$')
    plt.xticks(numpy.arange(16.0, 19.1, 0.5))
    plt.xlabel('$\log_{10}(E/\mathrm{eV})$', fontsize=fontsize)
    plt.ylabel('Acceptance (km$^2$ sr)', fontsize=fontsize)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim(16.0, 19)
    plt.tight_layout()
    plt.legend(title='Threshold', loc=4)
    plt.ylim(1e-4, 1e1)
    #plt.savefig('/home/avz5228/Pictures/acceptance.pdf')
    


    # In[8]:


    # CR flux
    def flux(energy_array_eV):
        # parameterization from Auger ICRC 2017
        J0 = 2.7513872e-19  # km^-2 sr^-1 yr^-1 eV^-1
        E_ankle = 5.08e18  # eV
        E_s = 3.9e19  # eV
        gam_1 = 3.293
        gam_2 = 2.53
        dgam = 2.5
        flux = numpy.ones(len(energy_array_eV))
        cut_1 = energy_array_eV <= E_ankle
        cut_2 = energy_array_eV > E_ankle

        flux[cut_1] = J0 * (energy_array_eV[cut_1] / E_ankle) ** -gam_1
        flux[cut_2] = J0 * (energy_array_eV[cut_2] / E_ankle) ** -gam_2 * (1. + (E_ankle / E_s) ** dgam) / (
        (1. + (energy_array_eV[cut_2] / E_s) ** dgam))
        return flux

    flx = flux(10 ** e_array)
    T_live = 24. / (365.25 * 24.)

    # plot the expected CR rate
    fig, ax = plt.subplots(figsize=(8,6))
    width = numpy.diff(e_array)[0]
    flux_y = flx * 10 ** e_array * acc * numpy.log(10.) * 0.1 * T_live
    plt.bar(e_array-width/2.0, flux_y, width = width)
    plt.plot(e_array, flux_y, drawstyle='steps', label=r'$5\sigma$', lw=2,c='k')
    plt.xlabel('$\log_{10}(E/\mathrm{eV})$', fontsize=fontsize)
    plt.ylabel('Predicted Cosmic Ray Events\nPer Day Per Bin', fontsize=fontsize)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlim(16.0, 19)
    y1, y2 = plt.ylim()
    plt.ylim(0., y2)

    ann = ax.annotate("Summed Rate:\n$\sim%0.1f$ Events Per Day"%sum(flux_y),
                  xy=(e_array[len(e_array)//2 - 2], 0.9*flux_y[len(e_array)//2 - 2]), xycoords='data',
                  xytext=(e_array[len(e_array)//2 + 7], flux_y[len(e_array)//2 - 3]), textcoords='data',
                  size=fontsize, va="center", ha="center",
                  bbox=dict(boxstyle="round", fc="w"),
                  arrowprops=dict(arrowstyle="-|>",
                                  connectionstyle="arc3,rad=-0.2",
                                  fc="w"),
                  )
    # plt.text(0.7, 0.6, r'Total: $\sim8.5$ events/day', horizontalalignment='center',
    #      verticalalignment = 'center', transform = ax.transAxes, fontsize=12)

    plt.legend(title='Threshold')
    plt.tight_layout()
    #plt.savefig('/home/avz5228/Pictures/event_rate.pdf')
    fig.savefig('./figures/event_rate.pdf',dpi=300)
    

    # In[9]:

    fig, ax = plt.subplots()
    # histogram of zenith angle per CR energy, weighted by CR flux
    wh = 0
    cc = 0
    for k in range(len(antenna_thz_list_of_arrays)):
        weight = flx[k] * 10 ** e_array[k] * acc[k] * numpy.log(10.) * 0.1 * T_live
        h, b = numpy.histogram(antenna_thz_list_of_arrays[k], bins=numpy.arange(-5., 91., 5.))
        wh += h * weight
        #plt.plot(b[1:], h * weight, lw=(cc + 1) * 0.2, label="%2.1f" % e_array[k])
        cc += 1
    # normalized histogram of zenith angle, all energies, weighted
    width = numpy.diff(b[1:])[0]
    plt.bar(b[1:]-width/2.0, wh / numpy.sum(wh), width = width)
    plt.plot(b[1:], 90.0 - wh / numpy.sum(wh), drawstyle='steps', lw=2, c='k')
    y1, y2 = plt.ylim()
    # plt.ylim(0., y2)
    plt.xlim(0., 90.)
    plt.xticks(numpy.arange(0., 91., 10.))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel(r'Elevation Angle $\theta_{Xmax}$ (deg)', fontsize=fontsize)
    plt.ylabel('Relative Frequency of Occurrence', fontsize=fontsize)

    #plt.legend()
    #plt.savefig('/home/avz5228/Pictures/antenna_zenith.pdf')
    fig.savefig('./figures/antenna_zenith.pdf',dpi=300)
    


    # In[11]:


    for k in range(len(antenna_az_list_of_arrays)):
        antenna_az_list_of_arrays[k] = numpy.array(antenna_az_list_of_arrays[k], dtype=float)
        antenna_az_list_of_arrays[k][numpy.where(antenna_az_list_of_arrays[k]>180)] = antenna_az_list_of_arrays[k][numpy.where(antenna_az_list_of_arrays[k]>180)] - 360


    # In[12]:


    # histogram of antenna azimuth angle per CR energy, weighted by CR flux
    wh = 0
    cc = 0
    for k in range(len(antenna_az_list_of_arrays)):
        weight = flx[k] * 10 ** e_array[k] * acc[k] * numpy.log(10.) * 0.1 * T_live
        h, b = numpy.histogram(antenna_az_list_of_arrays[k], bins=numpy.arange(-180., 181., 10.))
        wh += h * weight
        cc += 1
        
    fig, ax = plt.subplots()
    # normalized histogram of azimuth angle, all energies, weighted

    width = numpy.diff(b[1:])[0]
    plt.bar(b[1:]-width/2.0, wh / numpy.sum(wh), width = width)
    plt.plot(b[1:], wh / numpy.sum(wh), drawstyle='steps', lw=2, c='k')
    y1, y2 = plt.ylim()
    plt.ylim(0., y2)
    plt.xlim(-180., 180.)
    plt.xticks(numpy.arange(-180., 181., 45.))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel(r'Azimuth Angle $\phi_{Xmax}$ (deg)', fontsize=fontsize)
    plt.ylabel('Relative Frequency of Occurrence', fontsize=fontsize)

    #plt.vlines(-12.58, y1, y2, color='red')
    fig.savefig('./figures/antenna_azimuth.pdf',dpi=300)
    #plt.savefig('/home/avz5228/Pictures/antenna_azimuth.pdf')
    


    # In[14]:

    fig = plt.figure(figsize=(20,5))
    ax1 = plt.subplot(1,2,1)
    # histogram of polarization angle
    wh = 0
    cc = 0
    for k in range(len(pol_list_of_arrays)):
        weight = flx[k] * 10 ** e_array[k] * acc[k] * numpy.log(10.) * 0.1 * T_live
        h, b = numpy.histogram(pol_list_of_arrays[k], bins=numpy.arange(-2.5, 91., 2.5))
        wh += h * weight
        cc += 1
    

    width = numpy.diff(b[1:])[0]
    plt.bar(b[1:]-width/2.0, wh / numpy.sum(wh), width = width)
    plt.plot(b[1:], wh / numpy.sum(wh), drawstyle='steps', lw=2, c='k')

    y1, y2 = plt.ylim()
    plt.ylim(0., y2)
    plt.xticks(numpy.arange(0., 91., 10.))
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel(r'Polarization Angle (deg)', fontsize=fontsize)
    plt.ylabel('Relative Frequency of Occurrence', fontsize=fontsize)


    # plt.axvline(polarization_degs[0], c='r', label=textwrap.fill('5911-73399 with Symmetric Filtering',width=25))
    # plt.axvline(polarization_degs[1], c='g', label='5911-73399 with Asymettric Filtering')
    # plt.axvline(polarization_degs[2], c='m', label='5911-73399 with Only Sine Subtraction')
    plt.axvline(polarization_degs[0], c='r', label='5911-73399\nPolarization = %0.2f$^\circ$'%polarization_degs[0], lw=4)
    plt.xlim(10., 70.)
    plt.legend()

    #plt.savefig('/home/avz5228/Pictures/polarization_angle.pdf')
    

    # In[15]:


    # 2D histogram of CR shower cores that caused a trigger, weighted by the CR flux
    wh = 0
    cc = 0
    for k in range(len(antenna_az_list_of_arrays)):
        weight = flx[k] * 10 ** e_array[k] * acc[k] * numpy.log(10.) * 0.1 * T_live
        h, bx, by = numpy.histogram2d(numpy.array(antenna_az_list_of_arrays[k]), numpy.array(antenna_thz_list_of_arrays[k]), bins=(numpy.arange(-90,91,5), numpy.arange(-5, 91, 5)))
        wh += h.T * weight

    X, Y = numpy.meshgrid(bx, 90.0 - by)
    ax2 = plt.subplot(1,2,2)

    m = plt.pcolormesh(X, Y, wh / numpy.sum(wh), cmap='Greys')#
    cbar = plt.colorbar(m)
    cbar.set_label('Relative Frequency of Occurrence', labelpad=15, fontsize=fontsize)
    #plt.xlabel(r'Azimuth Angle $\phi_{Xmax}$ (deg)', fontsize=fontsize)
    #plt.ylabel(r'Zenith Angle $\theta_{Xmax}$ (deg)', fontsize=fontsize)

    plt.scatter(cr_az, cr_el,s=40, c='r', label='5911-73399')
    plt.legend(loc='upper right',fontsize=18)


    plt.xlabel(r'Event Azimuth Angle (deg)', fontsize=fontsize)
    plt.ylabel(r'Event Elevation Angle (deg)', fontsize=fontsize)
    plt.xticks(numpy.arange(-90., 91., 30.))
    y1, y2 = plt.ylim()
    plt.ylim(0,60)
    plt.xlim(-60,60)
    #plt.title('Location of Ground Intersection Point \n for Triggered Showers')
    # plt.gca().invert_yaxis()
    plt.tight_layout()


    fig.savefig('./figures/cr_sim_properties.pdf',dpi=300)
    

