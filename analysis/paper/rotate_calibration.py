'''
Given the best measured time delays for each of the 6 pulsing sites from the July 2021 BEACON pulsing campaign, this
will attempt to minimize the antenna positions and cable delays.  
'''

#General Imports
import numpy
import itertools
import os
import sys
import csv
import scipy
import scipy.interpolate
import pymap3d as pm
from iminuit import Minuit
import inspect
import h5py
import time

#Personal Imports
from beaconroot.examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
from beacon.tools.data_handler import createFile, getTimes
import beacon.tools.station as bc
import beacon.tools.info as info
import beacon.tools.get_plane_tracks as pt
from beacon.tools.fftmath import TimeDelayCalculator
from beacon.tools.data_slicer import dataSlicerSingleRun
from beacon.tools.correlator import Correlator
import beacon.tools.config_reader as bcr


#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm

from beacon.analysis.aug2021.parse_pulsing_runs import PulserInfo, predictAlignment


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
raw_datapath = os.environ['BEACON_DATA']
#processed_datapath = os.path.join(os.environ['BEACON_PROCESSED_DATA'],'backup_pre_all_map_run_12-5-2021')
processed_datapath = os.environ['BEACON_PROCESSED_DATA']
print('SETTING processed_datapath TO: ', processed_datapath)

if True:
    crit_freq_low_pass_MHz = 85
    low_pass_filter_order = 6

    crit_freq_high_pass_MHz = 25
    high_pass_filter_order = 8

    sine_subtract = True
    sine_subtract_min_freq_GHz = 0.00
    sine_subtract_max_freq_GHz = 0.25
    sine_subtract_percent = 0.03
else:
    crit_freq_low_pass_MHz = 80
    low_pass_filter_order = 14

    crit_freq_high_pass_MHz = 20
    high_pass_filter_order = 4

    sine_subtract = False
    sine_subtract_min_freq_GHz = 0.02
    sine_subtract_max_freq_GHz = 0.15
    sine_subtract_percent = 0.01

plot_filter=False

apply_phase_response=True

mollweide = False

if mollweide == True:
    map_resolution_theta = 0.5 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.25 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    range_phi_deg = (min_phi, max_phi)
    range_theta_deg = (min_theta, max_theta)
else:        
    map_resolution_theta = 0.25 #degrees
    min_theta   = 0
    max_theta   = 120
    n_theta = numpy.ceil((max_theta - min_theta)/map_resolution_theta).astype(int)

    map_resolution_phi = 0.25 #degrees
    min_phi     = -180
    max_phi     = 180
    n_phi = numpy.ceil((max_phi - min_phi)/map_resolution_phi).astype(int)

    range_phi_deg = (min_phi, max_phi)
    range_theta_deg = (min_theta, max_theta)

upsample = 2**16
max_method = 0

apply_filter = True

min_event_cut = 0

deploy_index = info.returnDefaultDeploy()
datapath = os.environ['BEACON_DATA']
map_source_distance_m = info.returnDefaultSourceDistance()
waveform_index_range = info.returnDefaultWaveformIndexRange()


if __name__ == '__main__':
    plt.close('all')

    if False:
        theta_shift_hpol = 1.0847 #Shifts this many degrees UP (from where it would be pointing normally.)
        azimuth_shift_hpol = 0.2823 #Shifts this many degrees North (from where it would be pointing normally.)
        theta_shift_vpol = 2.0957
        azimuth_shift_vpol = 0.1448
        initial_calibration = str(info.returnDefaultDeploy())
    else:

        theta_shift_hpol = 0.1497 #Shifts this many degrees UP (from where it would be pointing normally.)
        azimuth_shift_hpol = 0.0694 #Shifts this many degrees North (from where it would be pointing normally.)
        theta_shift_vpol = 0.2810
        azimuth_shift_vpol = 0.2280

        initial_calibration = '/home/dsouthall/Projects/Beacon/beacon/config/september_2021_minimized_calibration_rotated_1663962064.json'



    #This code is intended to save the output configuration produced by this script. 
    initial_origin, initial_antennas_physical, initial_antennas_phase_hpol, initial_antennas_phase_vpol, initial_cable_delays, initial_description = bcr.configReader(initial_calibration,return_description=True)

    output_origin = initial_origin
    output_antennas_physical = initial_antennas_physical

    run = 5911
    eventid = 73399
    reader = Reader(datapath,run)
    cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, deploy_index=deploy_index, map_source_distance_m=map_source_distance_m)
    rotated_antennas_phase_hpol = {0:numpy.zeros(3), 1:numpy.zeros(3), 2:numpy.zeros(3), 3:numpy.zeros(3)}
    rotated_antennas_phase_vpol = {0:numpy.zeros(3), 1:numpy.zeros(3), 2:numpy.zeros(3), 3:numpy.zeros(3)}

    for antenna_index in range(4):
        if not numpy.all(initial_antennas_phase_hpol[antenna_index] == 0):
            a = numpy.copy(initial_antennas_phase_hpol[antenna_index])
            b = numpy.array([0,-1,0])

            dtheta_rad = numpy.deg2rad(theta_shift_hpol)
            rotated_antennas_phase_hpol[antenna_index] = cor.rotateAaboutBbyTheta(a, b, dtheta_rad, normalize_output=False)
            a = numpy.copy(rotated_antennas_phase_hpol[antenna_index])
            b = numpy.array([0,0,1])

            dtheta_rad = numpy.deg2rad(azimuth_shift_hpol)
            rotated_antennas_phase_hpol[antenna_index] = cor.rotateAaboutBbyTheta(a, b, dtheta_rad, normalize_output=False)
        if not numpy.all(initial_antennas_phase_vpol[antenna_index] == 0):
            a = numpy.copy(initial_antennas_phase_vpol[antenna_index])
            b = numpy.array([0,-1,0])

            dtheta_rad = numpy.deg2rad(theta_shift_vpol)
            rotated_antennas_phase_vpol[antenna_index] = cor.rotateAaboutBbyTheta(a, b, dtheta_rad, normalize_output=False)

            a = numpy.copy(rotated_antennas_phase_vpol[antenna_index])
            b = numpy.array([0,0,1])

            dtheta_rad = numpy.deg2rad(azimuth_shift_vpol)
            rotated_antennas_phase_vpol[antenna_index] = cor.rotateAaboutBbyTheta(a, b, dtheta_rad, normalize_output=False)


    rotated_cor = Correlator(reader,  upsample=upsample, n_phi=n_phi, range_phi_deg=range_phi_deg, n_theta=n_theta, range_theta_deg=range_theta_deg, waveform_index_range=waveform_index_range,crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, deploy_index=deploy_index, map_source_distance_m=map_source_distance_m)
    rotated_cor.overwriteAntennaLocations(initial_antennas_physical[0],initial_antennas_physical[1],initial_antennas_physical[2],initial_antennas_physical[3],rotated_antennas_phase_hpol[0],rotated_antennas_phase_hpol[1],rotated_antennas_phase_hpol[2],rotated_antennas_phase_hpol[3],rotated_antennas_phase_vpol[0],rotated_antennas_phase_vpol[1],rotated_antennas_phase_vpol[2],rotated_antennas_phase_vpol[3], verbose=True, suppress_time_delay_calculations=False)

    mean_corr_values_hpol, fig_hpol, map_ax_hpol = cor.map(eventid,'hpol', window_title='HPol Original')
    map_ax_hpol.set_title('HPol Original' + '\n' + map_ax_hpol.get_title())
    map_ax_hpol.set_ylim(-30,90)
    mean_corr_values_hpol_rotated, fig_hpol_rotated, map_ax_hpol_rotated = rotated_cor.map(eventid,'hpol', window_title='HPol Rotated')
    map_ax_hpol_rotated.set_title('HPol Rotated: %0.4f deg in theta and %0.4f deg in azimuth'%(theta_shift_hpol, azimuth_shift_hpol)  + '\n' + map_ax_hpol_rotated.get_title())
    map_ax_hpol_rotated.set_ylim(-30,90)


    mean_corr_values_vpol, fig_vpol, map_ax_vpol = cor.map(eventid,'vpol', window_title='vPol Original')
    map_ax_vpol.set_title('vPol Original' + '\n' + map_ax_vpol.get_title())
    map_ax_vpol.set_ylim(-30,90)
    mean_corr_values_vpol_rotated, fig_vpol_rotated, map_ax_vpol_rotated = rotated_cor.map(eventid,'vpol', window_title='vPol Rotated')
    map_ax_vpol_rotated.set_title('vPol Rotated: %0.4f deg in theta and %0.4f deg in azimuth'%(theta_shift_vpol, azimuth_shift_vpol)  + '\n' + map_ax_vpol_rotated.get_title())
    map_ax_vpol_rotated.set_ylim(-30,90)


    output_antennas_physical = initial_antennas_physical
    output_cable_delays = initial_cable_delays
    output_origin = initial_origin
    output_antennas_phase_hpol = rotated_antennas_phase_hpol
    output_antennas_phase_vpol = rotated_antennas_phase_vpol

    output_description = 'This calibration is a rotated form of %s.  Antennas 1,2,3 have been rotated to attempt shift pointing reconstruct by HPOL: %0.4f deg in theta and %0.4f deg in azimuth and VPOL: %0.4f deg in theta and %0.4f deg in azimuth'%(initial_calibration, theta_shift_hpol, azimuth_shift_hpol, theta_shift_vpol, azimuth_shift_vpol)
    json_path = initial_calibration.replace('.json','_rotated_%i.json'%time.time())
    
    if False:
        bcr.configWriter(json_path, output_origin, output_antennas_physical, output_antennas_phase_hpol, output_antennas_phase_vpol, output_cable_delays, description=output_description,update_latlonel=True,force_write=True, additional_text=None) #does not overwrite.
    else:

        for mode in ('phase_hpol', 'phase_vpol'):
            bcr.configSchematicPlotter(deploy_index, en_figsize=(16,16), eu_figsize=(16,9), mode=mode, mast_height=12*0.3048, antenna_scale_factor=5, mast_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            json_path = '/home/dsouthall/Projects/Beacon/beacon/config/september_2021_minimized_calibration_rotated_1663963370.json'
            bcr.configSchematicPlotter(json_path, en_figsize=(16,16), eu_figsize=(16,9), mode=mode, mast_height=12*0.3048, antenna_scale_factor=5, mast_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])