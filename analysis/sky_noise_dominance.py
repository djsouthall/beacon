'''
This is specifically made to track rms over time for the new 2 antennas (ch 4 and 5) install in Oct 2020.
'''
import numpy
import sys
import os
import inspect
import h5py
import time
from datetime import datetime
from pytz import timezone,utc
from datetime import datetime
from pprint import pprint
import glob
import scipy
import scipy.signal

sys.path.append(os.environ['BEACON_INSTALL_DIR']) 
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile
from tools.interpret import getReaderDict, getHeaderDict, getStatusDict #Must be imported before matplotlib or else plots don't load.
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.animation
import astropy
import astropy.time
import astropy.coordinates

plt.ion()

datapath = os.environ['BEACON_DATA']

def getSagCoords(run_start_time_utc_timestamp, run_stop_time_utc_timestamp, antenna_latlon, n_points=1000, plot=False):
    '''
    Sagitarius A is functionally the same as centre of galaxy.
    Parameters
    ----------
    run_start_time_utc_timestamp : float
        The utc timestamp corresponding to the start of the run.
    run_stop_time_utc_timestamp : float
        The utc timestamp corresponding to the stop of the run.
    n_points : int
        The number of points between the start and stop time to calculate the altitude. 
    plot : bool
        Enables plotting if True.
    '''
    try:
        lat = antenna_latlon[0]
        lon = antenna_latlon[1]
        antenna_location = astropy.coordinates.EarthLocation(lat=lat,lon=lon) #Barcroft Station
        
        #Location of Sagitarius A in Right Ascension and Declination
        ra = 266.427 * astropy.units.deg # Could also use Angle
        dec = -29.007778 * astropy.units.deg  # Astropy Quantity
        sagA = astropy.coordinates.SkyCoord(ra, dec, frame='icrs') #ICRS=Internation Celestial Reference System
        
        #Setting up astropy time objects
        time_window_utc_timestamp = numpy.linspace(0,(run_stop_time_utc_timestamp-run_start_time_utc_timestamp),n_points)
        
        start_time = astropy.time.Time(run_start_time_utc_timestamp,format='unix')
        stop_time = astropy.time.Time(run_stop_time_utc_timestamp,format='unix')
        time_window_object = astropy.time.Time(time_window_utc_timestamp,format='unix')


        time_window = (time_window_utc_timestamp/3600.0)*astropy.units.hour

        #Setting up frame for Sagitarius A
        frame = astropy.coordinates.AltAz(obstime=start_time+time_window,location=antenna_location)
        sagAaltazs = sagA.transform_to(frame)
        sun_loc = astropy.coordinates.get_sun(time_window_object).transform_to(frame)

        if plot == True:
            fig = plt.figure()
            fig.canvas.set_window_title('Sgr A* Alt')
            ax = plt.gca()
            #Make Landscape
            ax.axhspan(0, 90, alpha=0.2,label='Sky', color = 'blue')
            ax.axhspan(-90, 0, alpha=0.2,label='Ground', color = 'green')
            plt.plot(time_window, sagAaltazs.alt,label='Sgr A*')
            plt.plot(time_window, sun_loc.alt,label='Sun')

            #plt.ylim(1, 4)
            plt.xlabel('Hours from Start of Run',fontsize=16)
            plt.ylabel('Altitutude (Degrees)',fontsize=16)
            plt.ylim([-90,90])
            plt.legend(fontsize=16)

            fig = plt.figure()
            fig.canvas.set_window_title('Sgr A* Position')
            ax = plt.gca()
            #Make Landscape
            ax.axhspan(0, 90, alpha=0.2,label='Sky', color = 'blue')
            ax.axhspan(-90, 0, alpha=0.2,label='Ground', color = 'green')
            plt.scatter(sagAaltazs.az, sagAaltazs.alt,label='Sgr A*')
            plt.scatter(sun_loc.az, sun_loc.alt,label='Sun')

            #plt.ylim(1, 4)
            plt.xlabel('Azimuth (Degrees)',fontsize=16)
            plt.ylabel('Altitutude (Degrees)',fontsize=16)
            plt.ylim([-90,90])
            plt.legend(fontsize=16)

        return time_window_utc_timestamp, sagAaltazs.alt, sagAaltazs.az, sun_loc.alt, sun_loc.az
    except Exception as e:
        print('Error in getSagCoords().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)



if __name__ == '__main__':
    #Parameters
    sag_plot = True
    df_multiplier = 2
    width = (0,4)
    prominence = 5
    lw = None #Linewidth None is default

    time_bin_width = 60*60 #Seconds per calculation
    '''
    freq_ROI= [[0,200]]
    ignore_range = []
    ignore_peaks = False
    ignore_min_max = False
    plot_total_averaged = True
    plot_stacked_spectra = True
    plot_comparison = True
    plot_comparison_mins = True
    plot_animated = True
    plot_data_collection_check = False
    min_method = 3
    '''

    #Barcroft Station
    lat = 37.583342
    lon = -118.236484
    antenna_latlon = (lat,lon)

    timezone = timezone('US/Pacific')

    ts = []
    powers = []

    runs = numpy.arange(3530,3555)

    #determine_run_bin_edges:
    run_bin_edges = numpy.zeros((len(runs),2))
    for run_index, run in enumerate(runs):
        reader = Reader(datapath,run)
        filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.
        if filename is not None:
            with h5py.File(filename, 'r') as file:
                event_times = file['calibrated_trigtime'][...]
                run_bin_edges[run_index][0] = file['calibrated_trigtime'][0]
                run_bin_edges[run_index][1] = file['calibrated_trigtime'][-1]

    #determine time binnings:
    initial_time = numpy.min(run_bin_edges)
    final_time = numpy.max(run_bin_edges)
    time_bin_edges = numpy.arange(initial_time,final_time+1,time_bin_width)
    time_bin_centers = (time_bin_edges[:-1] + time_bin_edges[1:]) / 2
    rms_values_h = numpy.zeros(len(time_bin_edges)-1)
    rms_values_v = numpy.zeros(len(time_bin_edges)-1)

    utc_timestamps, sagA_alt, sagA_az, sun_alt, sun_az = getSagCoords(initial_time, final_time, antenna_latlon, n_points=len(time_bin_centers)*4, plot=True)

    
    for bin_index in range(len(time_bin_edges) - 1):
        lower_time_bound = time_bin_edges[bin_index]
        upper_time_bound = time_bin_edges[bin_index + 1]

        #determine runs in the time window
        ll_run = runs[run_bin_edges[:,1] >= lower_time_bound][0]
        ul_run = runs[run_bin_edges[:,0] <= upper_time_bound][-1]
        
        runs_in_window = runs[numpy.logical_and(runs>=ll_run, runs <= ul_run)]

        #Now loop over every run in this time window (only events within time window) and determine the time domain rms.

        rms_event_count = 0
        rms_sum_h = 0.0
        rms_sum_v = 0.0

        for run in runs_in_window:
            #Prepare to load correlation values.
            reader = Reader(datapath,run)
            try:
                print(reader.status())
            except Exception as e:
                print('Status Tree not present.  Returning Error.')
                print('\nError in %s'%inspect.stack()[0][3])
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)
                sys.exit(1)
            filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.

            if filename is not None:
                with h5py.File(filename, 'r') as file:
                    event_times = file['calibrated_trigtime'][...]
                    time_cut = numpy.logical_and(event_times >= lower_time_bound, event_times <= upper_time_bound)
                    trigger_type_cut = file['trigger_type'][...] == 1 #3=gps
                    cut = numpy.logical_and(time_cut,trigger_type_cut)
                    
                    rms_event_count += sum(cut)
                    rms_sum_h += numpy.sum(file['std'][...][:,4][cut]) #hard coded antennas for new h and v
                    rms_sum_v += numpy.sum(file['std'][...][:,5][cut])
                    std = file['std'][...]

            
        rms_values_h[bin_index] = rms_sum_h/rms_event_count #Average after the fact because might run over multiple runs for same time window. 
        rms_values_v[bin_index] = rms_sum_v/rms_event_count #Average after the fact because might run over multiple runs for same time window. 
        

    # plt.figure()
    # plt.hist(std[:,5])

    plt.figure()
    ax1 = plt.subplot(3,1,1)
    plt.plot((time_bin_centers - initial_time)/3600.0,rms_values_h,label='HPol RMS')
    plt.xlabel('Time Since First Event (hours)')
    plt.ylabel('Average STD (adu)')
    plt.grid(which='both', axis='both')
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.legend()

    ax = plt.subplot(3,1,2,sharex=ax1)
    plt.plot((time_bin_centers - initial_time)/3600.0,rms_values_v,label='VPol RMS')
    plt.xlabel('Time Since First Event (hours)')
    plt.ylabel('Average STD (adu)')
    plt.grid(which='both', axis='both')
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.legend()

    ax = plt.subplot(3,1,3,sharex=ax1)
    plt.plot(utc_timestamps/3600.0, sagA_alt, c='r',label='Sgr A*',linewidth=lw)
    plt.plot(utc_timestamps/3600.0, sun_alt, c='k',label='Sun',linewidth=lw)

    plt.xlabel('Time Since First Event (hours)',fontsize=16)
    plt.ylabel('Altitude (Degrees)',fontsize=16)
    plt.grid(which='both', axis='both')
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.legend(loc='upper right',fontsize=16)
    #plt.ylim((min(sagA_alt.degree)-5,max(sagA_alt.degree)+5))
    plt.ylim((-90,90))
    

    plt.figure()
    plt.subplot(3,1,1, sharex=ax1)
    plt.plot((time_bin_centers - initial_time)/3600.0,rms_values_v - rms_values_h,label='VPol RMS - HPol RMS')
    plt.xlabel('Time Since First Event (hours)')
    plt.ylabel('Average STD (adu)')
    plt.grid(which='both', axis='both')
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.legend()

    plt.subplot(3,1,2, sharex=ax1)
    plt.plot((time_bin_centers - initial_time)/3600.0,rms_values_v/rms_values_h,label='VPol RMS/HPol RMS')
    plt.xlabel('Time Since First Event (hours)')
    plt.ylabel('Average STD (adu)')
    plt.grid(which='both', axis='both')
    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-')
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.legend()


    ax = plt.subplot(3,1,3,sharex=ax1)
    plt.plot(utc_timestamps/3600.0, sagA_alt, c='r',label='Sgr A*',linewidth=lw)
    plt.plot(utc_timestamps/3600.0, sun_alt, c='k',label='Sun',linewidth=lw)

    plt.xlabel('Time Since First Event (hours)',fontsize=16)
    plt.ylabel('Altitude (Degrees)',fontsize=16)
    plt.grid(which='both', axis='both')
    ax.minorticks_on()
    ax.grid(b=True, which='major', color='k', linestyle='-')
    ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    plt.legend(loc='upper right',fontsize=16)
    #plt.ylim((min(sagA_alt.degree)-5,max(sagA_alt.degree)+5))
    plt.ylim((-90,90))
    