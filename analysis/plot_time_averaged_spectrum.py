#!/usr/bin/env python3
'''
This is meant to calculate the how many similar events within a run that each event has.
It will also save these values as a percentage of the run for easier comparison accross runs.
'''
import os
import sys
import h5py
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.interpret as interpret #Must be imported before matplotlib or else plots don't load.
import tools.info as info
from tools.fftmath import TimeDelayCalculator
from tools.data_handler import createFile

import matplotlib.pyplot as plt
import scipy.signal
import scipy.stats
import numpy
import sys
import math
import matplotlib
from scipy.fftpack import fft
import datetime as dt
import inspect
from ast import literal_eval
import astropy.units as apu
from astropy.coordinates import SkyCoord
import itertools
plt.ion()
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


import astropy
import astropy.time
import astropy.coordinates
import time
from datetime import datetime
from pytz import timezone,utc
from datetime import datetime
import matplotlib.dates as mdates


font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})

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



if __name__=="__main__":
    datapath = os.environ['BEACON_DATA']

    try:
        modulo_day = False #If true binning will occur with event times modulo 24 hours. 
        #Barcroft Station
        lat = 37.583342
        lon = -118.236484
        antenna_latlon = (lat,lon)

        runs = numpy.arange(3530,3630)#numpy.arange(3530,3571)


        run = runs[0]
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
        #Run 0 for setup 
        if filename is not None:
            with h5py.File(filename, 'a') as file:
                frequency_bin_edges_MHz = file['spectral_bin_average_series']['frequency_bin_edges_MHz'][...]
                frequency_bin_centers_MHz = (frequency_bin_edges_MHz[:-1] + frequency_bin_edges_MHz[1:]) / 2
                
                dsets = list(file['spectral_bin_average_series'].keys())
                antennas_of_interest = []
                for dset in dsets:
                    if 'ant' in dset:
                        antennas_of_interest.append(int(dset.replace('ant','')))
                antennas_of_interest = numpy.sort(antennas_of_interest)
                file.close()


        #WORKING ON PLOTTING THIS WITH NEW RUNS AND ADDING MODULO 1 DAY!!
        #Haven't ran time_averaged_spectrum on the new runs yet.  



        for antenna_index, antenna in enumerate(antennas_of_interest):
            data_set_name = 'ant%i'%antenna
            plt.figure()
            ax = plt.subplot(2,1,1)
            plt.title('Antenna %i'%antenna)
            for freq_index, freq in enumerate(frequency_bin_centers_MHz):
                if numpy.logical_and(freq > 20, freq < 100):    
                    vals_y = []
                    vals_x = []
                    for run in runs:
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
                            with h5py.File(filename, 'a') as file:
                                vals_x.append(file['spectral_bin_average_series']['time_window_centers'][...])
                                vals_y.append(file['spectral_bin_average_series'][data_set_name][...][:,freq_index])

                                file.close()

                        file.close()
                    
                    vals_x = numpy.concatenate(vals_x)
                    vals_y = numpy.concatenate(vals_y)
                    
                    if modulo_day == True:
                        time_window = 3*60*60 #seconds
                        x_bin_edges = numpy.arange(0,24*60*60 + time_window, time_window)#numpy.arange(vals_x[0],vals_x[-1]+time_window,time_window)
                        x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2                        
                        utc_timestamps, sagA_alt, sagA_az, sun_alt, sun_az = getSagCoords(vals_x[0], vals_x[0] + 24*60*60, antenna_latlon, n_points=len(vals_x)*4, plot=False)
                    else:
                        time_window = 3*60*60 #seconds
                        x_bin_edges = numpy.arange(vals_x[0],vals_x[-1]+time_window,time_window)
                        x_bin_centers = (x_bin_edges[:-1] + x_bin_edges[1:]) / 2
                        utc_timestamps, sagA_alt, sagA_az, sun_alt, sun_az = getSagCoords(vals_x[0], vals_x[-1], antenna_latlon, n_points=len(vals_x)*4, plot=False)
                    rebinned_y = numpy.zeros_like(x_bin_centers)
                    for bin_index in range(len(x_bin_centers)):
                        if modulo_day == True:
                            _vals_x = (vals_x - vals_x[0])%(24*60*60)
                            rebin_cut = numpy.logical_and(_vals_x >= x_bin_edges[bin_index], _vals_x < x_bin_edges[bin_index + 1])
                            rebinned_y[bin_index] = numpy.mean(vals_y[rebin_cut])
                        else:
                            rebin_cut = numpy.logical_and(vals_x >= x_bin_edges[bin_index], vals_x < x_bin_edges[bin_index + 1])
                            rebinned_y[bin_index] = numpy.mean(vals_y[rebin_cut])

                    plt.plot((x_bin_centers - x_bin_centers[0])/3600.0,(rebinned_y - numpy.mean(rebinned_y))/numpy.mean(rebinned_y),label='%0.2f MHz'%freq,alpha=0.8)

            if modulo_day == True:
                plt.xlabel('Time Since First Event Modulo 1 Day (hours)')
            else:
                plt.xlabel('Time Since First Event (hours)')                
            plt.ylabel('Fractional Variation\n(Binned Linear FFT Vals - mean )/ mean')
            #plt.ylabel('Average FFT Value in Freq Bin\n(mean subtracted) (arb)')
            plt.grid(which='both', axis='both')
            ax.minorticks_on()
            ax.grid(b=True, which='major', color='k', linestyle='-')
            ax.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.legend()
            if modulo_day == True:
                plt.xlim(0,30)

            ax2 = plt.subplot(2,1,2,sharex=ax)
            lw = None #Linewidth None is default
            plt.plot(utc_timestamps/3600.0, sagA_alt, c='r',label='Sgr A*',linewidth=lw)
            plt.plot(utc_timestamps/3600.0, sun_alt, c='k',label='Sun',linewidth=lw)

            plt.xlabel('Time Since First Event (hours)',fontsize=16)
            plt.ylabel('Altitude (Degrees)',fontsize=16)
            plt.grid(which='both', axis='both')
            ax2.minorticks_on()
            ax2.grid(b=True, which='major', color='k', linestyle='-')
            ax2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
            plt.legend(loc='upper right',fontsize=16)
            #plt.ylim((min(sagA_alt.degree)-5,max(sagA_alt.degree)+5))
            plt.ylim((-90,90))
            
        else:
            print('filename is None, indicating empty tree.  Skipping run %i'%run)

    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        sys.exit(1)
    sys.exit(0)

