#!/usr/bin/env python3
'''
This is meant to calculate the how many similar events within a run that each event has.
It will also save these values as a percentage of the run for easier comparison accross runs.
'''
import os
import sys
import h5py
import inspect
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy
import matplotlib
import matplotlib.pyplot as plt
plt.ion()
from scipy.interpolate import interp1d

import astropy.units as apu
from astropy.coordinates import SkyCoord
import astropy
import astropy.time
import time
from datetime import datetime
from pytz import timezone,utc
import matplotlib.dates as mdates

font = {'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [10, 11]
matplotlib.rcParams.update({'font.size': 16})

def getSunAzEl(timestamp_array_s, lat=37.583342, lon=-118.236484, interp=True, interp_step_s=5*60, plot=False):
    '''
    Given an array of timestamps and the observation latitude and longitude, this will return the elevation and azimuth
    directions.  For a quicker runtime for large numbers of events, interpolation can be enabled with astropy being
    sampled at the expected step interval of interp_step_s.  These values will then be interpolated.  
    
    Azimuth will be given ranging from -180 to 180, with East as 0 and North as 90.
    '''
    try:
        if interp == False:
            time = astropy.time.Time(timestamp_array_s,format='unix')
        else:
            interpolation_times = numpy.arange(min(timestamp_array_s) - interp_step_s*5, max(timestamp_array_s) + interp_step_s*6, interp_step_s)
            if len(interpolation_times) > len(timestamp_array_s):
                print('The interpolated times would be larger in size than the given times, so skipping interpolation.')
                interp = False
                del interpolation_times
                time = astropy.time.Time(timestamp_array_s,format='unix')
            else:   
                time = astropy.time.Time(interpolation_times,format='unix')

        frame = astropy.coordinates.AltAz(obstime=time,location=astropy.coordinates.EarthLocation(lat=lat,lon=lon))
        sun_loc = astropy.coordinates.get_sun(time).transform_to(frame)
        el = sun_loc.alt.deg 

        if interp == True:
            az = 90 - numpy.rad2deg(numpy.unwrap(sun_loc.az.rad,discont=numpy.pi)) #discont is not the wrap point, it is what it looks for as a delta to determine where a wrap occured #discont changed to period in future versions of python #To convert from how AltAz has E as 90 and N as 0, when I want the opposite for ENU.
            az = interp1d(interpolation_times, az, kind='cubic')(timestamp_array_s)
            el = interp1d(interpolation_times, el, kind='cubic')(timestamp_array_s)
            az = az%360 #Now this is wrapped and interpolated
        else:
            #This is wrapped
            az = (90 - sun_loc.az.deg)%360 #To convert from how AltAz has E as 90 and N as 0, when I want the opposite for ENU.

        #Center from -180 to 180
        az[az > 180.0] -= 360.0

        if plot:
            plt.figure()
            plt.title('Sun Location\nLat: %f\nLon: %f'%(lat,lon))
            plt.subplot(2,1,1)
            plt.plot(time.value, az)
            plt.ylabel('Azimuth (E = 0 deg, N = 90 deg)')
            plt.subplot(2,1,2)
            plt.plot(time.value, el)
            plt.ylabel('Elevation Angle')

        return az, el
    except Exception as e:
        print('Error in getSagCoords().')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)



if __name__=="__main__":
    datapath = os.environ['BEACON_DATA']

    timestamp_array_s = 1639428003 + numpy.arange(3600*24) #One day
    #Barcroft Station
    lat = 37.583342
    lon = -118.236484

    if False:
        interp_step_s=5*60
        interpolation_times = numpy.arange(min(timestamp_array_s) - interp_step_s*5, max(timestamp_array_s) + interp_step_s*6, interp_step_s)
        time = astropy.time.Time(interpolation_times,format='unix')
        frame = astropy.coordinates.AltAz(obstime=time,location=astropy.coordinates.EarthLocation(lat=lat,lon=lon))
        sun_loc = astropy.coordinates.get_sun(time).transform_to(frame)
        el = sun_loc.alt.deg

    elif True:
        az1, el1 = getSunAzEl(timestamp_array_s, lat=lat, lon=lon, interp=False, interp_step_s=5*60, plot=False)
        az2, el2 = getSunAzEl(timestamp_array_s, lat=lat, lon=lon, interp=True, interp_step_s=5*60, plot=False)

        if numpy.all(az1 == az2) and numpy.all(el1 == el2):
            print('Interpolation worked perfectly.')
        elif numpy.allclose(az1, az2, atol=0.01) and numpy.allclose(el1, el2, atol=0.01):
            print('Interpolation matched within 0.01 deg with max difference of %f'%numpy.max(az1-az2))
        else:
            print('Interpolation did not recreate the sun location with high precision.')

        plt.figure()
        plt.title('Sun Location\nLat: %f\nLon: %f'%(lat,lon))
        plt.subplot(2,1,1)
        plt.plot(timestamp_array_s, az1,label='az1')
        plt.plot(timestamp_array_s, az2,label='az2')
        plt.legend()
        # plt.plot(timestamp_array_s, numpy.unwrap(az1,discont=360.0)) #discont changed to period in future versions of python
        # plt.plot(timestamp_array_s, numpy.unwrap(az2,discont=360.0)) #discont changed to period in future versions of python
        plt.ylabel('Azimuth (E = 0 deg, N = 90 deg)')
        plt.subplot(2,1,2)
        plt.plot(timestamp_array_s, el1)
        plt.plot(timestamp_array_s, el2)
        plt.ylabel('Elevation Angle')

    else:
        from timeit import Timer#import timeit

        t1 = Timer(lambda:getSunAzEl(timestamp_array_s, lat=lat, lon=lon, interp=False, interp_step_s=5*60, plot=False))
        print('Starting Interp=False')
        start = time.time()
        print(t1.timeit(number=5))
        finish = time.time()
        print('Finished after %0.2f minutes'%((finish-start)/60.0))

        t2 = Timer(lambda:getSunAzEl(timestamp_array_s, lat=lat, lon=lon, interp=True, interp_step_s=5*60, plot=False))
        print('Starting Interp=True')
        start = time.time()
        print(t2.timeit(number=5))
        finish = time.time()
        print('Finished after %0.2f minutes'%((finish-start)/60.0))

