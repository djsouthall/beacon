'''
At the time of creation this script was generated, tested, and used with a build on anaconda that is incompatible with
ALL other code in this analysis package.  This was instead built using a clean conda build for the traffic class.

This also requires sign-in credentials to https://opensky-network.org/ to access the data. 

The conda environment was built using mamba (recommended), which can be downlaoded using:

# Install mamba because anaconda sucks at downloading and gets stuck
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
bash Mambaforge-$(uname)-$(uname -m).sh



# Recommended options if not set up yet
conda config --set channel_priority strict
conda config --remove channels conda-forge
# the conda-forge line might need to be removed from rhw conda config file manually
# The below line might not be necessary and is commented out
# conda config --add channels conda-forge

# Installation
conda create -n traffic -c conda-forge python=3.7 traffic
conda activate traffic
'''
import sys
import os
import pandas as pd
import csv
import numpy
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
plt.ion()

import traffic
from traffic.data import opensky
from traffic.drawing import countries, rivers

import cartopy.crs as ccrs
import cartopy.feature as cfeature



origin_longitude = -118.23761867
origin_latitude = 37.589339

airspace_bbox = ( origin_longitude - 2, origin_latitude - 3, origin_longitude + 5 , origin_latitude + 3 )#[West, South, East, North]

def prepareMapAxisBEACON():
    '''
    '''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.set_extent([airspace_bbox[0], airspace_bbox[2], airspace_bbox[1], airspace_bbox[3]], crs=ccrs.PlateCarree())
    # ax.set_extent([-130, -70, 37, 44], crs=ccrs.PlateCarree())
    ax.stock_img()
    states_provinces = cfeature.NaturalEarthFeature(category="cultural", name="admin_1_states_provinces_lines", scale="50m", facecolor="none")
    # SOURCE = "Natural Earth"                                           
    # LICENSE = "public domain"                                          
                                                                       
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    # ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(states_provinces, edgecolor="gray")

    plt.scatter(origin_longitude, origin_latitude, c="r")

    return fig, ax


def testPlotPlane(flight_icao24="a0a8da", flight_date_y_m_d="2021-09-22", flight=None, ax=None, fig=None):
    '''
    This contains the code used to plot an airplane we might see.

    If flight is given then the given strings for flight_* are ignored.
    '''
    if flight is None:
        flight = opensky.history(flight_date_y_m_d, icao24=flight_icao24,return_flight=True)
    
    if fig is None and ax is None:
        fig, ax = prepareMapAxisBEACON()

    flight.plot(ax)

    return fig, ax

'''
Need to look into downloading bulk data and storing it, like a days worth of airplanes that fly over US.  

https://traffic-viz.github.io/opensky_impala.html?highlight=opensky
'''


if __name__ == '__main__':

    if False:
        '''
        This is a targeted search to download trajectory info for a0a8da
        '''
        flight_icao24 = "a0a8da"
        flight_date_y_m_d = "2021-09-22"

        flight = opensky.history(flight_date_y_m_d, icao24=flight_icao24,return_flight=True)

        fig, ax = testPlotPlane(flight_icao24=flight_icao24, flight_date_y_m_d=flight_date_y_m_d, flight=flight)


    if False:
        '''
        Here I am attempting to download all flights within a box surrounding BEACON for the 10 minute window
        centered on event r5903 e86227.
        '''
        eventid = 86227
        run = 5903
        timestamp = 1632289193.891936
        flight_icao24 = "a0a8da"
        flight_date_y_m_d = "2021-09-22"

        # flight = opensky.history(
        #     start   =   timestamp-5*60,
        #     stop    =   timestamp+5*60,
        #     bounds  =   airspace_bbox,
        #     return_flight = True,
        #     )

        flight_traffic = opensky.history(
            start   =   timestamp-5*60,
            stop    =   timestamp+5*60,
            bounds  =   airspace_bbox,
            return_flight = False,
            )

        fig, ax = prepareMapAxisBEACON()
        for flight in flight_traffic:
            fig, ax = testPlotPlane(flight=flight, ax=ax, fig=fig)
            # flight_traffic.query('altitude > 12000')

    if True:
        '''
        This will download and pickle the dataframes for planes within the airspace_bbox for the first week of september.
        '''
        if 'BEACON_PICKLED_AIRPLANE_DATA' not in os.environ:
            print('Warning! BEACON_PICKLED_AIRPLANE_DATA environment variable not defined.')
            print('Setting storage_location_base to current directory.')
            storage_location_base = './'
        elif 'BEACON_PICKLED_AIRPLANE_DATA' in os.environ and not os.path.isdir(os.environ['BEACON_PICKLED_AIRPLANE_DATA']):
            print('WARNING, THE ENVIRONMENT VARIABLE BEACON_PICKLED_AIRPLANE_DATA DOES NOT POINT TO A DIRECTORY')
            print('Setting storage_location_base to current directory.')
            storage_location_base = './'
        else:    
            storage_location_base = os.environ['BEACON_PICKLED_AIRPLANE_DATA']

        box_string = str(airspace_bbox).replace('.','p').replace('(','box_').replace(')','').replace(', ', '_')
        storage_location = os.path.join(storage_location_base, box_string)

        if not os.path.exists(storage_location):
            print('Attempting to make airspace box folder %s'%storage_location)
            os.mkdir(storage_location)

        start_date = datetime(2021, 8, 31)
        stop_date = datetime(2021, 10, 2)
        delta = stop_date - start_date

        for i in range(delta.days + 1):
            day = start_date + timedelta(days=i) 
            day_string = str(day.date())
            for h in range(24):
                start_time = day + timedelta(hours=h)
                start_time_utc_timestamp = int(start_time.strftime("%s")) - int(start_time.strftime("%s"))%3600 #Ensuring to start on the hour, faster for impala
                stop_time_utc_timestamp = start_time_utc_timestamp + 3600

                filename = os.path.join(storage_location , day_string + '_%i.pkl'%h)

                if os.path.exists(filename):
                    print('filename exists: %s'%filename)
                    print('Skipping')
                    continue
                else:
                    print('filename does not exist: %s'%filename)
                    print('Retreiving airplane data')

                    flight_traffic = opensky.history(
                        start   =   start_time_utc_timestamp,
                        stop    =   stop_time_utc_timestamp,
                        bounds  =   airspace_bbox,
                        return_flight = False,
                        )

                    print('Pickling pandas dataframe')
                    flight_traffic.data.to_pickle(filename)
                    print('Done.  Filesize = %.3f MB'%(os.path.getsize(filename)/(1024*1024)))
