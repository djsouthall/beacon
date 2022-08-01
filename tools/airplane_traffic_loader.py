#!/usr/bin/env python3
'''
This script is intended to interface with the dataframes downloaded using airplane_traffic_loader.py

This script is designed to be run using the same anaconda environment as the rest of BEACON, and should not need
the traffic package available, as the data used is downloaded and stored in pickled dataframes.  

Requires the pandas version of the loader to be at least as recent as the pandas version of the downlaoder/saver.
To do this I had to clone the conda environment I had on midway and then update pandas to 1.4.0.
'''

import sys
import os
import pandas as pd
import csv
import numpy
import matplotlib.pyplot as plt
from datetime import date, timedelta, datetime
import pymap3d as pm

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info

plt.ion()

airplane_datapath = os.path.join(os.environ['BEACON_PICKLED_AIRPLANE_DATA'],'box_-120p23761867_34p589339_-113p23761867_40p589339')
deploy_index = info.returnDefaultDeploy()
default_origin = info.loadAntennaZeroLocation(deploy_index=info.returnDefaultDeploy())

def getFileNamesFromTimestamps(start_time_utc_timestamp, stop_time_utc_timestamp, verbose=False):
    '''
    Given 2 timestamps, this will determine the filenames which have timestamps between those 2 times.
    '''
    start_datetime = datetime.fromtimestamp(start_time_utc_timestamp)
    start_date = datetime(start_datetime.year, start_datetime.month, start_datetime.day)
    start_hour = start_datetime.hour

    stop_datetime = datetime.fromtimestamp(stop_time_utc_timestamp)
    stop_date = datetime(stop_datetime.year, stop_datetime.month, stop_datetime.day)
    stop_hour = stop_datetime.hour

    delta = stop_datetime - start_datetime
    print('DELTA = ', delta)
    print('')

    # import pdb; pdb.set_trace()
    filenames = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i) 
        day_string = str(day.date())
        for h in range(24):
            # pdb.set_trace()
            if i == 0 and i == len(range(delta.days + 1)) - 1:
                #All within the same day
                if h < start_hour or h > stop_hour:
                    continue
            elif i == 0:
                #On first day, don't include files before timestamp
                if h < start_hour:
                    continue
            elif i == len(range(delta.days + 1)) - 1:
                #On last day, don't include files after timestamp
                if h > stop_hour:
                    continue

            #Construct filename
            filename = os.path.join(airplane_datapath , day_string + '_%i.pkl'%h)
            if os.path.exists(filename):
                filenames.append(filename)
            else:
                if verbose:
                    print('WARNING! Expected filename in given time range not found:')
                    print(filename)

    return filenames


def enu2Spherical(enu):
    '''
    2d array like ((e_0, n_0, u_0), (e_1, n_1, u_1), ... , (e_i, n_i, u_i))

    Return in degrees
    '''
    r = numpy.linalg.norm(enu, axis=1)
    phi = numpy.degrees(numpy.arctan2(enu[:,1],enu[:,0]))
    theta = numpy.degrees(numpy.arccos(enu[:,2]/r))
    # import pdb; pdb.set_trace()
    return numpy.vstack((r,phi,theta)).T

def addDirectionInformationToDataFrame(df, origin=default_origin, altitude_str='geoaltitude'):
    '''
    See https://opensky-network.org/datasets/states/README.txt

    This will add ENU, and distance, azimuth, and zenith information to the dataframe.

    Note that both geoaltitude and altitude (barometric) are available.
    '''
    e = numpy.zeros(len(df.index))
    n = numpy.zeros(len(df.index))
    u = numpy.zeros(len(df.index))
    utc_timestamp = numpy.zeros(len(df.index))

    #import pdb; pdb.set_trace()

    for i, (index, row) in enumerate(df.iterrows()):
        e[i], n[i], u[i] = pm.geodetic2enu(row['latitude'], row['longitude'], row[altitude_str]*0.3048, origin[0], origin[1], origin[2])
        utc_timestamp[i] = row['timestamp'].timestamp()

    r, phi, theta = enu2Spherical(numpy.vstack((e,n,u)).T).T

    df['east'] = e
    df['north'] = n
    df['up'] = u

    df['distance'] = r
    df['azimuth'] = phi
    df['zenith'] = theta

    df['utc_timestamp'] = utc_timestamp
    return df

def readPickle(filename):
    '''
    This executes pandas.read_pickle, but also adds a column for utc timestamp.

    This is for easier time slicing before calculations are performed.  
    '''
    df = pd.read_pickle(filename)
    df['utc_timestamp'] = numpy.array([row['timestamp'].timestamp() for index, row in df.iterrows()])
    return df



def getDataFrames(start_time_utc_timestamp, stop_time_utc_timestamp, origin=default_origin, query=None, verbose=False):
    '''
    Given 2 timestamps, this will load in all pandas dataframes which have timestamps between those 2 times.

    A query is automatically used to match table to given timestamps.
    '''
    filenames = getFileNamesFromTimestamps(start_time_utc_timestamp, stop_time_utc_timestamp, verbose=verbose)
    if verbose == True:
        print('Filenames:')
        print(filenames)

    time_query = 'utc_timestamp >= %f and utc_timestamp <= %f'%(start_time_utc_timestamp, stop_time_utc_timestamp)

    if len(filenames) > 0:
        for index, filename in enumerate(filenames):
            if verbose:
                print(index)
                print(filename)

            if index == 0:
                if query is None:
                    df = addDirectionInformationToDataFrame(readPickle(filename).query(time_query), origin=origin)
                else:
                    df = addDirectionInformationToDataFrame(readPickle(filename).query(time_query), origin=origin).query(query)
            else:
                if query is None:
                    addition = addDirectionInformationToDataFrame(readPickle(filename).query(time_query), origin=origin)
                else:
                    addition = addDirectionInformationToDataFrame(readPickle(filename).query(time_query), origin=origin).query(query)

                df = df.append(addition)
            if verbose:
                print('len(df)', len(df))
        return df
    else:
        print('No filenames matched the given time window.')
        return None


if __name__ == '__main__':
    eventid = 86227
    run = 5903
    timestamp = 1632289193.891936
    flight_icao24 = "a0a8da"
    flight_date_y_m_d = "2021-09-22"


    if True:
        start_time_utc_timestamp = timestamp - 10*60
        stop_time_utc_timestamp = timestamp + 10*60 
        df = getDataFrames(start_time_utc_timestamp, stop_time_utc_timestamp, origin=default_origin, query=None, verbose=False)

    else:

        # flight_icao24 = "a0a8da"

        # filenames = getFileNamesFromTimestamps(timestamp - 3600*2, timestamp + 3600*2)

        filenames = [   '/home/dsouthall/scratch-midway2/airplane_tracker_data/box_-120p23761867_34p589339_-113p23761867_40p589339/2021-08-31_0.pkl',
                        '/home/dsouthall/scratch-midway2/airplane_tracker_data/box_-120p23761867_34p589339_-113p23761867_40p589339/2021-08-31_1.pkl',
                        '/home/dsouthall/scratch-midway2/airplane_tracker_data/box_-120p23761867_34p589339_-113p23761867_40p589339/2021-08-31_2.pkl',
                        '/home/dsouthall/scratch-midway2/airplane_tracker_data/box_-120p23761867_34p589339_-113p23761867_40p589339/2021-08-31_3.pkl']



        # df = pd.read_pickle(filenames[0])
        
        query = 'altitude > 12000'
        for index, filename in enumerate(filenames):
            df = pd.read_pickle(filename)

            # print('\n' + filename)
            # print(df.loc[0])

            if index == 0:
                if query is None:
                    df = pd.read_pickle(filename)
                else:
                    df = pd.read_pickle(filename).query(query)
            else:
                if query is None:
                    # df = df.append(pd.read_pickle(filename))
                    df.append(pd.read_pickle(filename))
                else:
                    # df = df.append(pd.read_pickle(filename).query(query))
                    df.append(pd.read_pickle(filename).query(query))

        df = addDirectionInformationToDataFrame(df, origin=default_origin)
