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

def getFileNamesFromTimestamps(start_time_utc_timestamp, stop_time_utc_timestamp, verbose=False):
    '''
    Given 2 timestamps, this will determine the filenames which have timestamps between those 2 times.
    '''
    start_datetime = datetime.fromtimestamp(start_time_utc_timestamp)
    start_date = datetime(start_datetime.year, start_datetime.month, start_datetime.day)
    start_hour = start_datetime.hour

    stop_datetime = datetime.fromtimestamp(stop_time_utc_timestamp)
    stop_date = datetime(stop_datetime.year, stop_datetime.month, stop_datetime.day)
    stop_hour = start_datetime.hour

    delta = stop_date - start_date


    filenames = []
    for i in range(delta.days + 1):
        day = start_date + timedelta(days=i) 
        day_string = str(day.date())
        for h in range(24):
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





def getDataFrames(start_time_utc_timestamp, stop_time_utc_timestamp, query=None, verbose=False):
    '''
    Given 2 timestamps, this will load in all pandas dataframes which have timestamps between those 2 times.
    '''
    filenames = getFileNamesFromTimestamps(start_time_utc_timestamp, stop_time_utc_timestamp, verbose=verbose)

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
                # df = df.merge(pd.read_pickle(filename))
                df.merge(pd.read_pickle(filename))
            else:
                # df = df.merge(pd.read_pickle(filename).query(query))
                df.merge(pd.read_pickle(filename).query(query))
    return df


'''
Need to convert the lat lon alt info to enu and then theta phi.  Consider using pm.geodetic2enu(location[0],location[1],location[2],origin[0],origin[1],origin[2])
origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
'''

if __name__ == '__main__':
    eventid = 86227
    run = 5903
    timestamp = 1632289193.891936
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
                # df = df.merge(pd.read_pickle(filename))
                df.merge(pd.read_pickle(filename))
            else:
                # df = df.merge(pd.read_pickle(filename).query(query))
                df.merge(pd.read_pickle(filename).query(query))

    # df = getDataFrames(timestamp - 3600*2, timestamp + 3600*2, query=None)