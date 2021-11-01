'''
This will process the csv files and convert them to hdf5 with some useful changes like combining
the name columns and calculating the closest approach distance for each flight track.  
'''
import numpy
import pylab
import glob
import csv
import sys
import os
import datetime
import pandas as pd
import pymap3d as pm
import itertools
import matplotlib.pyplot as plt
import h5py

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
c = 2.99700e8 #m/s


flight_data_location_raw = '/project2/avieregg/beacon/flight_backup_jan2020/data/raw/' #This is the input, the files will be csv. 
flight_data_location_altered = '/project2/avieregg/beacon/flight_backup_jan2020/data/altered/' #This is the output.  The files will be hdf5.

def readTxt(infile, header=0, delimeter=','):
    '''
    This is intended only to read text from the plane tracker csv.
    It does additional processing on each row (combing first and second columns)
    and so would work poorly for other applications. 
    '''
    with open(infile,'r') as file:
        lines = file.readlines()
        vals = []
        for index, line in enumerate(lines):
            if index > header - 1:
                if len(line.split(delimeter)) <= 1:
                    continue
                else:
                    line = line.replace('\n','').split(delimeter)
                    line[1] = line[0].replace(' ','') + '_' + line[1].replace(' ','') #consolodating names to 1 value.
                    line.pop(0) #removing redundent first column.
                    vals.append( line )
    return vals

def filenameToDatetime(filename):
    return datetime.datetime.strptime(filename.split('/')[-1],'barcroft_%Y-%m-%d-%H.csv')

def getFileNamesFromTimestamps(start,stop,hour_window=12):
    '''
    Given a start and stop utc stime stamp, this will determine which
    files are appropriate to open, then pull the relevant rows from these
    files and return them. 

    Hour window will be how many hours of files on either side of the actual
    range to include.  This will help catch events that have timestamped events
    in files not correlating well with the timestamp title of the file (i.e. if
    an event is reported at NAME.csv timestamp but is reported a planes location
    that was measured earlier, you can catch that).
    '''
    files = numpy.array(glob.glob(flight_data_location_raw+'*.csv'))
    timestamps = []
    for index, file in enumerate(files):
        timestamps.append(filenameToDatetime(file).timestamp())

    timestamps = numpy.array(timestamps)
    sort_indices = numpy.argsort(timestamps)
    sorted_files = files[sort_indices]
    timestamps = timestamps[sort_indices]

    cut_indices = numpy.where(numpy.logical_and(timestamps >= start, timestamps <= stop))[0]
    if numpy.size(cut_indices) != 0:
        start_file_index = max(0,min(cut_indices)-hour_window)
        stop_file_index = min(len(files),max(cut_indices) + 1 + hour_window)
        return sorted_files[start_file_index:stop_file_index]
    else:
        return []

def getTracks(start,stop,hour_window = 12):
    '''
    Uses getFileNamesFromTimestamps to get the relevant files, then actually opens
    and scans these files for the relevant positional information.  It will then
    convert the information to an easily usable format sorted by planes.
    '''
    relevant_files = getFileNamesFromTimestamps(start,stop,hour_window=hour_window)
    all_vals = []
    for index, file in enumerate(relevant_files):
        vals = readTxt(file)
        if index == 0:
            all_vals = vals
        else:
            if numpy.size(vals) != 0:
                all_vals = numpy.vstack((all_vals,vals))

    if numpy.size(all_vals) != 0:
        all_vals = numpy.array(all_vals)
        all_vals = all_vals[numpy.argsort(all_vals[:,1].astype(float))] #Sorting lines by timestamp. 

        unique_flights = numpy.unique(all_vals[:,0])

        return unique_flights,all_vals
    else:
        return [],[]

def getENUTrackDict(start,stop,hour_window = 12,flights_of_interest=[]):
    '''
    This will return a dict with the trajectories of each plane observed in the 
    period of time specified (given in UTC timestamps).
    '''
    unique_flights,all_vals = getTracks(start,stop,hour_window=hour_window)

    if numpy.size(flights_of_interest) != 0:
        unique_flights = [u.replace(' ','') for u in unique_flights]
        all_vals[:,0] = [u.replace(' ','') for u in all_vals[:,0]]
        unique_flights = unique_flights[numpy.isin(unique_flights,flights_of_interest)]
        all_vals = all_vals[numpy.isin(all_vals[:,0],flights_of_interest),:]

    if numpy.size(all_vals) != 0:
    
        lat = all_vals[:,2].astype(float)
        lon = all_vals[:,3].astype(float)
        alt = all_vals[:,4].astype(float)*0.3048 #Expressing alt in meters now. 
        timestamps = all_vals[:,1].astype(float)

        flight_tracks_ENU = {}
        origin = info.loadAntennaZeroLocation(deploy_index = 1) #Antenna 0

        for unique_flight in unique_flights:
            flight_cut = all_vals[:,0] == unique_flight
            enu = pm.geodetic2enu(lat[flight_cut],lon[flight_cut],alt[flight_cut],origin[0],origin[1],origin[2])  #converts to ENU
            ts = timestamps[flight_cut]
            # x, y, z, t
            flight_tracks_ENU[unique_flight.replace(' ','')] = numpy.vstack((numpy.asarray(enu),ts[None,:])).T

        return flight_tracks_ENU, all_vals
    else:
        return [], []

def getTimeDelaysFromTrack(track):
    '''
    Given a trajectory (each row specified x,y,z,t in ENU), this will determine the
    expected set of time delays based on the current saved antenna positions.
    '''
    antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()# MAKE SURE TO PUT THE DEPLOY INDEX CORRECTLY
    cable_delays = info.loadCableDelays()
    pairs = list(itertools.combinations((0,1,2,3), 2))

    labels = ['Physical','Hpol Phase Center','Vpol Phase Center']
    print_prefixs = {   'Physical':'expected_time_differences_physical' ,
                        'Hpol Phase Center':'expected_time_differences_hpol' ,
                        'Vpol Phase Center':'expected_time_differences_vpol'}

    tof = {}
    dof = {}
    dt = {}
    for index, antennas in enumerate([antennas_physical,antennas_phase_hpol,antennas_phase_vpol]):
        tof[labels[index]] = {}
        dof[labels[index]] = {}
        dt[labels[index]] = {}

        #print('\nCalculating expected time delays from %s location'%labels[index])
        for antenna, location in antennas.items():
            tof[labels[index]][antenna] = []
            dof[labels[index]][antenna] = []
            for plane_location in track:
                distance = numpy.sqrt((plane_location[0] - location[0])**2 + (plane_location[1] - location[1])**2 + (plane_location[2] - location[2])**2)
                time = (distance / c)*1e9 #ns
                if index == 0:
                    time += 0 #Physical, assuming no cable delay
                elif index == 1:
                    time += cable_delays['hpol'][antenna]
                elif index == 2:
                    time += cable_delays['vpol'][antenna]
                tof[labels[index]][antenna].append(time)
                dof[labels[index]][antenna].append(distance)

            tof[labels[index]][antenna] = numpy.array(tof[labels[index]][antenna])
            dof[labels[index]][antenna] = numpy.array(dof[labels[index]][antenna])
        dt[labels[index]] = {}
        for pair in pairs:
            dt[labels[index]][pair] = tof[labels[index]][pair[0]] - tof[labels[index]][pair[1]] 

    return tof, dof, dt 

def catalogMinDistancesPerFlight(file,flights_of_interest=[],return_vals=False):
    '''
    Returns the minimum plane-ant0 distance for each flight. 
    '''
    #print(file)
    vals = numpy.array(readTxt(file))
    try:
        times = vals[:,1].astype(float)
        flight_tracks_ENU, all_vals = getENUTrackDict(min(times), max(times),hour_window=0,flights_of_interest=flights_of_interest)

        if numpy.size(flight_tracks_ENU) != 0:

            min_d = {}
            for key in (flight_tracks_ENU.keys()):
                track = flight_tracks_ENU[key]
                tof, dof, dt = getTimeDelaysFromTrack(track)
                min_d[key] = min(dof['Physical'][0])

            if return_vals == True:
                return min_d,vals
            else:
                return min_d
        else:
            if return_vals:
                return [],[]
            else:
                return []
    except Exception as e:
        print(e)
        if return_vals:
            return [],[]
        else:
            return []

def writeFilesWithDistance():
    files = glob.glob(flight_data_location_raw+'*.csv')
    for infile in files:
        outfile = infile.replace('/raw/','/altered/').replace('.csv','.h5')
        print(infile, '  -->  ', outfile)
        min_d, vals = catalogMinDistancesPerFlight(infile,return_vals=True)
        if numpy.size(min_d) != 0:
            names = vals[:,0]
            times = vals[:,1].astype(float)
            lat = vals[:,2].astype(float)
            lon = vals[:,3].astype(float)
            alt = vals[:,4].astype(float)*0.3048
            min_d = [min_d[f] for f in names]
            N = len(names)

            with h5py.File(outfile, 'w') as output:
                dtype = str(names.dtype).replace('U','S').replace('>','').replace('<','')
                output.create_dataset('names', (N,), dtype=dtype, compression='gzip', compression_opts=9, shuffle=True)
                output['names'][...] = names.astype(dtype)

                output.create_dataset('timestamps', (N,), dtype=float, compression='gzip', compression_opts=9, shuffle=True)
                output['timestamps'][...] = times

                output.create_dataset('lat', (N,), dtype=float, compression='gzip', compression_opts=9, shuffle=True)
                output['lat'][...] = lat

                output.create_dataset('lon', (N,), dtype=float, compression='gzip', compression_opts=9, shuffle=True)
                output['lon'][...] = lon

                output.create_dataset('alt', (N,), dtype=float, compression='gzip', compression_opts=9, shuffle=True)
                output['alt'][...] = alt

                output.create_dataset('closest_approach', (N,), dtype=float, compression='gzip', compression_opts=9, shuffle=True)
                output['closest_approach'][...] = min_d

                output.attrs['min_t'] = min(times)
                output.attrs['max_t'] = max(times)

                output.close()
                    


if __name__ == '__main__':
    writeFilesWithDistance()
    