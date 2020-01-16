'''
This will process the hdf5 files created using convert_flight_csv_to_hdf5
providing functions for searching through multiple hdf5 files.
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
from matplotlib.collections import LineCollection

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
c = 2.99700e8 #m/s

flight_data_location_hdf5 = '/project2/avieregg/beacon/flight_backup_jan2020/data/altered/' #The files will be hdf5.

def filenameToDatetime(filename):
    return datetime.datetime.strptime(filename.split('/')[-1],'barcroft_%Y-%m-%d-%H.h5')

def getFileNamesFromTimestamps(start,stop,hour_window=12):
    '''
    Given a start and stop utc stime stamp, this will determine which
    files are appropriate to open, then pull the relevant rows from these
    files and return them. 

    Hour window will be how many hours of files on either side of the actual
    range to include.  This will help catch events that have timestamped events
    in files not correlating well with the timestamp title of the file (i.e. if
    an event is reported at NAME.h5 timestamp but is reported a planes location
    that was measured earlier, you can catch that).
    '''
    files = numpy.array(glob.glob(flight_data_location_hdf5+'*.h5'))
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

def getTracks(start,stop,min_approach_cut_km,hour_window = 12):
    '''
    Uses getFileNamesFromTimestamps to get the relevant files, then actually opens
    and scans these files for the relevant positional information.  It will then
    convert the information to an easily usable format sorted by planes.
    '''
    relevant_files = getFileNamesFromTimestamps(start,stop,hour_window=hour_window)
    all_vals_exists = False
    for index, rfile in enumerate(relevant_files):
        with h5py.File(rfile, 'r') as file:
            cut = file['closest_approach'][...]/1000 <= min_approach_cut_km
            dtype = [('names','U32'),('timestamps',float), ('lat',float), ('lon',float), ('alt',float), ('closest_approach',float)] 
            #cut = numpy.logical_and(cut,numpy.isin(numpy.arange(len(cut)),numpy.unique(file['timestamps'][...],return_index=True)[1])) #bool cut of first occurance of each timestamp to remove repeated.


            vals = numpy.zeros(sum(cut),dtype=dtype)
            vals['names'] = numpy.array([s.split('_')[-1] for s in numpy.array(file['names'][cut],dtype='U32')]) #Only using the hex portion.  
            vals['timestamps'] = numpy.array(file['timestamps'][cut],dtype=float)
            vals['lat'] = numpy.array(file['lat'][cut],dtype=float)
            vals['lon'] = numpy.array(file['lon'][cut],dtype=float)
            vals['alt'] = numpy.array(file['alt'][cut],dtype=float)
            vals['closest_approach'] = numpy.array(file['closest_approach'][cut],dtype=float)

            #vals = numpy.vstack((numpy.array(file['names'][cut],dtype=str),numpy.array(file['timestamps'][cut],dtype=float),numpy.array(file['lat'][cut],dtype=float),numpy.array(file['lon'][cut],dtype=float),numpy.array(file['alt'][cut],dtype=float),numpy.array(file['closest_approach'][cut],dtype=float))).T
        if all_vals_exists  == False:
            all_vals = vals
            all_vals_exists = True
        else:
            all_vals = numpy.append(all_vals,vals)

    if numpy.size(all_vals) != 0:
        all_vals = all_vals[numpy.argsort(all_vals['timestamps'].astype(float))] #Sorting lines by timestamp. 
        unique_flights = numpy.unique(all_vals['names'])
        return unique_flights,all_vals
    else:
        return [],[]

def getENUTrackDict(start,stop,min_approach_cut_km,hour_window = 12,flights_of_interest=[]):
    '''
    This will return a dict with the trajectories of each plane observed in the 
    period of time specified (given in UTC timestamps).

    NOT UPDATED YET
    '''
    unique_flights, all_vals = getTracks(start,stop,min_approach_cut_km,hour_window=hour_window)

    if numpy.size(flights_of_interest) != 0:
        unique_flights = unique_flights[numpy.isin(unique_flights,flights_of_interest)]
        all_vals = all_vals[numpy.isin(all_vals['names'],flights_of_interest)]

    if numpy.size(all_vals) != 0:
        flight_tracks_ENU = {}
        origin = info.loadAntennaZeroLocation(deploy_index = 1) #Antenna 0

        for unique_flight in unique_flights:
            flight_cut = numpy.where(all_vals['names'] == unique_flight)[0]
            flight_cut = flight_cut[numpy.unique(all_vals['timestamps'][flight_cut],return_index=True)[1]] #Removing repeated timestamps per flight
            ts = all_vals['timestamps'][flight_cut]

            #cut = numpy.logical_and(cut,numpy.isin(numpy.arange(len(cut)),numpy.unique(file['timestamps'][...],return_index=True)[1])) #bool cut of first occurance of each timestamp to remove repeated.

            enu = pm.geodetic2enu(all_vals['lat'][flight_cut],all_vals['lon'][flight_cut],all_vals['alt'][flight_cut],origin[0],origin[1],origin[2])  #converts to ENU
            sorted_track_indices = numpy.argsort(ts)
            # x, y, z, t
            flight_tracks_ENU[unique_flight] = numpy.vstack((numpy.asarray(enu),ts[None,:])).T[sorted_track_indices]

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
    for index, antennas in enumerate([antennas_phase_hpol,antennas_phase_vpol]):
        tof[print_prefixs[labels[index]]] = {}
        dof[print_prefixs[labels[index]]] = {}
        dt[print_prefixs[labels[index]]] = {}

        #print('\nCalculating expected time delays from %s location'%print_prefixs[labels[index]])
        for antenna, location in antennas.items():
            tof[print_prefixs[labels[index]]][antenna] = []
            dof[print_prefixs[labels[index]]][antenna] = []
            for plane_location in track:
                distance = numpy.sqrt((plane_location[0] - location[0])**2 + (plane_location[1] - location[1])**2 + (plane_location[2] - location[2])**2)
                time = (distance / c)*1e9 #ns
                if index == 0:
                    time += 0 #Physical, assuming no cable delay
                elif index == 1:
                    time += cable_delays['hpol'][antenna]
                elif index == 2:
                    time += cable_delays['vpol'][antenna]
                tof[print_prefixs[labels[index]]][antenna].append(time)
                dof[print_prefixs[labels[index]]][antenna].append(distance) #Does not include cable delays

            tof[print_prefixs[labels[index]]][antenna] = numpy.array(tof[print_prefixs[labels[index]]][antenna])
            dof[print_prefixs[labels[index]]][antenna] = numpy.array(dof[print_prefixs[labels[index]]][antenna])
        dt[print_prefixs[labels[index]]] = {}
        for pair in pairs:
            dt[print_prefixs[labels[index]]][pair] = tof[print_prefixs[labels[index]]][pair[0]] - tof[print_prefixs[labels[index]]][pair[1]] 

    return tof, dof, dt 

if __name__ == '__main__':
    files = numpy.array(glob.glob(flight_data_location_hdf5+'*.h5'))    
    time = filenameToDatetime(files[100]) #random time just to test code. 
    start = time.timestamp(),
    stop = time.timestamp()+60*60
    min_approach_cut_km = 500 #km
    #unique_flights,all_vals = getTracks(start,stop,min_approach_cut_km,hour_window = 12)
    flight_tracks_ENU, all_vals = getENUTrackDict(start,stop,min_approach_cut_km,hour_window = 0,flights_of_interest=[])


    cm = plt.cm.get_cmap('viridis')

    #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    #NS Pairs = (0,2), (1,3)
    #EW Pairs = (0,1), (2,3)


    plot_distance_cut_limit = None# 25 #km

    if plot_distance_cut_limit is not None:
        norm = plt.Normalize(0,plot_distance_cut_limit)
    else:
        norm = plt.Normalize(0,100)


    #Plot lonlat
    '''
    Make a plot that hopefully illustrates where planes are visible for us.  Why are the tracks
    disappear where they do?
    TODO
    '''
    plt.figure()
    zero = info.loadAntennaZeroLocation()
    plt.scatter(all_vals['lon'],all_vals['lat'],alpha=0.5,s=1)
    #plt.scatter(all_vals['lon'][all_vals['names'] == 'a14c0f'],all_vals['lat'][all_vals['names'] == 'a14c0f'],c=all_vals['timestamps'][all_vals['names'] == 'a14c0f'],alpha=0.5,s=1)
    #plt.colorbar()    
    plt.scatter(zero[1],zero[0],c='r')


    #Plot tracks
    plt.figure()
    existing_locations_A = numpy.array([])
    existing_locations_B = numpy.array([])

    use_north_south = False

    #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    #NS Pairs = (0,2), (1,3)
    #EW Pairs = (0,1), (2,3)
    if use_north_south == True:
        ant_i = 0
        ant_j = 2
        ant_k = 1
        ant_l = 3
    else:  
        ant_i = 0
        ant_j = 1
        ant_k = 2
        ant_l = 3
 

    for flight in list(flight_tracks_ENU.keys()):
        track = flight_tracks_ENU[flight]
        tof, dof, dt = getTimeDelaysFromTrack(track)
        distance = numpy.sqrt(numpy.sum(track[:,0:3]**2,axis=1))/1000 #km

        if plot_distance_cut_limit is not None:
            plot_distance_cut = distance <= plot_distance_cut_limit
        else:
            plot_distance_cut = numpy.ones_like(distance,dtype=bool)

        
        ax = plt.subplot(2,1,1)
        x = (track[plot_distance_cut,3] - start)/60
        y = dt['expected_time_differences_hpol'][(ant_i, ant_j)][plot_distance_cut]
        plt.plot(x,y,linestyle = '--',alpha=0.5)
        text_color = plt.gca().lines[-1].get_color()
        plt.scatter(x,y,c=distance[plot_distance_cut],cmap=cm,norm=norm)

        #Attempt at avoiding overlapping text.
        text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
        if existing_locations_A.size != 0:
            if len(numpy.shape(existing_locations_A)) == 1:
                dist = numpy.sqrt((text_loc[0]-existing_locations_A[0])**2 + (text_loc[1]-existing_locations_A[1])**2) #weird units but works
                while dist < 15:
                    text_loc[1] -= 1
                    dist = numpy.sqrt((text_loc[0]-existing_locations_A[0])**2 + (text_loc[1]-existing_locations_A[1])**2) #weird units but works
            else:
                dist = min(numpy.sqrt((existing_locations_A[:,0] - text_loc[0])**2 + (existing_locations_A[:,1] - text_loc[1])**2))
                while dist < 15:
                    text_loc[1] -= 1
                    dist = min(numpy.sqrt((existing_locations_A[:,0] - text_loc[0])**2 + (existing_locations_A[:,1] - text_loc[1])**2)) #weird units but works
            existing_locations_A = numpy.vstack((existing_locations_A,text_loc))
        else:
            existing_locations_A = text_loc           

        plt.text(text_loc[0],text_loc[1],flight,color=text_color,withdash=True)



        plt.subplot(2,1,2,sharex=ax)
        x = (track[plot_distance_cut,3] - start)/60
        y = dt['expected_time_differences_hpol'][(ant_k, ant_l)][plot_distance_cut]
        plt.plot(x,y,linestyle = '--',alpha=0.5)
        text_color = plt.gca().lines[-1].get_color()
        plt.scatter(x,y,c=distance[plot_distance_cut],cmap=cm,norm=norm)

        #Attempt at avoiding overlapping text.
        text_loc = numpy.array([numpy.mean(x)-5,numpy.mean(y)])
        if existing_locations_B.size != 0:
            if len(numpy.shape(existing_locations_B)) == 1:
                dist = numpy.sqrt((text_loc[0]-existing_locations_B[0])**2 + (text_loc[1]-existing_locations_B[1])**2) #weird units but works
                while dist < 15:
                    text_loc[1] -= 1
                    dist = numpy.sqrt((text_loc[0]-existing_locations_B[0])**2 + (text_loc[1]-existing_locations_B[1])**2) #weird units but works
            else:
                dist = min(numpy.sqrt((existing_locations_B[:,0] - text_loc[0])**2 + (existing_locations_B[:,1] - text_loc[1])**2))
                while dist < 15:
                    text_loc[1] -= 1
                    dist = min(numpy.sqrt((existing_locations_B[:,0] - text_loc[0])**2 + (existing_locations_B[:,1] - text_loc[1])**2)) #weird units but works
            existing_locations_B = numpy.vstack((existing_locations_B,text_loc))
        else:
            existing_locations_B = text_loc           

        plt.text(text_loc[0],text_loc[1],flight,color=text_color,withdash=True)




    plt.subplot(2,1,1)        
    plt.xlabel('Time Since Timestamp=%0.1f (min)'%start)
    plt.ylabel('Expected Observed Time Difference\nB/w Hpol %i and %i (ns)'%(0,2))
    cbar = plt.colorbar()
    cbar.set_label('Distance From BEACON (km)', rotation=90)

    plt.subplot(2,1,2,sharex=ax)        
    plt.xlabel('Time Since Timestamp=%0.1f (min)'%start)
    plt.ylabel('Expected Observed Time Difference\nB/w Hpol %i and %i (ns)'%(1,3))
    cbar = plt.colorbar()
    cbar.set_label('Distance From BEACON (km)', rotation=90)
