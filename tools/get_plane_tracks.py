'''
This will process the hdf5 files created using convert_flight_csv_to_hdf5
providing functions for searching through multiple hdf5 files.
'''

#General Imports
import numpy
import pylab
import glob
import csv
import sys
import os
import datetime
import pandas as pd
import itertools
import h5py
from pprint import pprint

#Personal Imports
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile
import tools.info as info

#Plotting Imports
import pymap3d as pm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


c = 2.99700e8 #m/s

flight_data_location_hdf5 = '/project2/avieregg/beacon/flight_backup_jan2020/data/altered/' #The files will be hdf5.

def filenameToDatetime(filename):
    try:
        return datetime.datetime.strptime(filename.split('/')[-1],'barcroft_%Y-%m-%d-%H.h5')
    except Exception as e:
        print('Error in filenameToDatetime.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    


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
    try:
        files = numpy.array(glob.glob(flight_data_location_hdf5+'*.h5'))
        timestamps = []
        for index, file in enumerate(files):
            timestamps.append(filenameToDatetime(file).timestamp())

        timestamps = numpy.array(timestamps)
        sort_indices = numpy.argsort(timestamps)
        sorted_files = files[sort_indices]
        timestamps = timestamps[sort_indices]


        # cut_indices = numpy.where(numpy.logical_and(timestamps >= start, timestamps <= stop))[0] #THIS ONLY CATCHES THINGS WHEN RUNS CROSS SOME TIMESTAMP WINDOW< NOT CORRECT
        # if numpy.size(cut_indices) != 0:
        #     start_file_index = max(0,min(cut_indices)-hour_window)
        #     stop_file_index = min(len(files),max(cut_indices) + 1 + hour_window)
        #     return sorted_files[start_file_index:stop_file_index]
        # else:
        #     return []
        try:
            start_file_index = numpy.max(numpy.where(timestamps <= start)[0])
            stop_file_index = numpy.min(numpy.where(timestamps > stop)[0])
            return sorted_files[start_file_index:stop_file_index]
        except Exception as e:
            print('Error in getting files from timestamp.')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            return []

    except Exception as e:
        print('Error in getFileNamesFromTimestamps.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def getTracks(start,stop,min_approach_cut_km,hour_window = 12):
    '''
    Uses getFileNamesFromTimestamps to get the relevant files, then actually opens
    and scans these files for the relevant positional information.  It will then
    convert the information to an easily usable format sorted by planes.
    '''
    try:
        relevant_files = getFileNamesFromTimestamps(start,stop,hour_window=hour_window)
        all_vals_exists = False
        if numpy.size(relevant_files) != 0:
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
        else:
            return [],[]
    except Exception as e:
        print('Error in getTracks.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def getENUTrackDict(start,stop,min_approach_cut_km,hour_window = 12,flights_of_interest=[]):
    '''
    This will return a dict with the trajectories of each plane observed in the 
    period of time specified (given in UTC timestamps).

    NOT UPDATED YET
    '''
    try:
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
            return {}, []
    except Exception as e:
        print('Error in getENUTrackDict.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def getTimeDelaysFromTrack(track, adjusted_antennas_physical=None,adjusted_antennas_phase_hpol=None,adjusted_antennas_phase_vpol=None, adjusted_cable_delays=None):
    '''
    Given a trajectory (each row specified x,y,z,t in ENU), this will determine the
    expected set of time delays based on the current saved antenna positions.

    If you want to test a new location for the antennas you can pass dictionaries
    the new locations as kwargs.  They are expected to be the same format as from
    loadAntennaLocationsENU.

    Same for cable delays.
    '''
    try:
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()# MAKE SURE TO PUT THE DEPLOY INDEX CORRECTLY


        if adjusted_antennas_physical is not None:
            print('Using adjusted_antennas_physical rather than antennas_physical.')
            antennas_physical = adjusted_antennas_physical
        if adjusted_antennas_phase_hpol is not None:
            print('Using adjusted_antennas_phase_hpol rather than antennas_phase_hpol.')
            antennas_phase_hpol = adjusted_antennas_phase_hpol
        if adjusted_antennas_phase_vpol is not None:
            print('Using adjusted_antennas_phase_vpol rather than antennas_phase_vpol.')
            antennas_phase_vpol = adjusted_antennas_phase_vpol

        if adjusted_cable_delays is not None:
            print('Using given values of cable delays rather than the currently saved values.')
            cable_delays = adjusted_cable_delays.copy()
        else:
            cable_delays = info.loadCableDelays()

        pprint(cable_delays)

        pairs = list(itertools.combinations((0,1,2,3), 2))

        labels = ['Physical','Hpol Phase Center','Vpol Phase Center']
        print_prefixs = {   'Physical':'expected_time_differences_physical' ,
                            'Hpol Phase Center':'expected_time_differences_hpol' ,
                            'Vpol Phase Center':'expected_time_differences_vpol'}

        tof = {}
        dof = {}
        dt = {}

        for index, antennas in enumerate([antennas_physical, antennas_phase_hpol, antennas_phase_vpol]):
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
    except Exception as e:
        print('Error in getTimeDelaysFromTrack.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

def getKnownPlaneTracks(ignore_planes=[]):
    '''
    Given the plane information outlines in loadKnownPlaneDict this will return
    a compisite dtype array with information about lat, lon, alt, and timing for
    each unique reported position of each plane in loadKnownPlaneDict.

    This can then be interpolated by the user at timestamps to get expected
    position corresponding to events in a run. 
    '''
    try:
        known_planes = info.loadKnownPlaneDict(ignore_planes=ignore_planes)
        output_tracks = {}
        calibrated_trigtime = {}
        for key in list(known_planes.keys()):
            runs = numpy.unique(known_planes[key]['eventids'][:,0])
            calibrated_trigtime[key] = numpy.zeros(len(known_planes[key]['eventids'][:,0]))
            for run in runs:
                run_cut = known_planes[key]['eventids'][:,0] == run
                reader = Reader(os.environ['BEACON_DATA'],run)
                eventids = known_planes[key]['eventids'][run_cut,1]
                try:
                    filename = createFile(reader) #Creates an analysis file if one does not exist.  Returns filename to load file.  
                    with h5py.File(filename, 'r') as file:
                        calibrated_trigtime[key][run_cut] = file['calibrated_trigtime'][...][eventids]
                except Exception as e:
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)
                    print('Calculating calibrated trig times.')
                    calibrated_trigtime[key][run_cut] = getEventTimes(reader,plot=False,smooth_window=101)[eventids]

            known_flight = known_planes[key]['known_flight']
            all_vals = getTracks(min(calibrated_trigtime[key]),max(calibrated_trigtime[key]),1000,hour_window=12)[1]
            vals = all_vals[all_vals['names'] == known_planes[key]['known_flight']]
            vals = vals[numpy.unique(vals['timestamps'],return_index=True)[1]] 
            output_tracks[key] = vals
        return known_planes, calibrated_trigtime, output_tracks
    except Exception as e:
        print('Error in getKnownPlaneTracksLatLon.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

class PlanePoly:
    def __init__(self, t, enu, order=7,plot=False):
        '''
        Given times (array of numbers) and enu data (tuple of 3 arrays of same length as t),
        this will create 3 polynomial fit functions to each of the ENU coordinates.  These
        can then be used to calculate ENU coordinates of the planes at interpolated values
        in time.
        '''
        try:
            self.order = order
            self.time_offset = min(t)
            self.fit_params = []
            self.poly_funcs = []
            self.range = (min(t),max(t))

            if plot == True:
                plane_norm_fig = plt.figure()
                plane_norm_ax = plt.gca()
            for i in range(3):
                index = numpy.sort(numpy.unique(enu[i],return_index=True)[1]) #Indices of values to be used in fit.
                self.fit_params.append(numpy.polyfit(t[index] - self.time_offset, enu[i][index],self.order)) #Need to shift t to make fit work
                self.poly_funcs.append(numpy.poly1d(self.fit_params[i]))
                if plot == True:
                    plane_norm_ax.plot(t,self.poly_funcs[i](t - self.time_offset)/1000.0,label='Fit %s Coord order %i'%(['E','N','U'][i],order))
                    plane_norm_ax.scatter(t[index],enu[i][index]/1000.0,label='Data for %s'%(['E','N','U'][i]),alpha=0.3)
            if plot == True:
                plt.legend()
                plt.minorticks_on()
                plt.grid(b=True, which='major', color='k', linestyle='-')
                plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
                plt.xlabel('Timestamp')
                plt.ylabel('Distance from Ant0 Along Axis (km)')

        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)


    def poly(self,t):
        '''
        Will return the resulting values of ENU for the given t.
        '''
        try:
            return numpy.vstack((self.poly_funcs[0](t-self.time_offset),self.poly_funcs[1](t-self.time_offset),self.poly_funcs[2](t-self.time_offset))).T
        except Exception as e:
            print('\nError in %s'%inspect.stack()[0][3])
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)



if __name__ == '__main__':
    files = numpy.array(glob.glob(flight_data_location_hdf5+'*.h5')) 
    time = filenameToDatetime(numpy.random.choice(files)) #random time just to test code. 
    start = time.timestamp()
    stop = time.timestamp()+0.5*60*60
    min_approach_cut_km = 25 #km
    #unique_flights,all_vals = getTracks(start,stop,min_approach_cut_km,hour_window = 12)
    flight_tracks_ENU, all_vals = getENUTrackDict(start,stop,min_approach_cut_km,hour_window = 0,flights_of_interest=[])


    cm = plt.cm.get_cmap('viridis')

    #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
    #NS Pairs = (0,2), (1,3)
    #EW Pairs = (0,1), (2,3)


    plot_distance_cut_limit = 50 #km

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

    # plt.figure()
    # zero = info.loadAntennaZeroLocation()
    # plt.scatter(all_vals['lon'],all_vals['lat'],alpha=0.5,s=1,label='Flight Tracks')

    
    # #plt.scatter(all_vals['lon'][all_vals['names'] == 'a14c0f'],all_vals['lat'][all_vals['names'] == 'a14c0f'],c=all_vals['timestamps'][all_vals['names'] == 'a14c0f'],alpha=0.5,s=1)
    # #plt.colorbar()    
    # plt.scatter(zero[1],zero[0],c='r',label='Beacon Ant 0')
    # plt.xlabel('Longitude')
    # plt.ylabel('Latitude')
    # plt.minorticks_on()
    # plt.grid(b=True, which='major', color='k', linestyle='-')
    # plt.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.5)
    # plt.legend()



    for ant_i, ant_j in [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]:
        #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        #NS Pairs = (0,2), (1,3)
        #EW Pairs = (0,1), (2,3)
        #Plot tracks
        plt.figure()
        existing_locations_A = numpy.array([])
        existing_locations_B = numpy.array([])

        for flight in list(flight_tracks_ENU.keys()):
            track = flight_tracks_ENU[flight]
            tof, dof, dt = getTimeDelaysFromTrack(track)
            distance = numpy.sqrt(numpy.sum(track[:,0:3]**2,axis=1))/1000 #km

            if plot_distance_cut_limit is not None:
                plot_distance_cut = distance <= plot_distance_cut_limit
            else:
                plot_distance_cut = numpy.ones_like(distance,dtype=bool)

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

        plt.xlabel('Time Since Timestamp=%0.1f (min)'%start)
        plt.ylabel('Expected Observed Time Difference\nB/w Hpol %i and %i (ns)'%(ant_i,ant_j))
        cbar = plt.colorbar()
        cbar.set_label('Distance From BEACON (km)', rotation=90)

    #use_north_south = False
    '''
    for use_north_south in [False,True]:
        #pairs = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3)
        #NS Pairs = (0,2), (1,3)
        #EW Pairs = (0,1), (2,3)
        #Plot tracks
        plt.figure()
        existing_locations_A = numpy.array([])
        existing_locations_B = numpy.array([])
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
        plt.ylabel('Expected Observed Time Difference\nB/w Hpol %i and %i (ns)'%(ant_i,ant_j))
        cbar = plt.colorbar()
        cbar.set_label('Distance From BEACON (km)', rotation=90)

        plt.subplot(2,1,2,sharex=ax)        
        plt.xlabel('Time Since Timestamp=%0.1f (min)'%start)
        plt.ylabel('Expected Observed Time Difference\nB/w Hpol %i and %i (ns)'%(ant_k,ant_l))
        cbar = plt.colorbar()
        cbar.set_label('Distance From BEACON (km)', rotation=90)

        '''