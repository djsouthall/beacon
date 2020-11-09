'''
This will define the antennas and station classes that may be used in the future.
Initially these will basically just store locations.
'''
import numpy
import itertools
import sys
import os
import pymap3d as pm
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])

import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()




class Antenna:
    '''
    Stores the attributes, information, and functions for antennas of the BEACON array.  
    When creating an antenna, the location should be given as relative to the station location
    in ENU coordinates.


    Parameters
    ----------


    Attributes
    ----------

    See Also
    --------
    Antenna
    '''
    def __init__(self,antenna_key,physical_enu,h_phase_enu=None,v_phase_enu=None):
        try:
            self.key = antenna_key
            self.physical_x = physical_enu[0]
            self.physical_y = physical_enu[1]
            self.physical_z = physical_enu[2]

            if h_phase_enu is not None:
                self.h_phase_x = h_phase_enu[0]
                self.h_phase_y = h_phase_enu[1]
                self.h_phase_z = h_phase_enu[2]
            else:
                self.h_phase_x = self.physical_x
                self.h_phase_y = self.physical_y
                self.h_phase_z = self.physical_z

            if v_phase_enu is not None:
                self.v_phase_x = v_phase_enu[0]
                self.v_phase_y = v_phase_enu[1]
                self.v_phase_z = v_phase_enu[2]
            else:
                self.v_phase_x = self.physical_x
                self.v_phase_y = self.physical_y
                self.v_phase_z = self.physical_z


        except Exception as e:
            print('Error in Antenna.__init__()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
           
    def setPhaseCenter(h_phase_x=None,h_phase_y=None,h_phase_z=None,v_phase_x=None,v_phase_y=None,v_phase_z=None):
        try:
            if h_phase_x is not None:
                self.h_phase_x = h_phase_x
            if h_phase_y is not None:
                self.h_phase_y = h_phase_y
            if h_phase_z is not None:
                self.h_phase_z = h_phase_z
            if v_phase_x is not None:
                self.v_phase_x = v_phase_x
            if v_phase_y is not None:
                self.v_phase_y = v_phase_y
            if v_phase_z is not None:
                self.v_phase_z = v_phase_z
        except Exception as e:
            print('Error in Antenna.setPhaseCenter()')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

class Station:
    '''
    Stores the attributes, information, and functions for a station of antennas for the BEACON array.  
    When adding antennas, the locations should be relative to the given station location.  So for instance
    if you decide to set Antenna 0 as the station location then the (x,y,z) coordinates of Antenna 0 should
    be given as (0.0,0.0,0.0).

    Parameters
    ----------


    Attributes
    ----------

    See Also
    --------
    Antenna
    '''
    def __init__(self,station_key,global_coord):
        try:
            self.key = station_key
            self.latitude = global_coord[0]#latitude
            self.longitude = global_coord[1]#longitude
            self.elevation = global_coord[2]#elevation

            self.antennas  = {}

            self.known_sources = {}
        except Exception as e:
            print('Error in Station.__init__().')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addAntennaRelative(self,antenna_key,physical_enu,h_phase_enu=None,v_phase_enu=None):
        '''
        Adds an antenna using relative (x,y,z) coordinates relative to the station location. 
        '''
        try:
            self.antennas[antenna_key] = Antenna(antenna_key,physical_enu,h_phase_enu=h_phase_enu,v_phase_enu=v_phase_enu)
        except Exception as e:
            print('Error in Station.addAntennaRelative().')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def addSource(self,source_key,global_coord):
        self.known_sources[source_key] = Source(source_key,global_coord)
        self.known_sources[source_key].calculateRelativeCoordinatesToStation(self)



class Source:
    '''
    Stores the attributes, information, and functions for an RF source such as a pulser.

    Parameters
    ----------


    Attributes
    ----------

    See Also
    --------
    Antenna
    '''
    def __init__(self,source_key,global_coord):
        try:
            self.key = source_key
            
            self.latitude = global_coord[0]#latitude
            self.longitude = global_coord[1]#longitude
            self.elevation = global_coord[2]#elevation

            self.relative_enu = {}
        except Exception as e:
            print('Error in Station.__init__().')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

    def calculateRelativeCoordinatesToStation(self,station):
        '''
        Given a station this will determine relative coordinates of the source to the
        station and store them in a dict using the station key.
        '''
        try:
            self.relative_enu[station.key] = pm.geodetic2enu(self.latitude,self.longitude,self.elevation,station.latitude,station.longitude,station.elevation)
        except Exception as e:
            print('Error in Source.calculateRelativeCoordinatesToStation().')
            print(e)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

        



if __name__ == '__main__':
    try:
        pairs = list(itertools.combinations((0,1,2,3), 2))
        Antennas = {0:(0.0,0.0,0.0),1:(-6.039,-1.618,2.275),2:(-1.272,-10.362,1.282),3:(3.411,-11.897,-0.432)}
        pulser_location = (37.5861,-118.2332,3779.52)#latitude,longitude,elevation
        A0Location = (37.5893,-118.2381,3894.12)#latitude,longtidue,elevation

        station = Station('station_0',A0Location)
        station.addSource('pulser_0',pulser_location)
        for key, value in Antennas.items():
            station.addAntennaRelative('ant_%s'%str(key),value)


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






