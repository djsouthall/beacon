import json
import sys
import os
import copy
import pymap3d as pm
import numpy
import datetime

def loadSampleData():
    '''
    This loads sample dictionaries of each of the required configuration datatypes such that they can be used for
    testing and examples.
    '''
    origin = (37.589310, -118.237621, 3852.842222) #latitude,longtidue,elevation in m
    antennas_physical = {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}
    antennas_phase_hpol = {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}            
    antennas_phase_vpol = {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}
    cable_delays =  {   'hpol': [423.37836156, 428.43979143, 415.47714969, 423.58803498], \
                        'vpol': [428.59277751, 430.16685915, 423.56765695, 423.50469285]}
    return origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays

def default(obj):
    if isinstance(obj, numpy.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

def configWriter(json_path, origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description="",update_latlonel=False,force_write=True):
    '''
    The counterpart to configReader, given the normal calibration dictionaries, this will write a calibration file
    that can be loaded for later analysis.

    Parameters
    ----------
    json_path : str
        The path (including filename and extension) of the output file to be generated.
    origin : array-like
            The latitude, longitude, and elevation (in m) contained within a 3 element array-like object: [lat, lon, el]
    antennas_physical : dict
        A dictionary containing the ENU positions of the antennas (ENU with regards to the origin).  Example below:
        {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}
    antennas_phase_hpol : dict
        A dictionary containing the ENU positions of the hpol antennas (ENU with regards to the origin).  Phase here
        implies these positions are derived from some calibration, as opposed to the physical locations which are
        likely measured.  Example below:
        {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}            
    antennas_phase_vpol : dict
        A dictionary containing the ENU positions of the vpol antennas (ENU with regards to the origin).  Phase here
        implies these positions are derived from some calibration, as opposed to the physical locations which are
        likely measured.  Example below:
        {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}
    cable_delays : dict
        A dictionary containing the hpol and vpol cable delays.  Example below:
        cable_delays =  {   'hpol': [423.37836156, 428.43979143, 415.47714969, 423.58803498], \
                            'vpol': [428.59277751, 430.16685915, 423.56765695, 423.50469285]}
    description : str
        A string that can be used to describe the calibration.  Will be included at the top of the json file.
    update_latlonel : bool
        If True, then latlonel information will be calculated and stored based on the input enu coordinates.  Default
        behaviour is False.  
    force_write : bool
        If True then then this will alter the filename by appending it with _# until it reaches a number/filename
        that does not already exist.  This is intended to avoid overwriting existing calibration files.  If False
        then the calibration will not be saved at all.  
    '''

    if ~os.path.exists(json_path) or force_write == True:
        # Handle the case where the file exists.  Setup to change the extension number and add up. 
        
        if os.path.exists(json_path):
            #Beacuse of the previous or condition, if you get here then the file already exists AND force_write == True, so altering output filename.
            print('Filename exists.  Modifying output file name.')
            json_path = json_path.replace('.json', '_' + str(datetime.datetime.now()).replace(' ', '_').replace('.','p').replace(':','-') + '.json')

        print("Saving configuration file to %s"%json_path)
        data = {}
        data["description"] = description
        
        data["origin"] = {}
        data["origin"]["latlonel"] = origin

        data["antennas"] = {}


        for mast in range(4):
            data["antennas"]["ant%i"%mast] = {}

            data["antennas"]["ant%i"%mast]["physical"] = {}
            data["antennas"]["ant%i"%mast]["physical"]["latlonel"] = []
            data["antennas"]["ant%i"%mast]["physical"]["enu"] = antennas_physical[mast]

            data["antennas"]["ant%i"%mast]["hpol"] = {}
            data["antennas"]["ant%i"%mast]["hpol"]["latlonel"] = []
            data["antennas"]["ant%i"%mast]["hpol"]["enu"] = antennas_phase_hpol[mast]
            data["antennas"]["ant%i"%mast]["hpol"]["cable_delay"] = cable_delays["hpol"][mast]
            data["antennas"]["ant%i"%mast]["hpol"]["channel"] = 2*mast

            data["antennas"]["ant%i"%mast]["vpol"] = {}
            data["antennas"]["ant%i"%mast]["vpol"]["latlonel"] = []
            data["antennas"]["ant%i"%mast]["vpol"]["enu"] = antennas_phase_vpol[mast]
            data["antennas"]["ant%i"%mast]["vpol"]["cable_delay"] = cable_delays["vpol"][mast]
            data["antennas"]["ant%i"%mast]["vpol"]["channel"] = 2*mast + 1


        if update_latlonel == True:
            print('Updated latlonel data.')
            data = updateLatlonelFromENU(data)

        with open(json_path, 'w') as outfile:
            outfile.write(json.dumps(data, indent=4, sort_keys=False, default=default))
        return json_path
    else:
        print('No output file created.')

def configWriterFromLatLonElJSON(json_path,verbose=True):
    '''
    This basically calls configWriter for but first loads in the data from json_path's latlonel information to get ENU.
    It serves to turn a config that only has latlon el data into one that also has ENU data.
    '''
    origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays,  description = configReader(json_path, return_mode='enu', check=True, verbose=verbose, return_description=True)
    configWriter(json_path, origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description=description,update_latlonel=True,force_write=True)


def checkConfigConsistency(data, decimals=6, verbose=True):
    '''
    Given the data loaded from the configuration data file, this will check for defined ENU and latlonel coordinates, 
    and confirm if they agree with eachother.

    Parameters
    ----------
    data : dict
        The dict generated when loading a configuration from a json file.
    decimals : int
        Values are rounded before compared.  This chooses the precision using the numpy.around funciton.
    '''
    data = copy.deepcopy(data) #So as to not edit the original.
    origin = data['origin']['latlonel'] #Lat Lon Elevation, use for generating ENU from other latlonel values.     

    for mast in range(4):
        for key in ['hpol', 'vpol']:
            if numpy.asarray(data['antennas']['ant%i'%mast][key]['enu']).size > 0 and numpy.asarray(data['antennas']['ant%i'%mast][key]['latlonel']).size > 0:
                enu_from_latlonel = numpy.asarray(pm.geodetic2enu(data['antennas']['ant%i'%mast][key]['latlonel'][0],data['antennas']['ant%i'%mast][key]['latlonel'][1],data['antennas']['ant%i'%mast][key]['latlonel'][2],origin[0],origin[1],origin[2]))
                enu_from_json = numpy.asarray(data['antennas']['ant%i'%mast][key]['enu'])

                match = numpy.all(numpy.around(enu_from_latlonel,decimals=decimals) == numpy.around(enu_from_json,decimals=decimals))
                
                max_precision_to_check = 10
                decimals = max_precision_to_check
                while numpy.all(numpy.around(enu_from_latlonel,decimals=decimals) != numpy.around(enu_from_json,decimals=decimals)) or decimals == 0:
                    decimals -= 1

                print('Checking mast %i %s coordinates:  Match up to %i decimals'%(mast,key, decimals) + ['',' (Max precision checked)'][decimals == max_precision_to_check])
            else:
                print(key + ' does not contain both latlonel and ENU data.')

def updateLatlonelFromENU(data, verbose=True, decimals=8):
    '''
    Given data, this will take the ENU data and use it to update the latlonel data in a deep copy of the original dict.

    Parameters
    ----------
    data : dict
        The dict generated when loading a configuration from a json file.
    '''
    data = copy.deepcopy(data) #So as to not edit the original.
    origin = data['origin']['latlonel'] #Lat Lon Elevation, use for generating ENU from other latlonel values.     

    for mast in range(4):
        for key in ['physical', 'hpol', 'vpol']:
            if numpy.asarray(data['antennas']['ant%i'%mast][key]['enu']).size > 0:
                data['antennas']['ant%i'%mast][key]['latlonel'] = numpy.round(numpy.asarray(pm.enu2geodetic(data['antennas']['ant%i'%mast][key]['enu'][0],data['antennas']['ant%i'%mast][key]['enu'][1],data['antennas']['ant%i'%mast][key]['enu'][2],origin[0],origin[1],origin[2])),decimals=decimals)
            else:
                if verbose:
                    print(key + ' does not contain ENU data for mast %i %s.'%(mast, key))
    return data

def updateENUFromLatlonel(data, verbose=True):
    '''
    Given data, this will take the latlonel data and use it to update the ENU data in a deep copy of the original dict.

    Parameters
    ----------
    data : dict
        The dict generated when loading a configuration from a json file.
    '''
    data = copy.deepcopy(data) #So as to not edit the original.
    origin = data['origin']['latlonel'] #Lat Lon Elevation, use for generating ENU from other latlonel values.     

    for mast in range(4):
        for key in ['physical', 'hpol', 'vpol']:
            if numpy.asarray(data['antennas']['ant%i'%mast][key]['latlonel']).size > 0:
                data['antennas']['ant%i'%mast][key]['enu'] = numpy.asarray(pm.geodetic2enu(data['antennas']['ant%i'%mast][key]['latlonel'][0],data['antennas']['ant%i'%mast][key]['latlonel'][1],data['antennas']['ant%i'%mast][key]['latlonel'][2],origin[0],origin[1],origin[2]))
            else:
                if verbose:
                    print(key + ' does not contain ENU data for mast %i %s.'%(mast, key))
    return data

def configReader(json_path, return_mode='enu', check=True, verbose=True, return_description=False):
    '''
    This will read in the custom config files and return them in a usable format that allows the json files to
    interface readily with existing analysis code.

    This involves producing dictionaries for each of:
        origin : array-like
            The latitude, longitude, and elevation (in m) contained within a 3 element array-like object: [lat, lon, el]
        antennas_physical : dict
            A dictionary containing the ENU positions of the antennas (ENU with regards to the origin).  Example below:
            {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}
        antennas_phase_hpol : dict
            A dictionary containing the ENU positions of the hpol antennas (ENU with regards to the origin).  Phase here
            implies these positions are derived from some calibration, as opposed to the physical locations which are
            likely measured.  Example below:
            {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}            
        antennas_phase_vpol : dict
            A dictionary containing the ENU positions of the vpol antennas (ENU with regards to the origin).  Phase here
            implies these positions are derived from some calibration, as opposed to the physical locations which are
            likely measured.  Example below:
            {0 : [0.0, 0.0, 0.0], 1 : [-38.7, -18.6, 23.9], 2 : [-5.4, -45.5, -5.7], 3 : [-30.9, -44.9, 10.1]}
        cable_delays : dict
            A dictionary containing the hpol and vpol cable delays.  Example below:
            cable_delays =  {   'hpol': [423.37836156, 428.43979143, 415.47714969, 423.58803498], \
                                'vpol': [428.59277751, 430.16685915, 423.56765695, 423.50469285]}

    These dictionaries are normally stored in plain text the the beacon analysis tools/info.py script, which loads
    them when give a specific deploy index.  By using a "deploy_index" corresponding to a json file path, tools/info.py
    will instead call this script to obtain the required calibration information.


    Parameters
    ----------
    json_path : str
        The path (including filename and extension) of the input file to be read.
    return_mode : str
        Either 'enu' (default) or 'latlonel'.  This will select whether to return the ENU coordinates for 
        antennas_physical, antennas_phase_hpol, and antennas_phase_vpol, or their lat/lon/el coordinates.  Origin is
        always given in lat/lon/el.  The default behaviour is for ENU, as this is what is used throughout the code.
        Only return lat/lon/el if you know what you are doing. 
    check : bool
        If True then this will run checkConfigConsistency.  Default is False.
    '''
    with open(json_path) as json_file:
        data = json.load(json_file)

    description = str(data['description'])
    if verbose:
        print('Calibration Description:')
        print(description)


    origin = data['origin']['latlonel'] #Lat Lon Elevation, use for generating ENU from other latlonel values.     
    antennas_physical = {}
    antennas_phase_hpol = {}
    antennas_phase_vpol = {}
    cable_delays = {'hpol' : [0.0 , 0.0 , 0.0 , 0.0],'vpol' : [0.0 , 0.0 , 0.0 , 0.0]}

    if check:
        print('Check before updating values:')
        checkConfigConsistency(data)
    if return_mode == 'latlonel':
        print('WARNING!!! Loading antenna positions as latlonel, which is not the default behaviour and should be done with caution.')
        print('Done using the ENU data.')
        data = updateLatlonelFromENU(data, verbose=verbose)
    elif return_mode == 'enu':
        data = updateENUFromLatlonel(data, verbose=verbose)
    if check:
        print('Check after updating values:')
        checkConfigConsistency(data)

    for mast in range(4):
        antennas_physical[mast] = data['antennas']['ant%i'%mast]['physical'][return_mode]
        antennas_phase_hpol[mast] = data['antennas']['ant%i'%mast]['hpol'][return_mode]
        antennas_phase_vpol[mast] = data['antennas']['ant%i'%mast]['vpol'][return_mode]
        cable_delays['hpol'][mast] = data['antennas']['ant%i'%mast]['hpol']['cable_delay']
        cable_delays['vpol'][mast] = data['antennas']['ant%i'%mast]['vpol']['cable_delay']
    if return_description == True:
        return origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description
    else:
        return origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays

def generateConfigFromDeployIndex(outpath, deploy_index, description=None):
    '''
    Given a deploy_index that has traditionally defined meta data, this will pull that data and write it to a json file.

    Parameters
    ----------
    outpath : str
        The directory you want the auto-named config file to be placed.  Expected to end with a /.  
    deploy_index : int
        The deploy_index you want to create the calibraiton file for.  This corresponds to the info.py script.
    '''
    if type(deploy_index) == int:
        cable_delays = info.loadCableDelays(deploy_index=deploy_index)
        origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=deploy_index)
        if outpath[-1] != '/':
            outpath += '/'
        json_path = outpath + 'config_deploy_index_%i.json'%deploy_index
        if description is None:
            description = "An autogenerated calibration file for using the existing values for deploy_index = %i"%deploy_index
        configWriter(json_path, origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description=description,update_latlonel=True,force_write=True)
    else:
        print('This function only handles legacy integer deploy_index values.')


if __name__ == '__main__':
    if False:
        sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
        config_file_dir = os.environ['BEACON_ANALYSIS_DIR'] + 'config/'


        print('Running config_reader.py in debug mode:') #As apposed to loading functions for use elsewhere. 
        test_file = config_file_dir + 'deploy_30.json'
        origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays = configReader(test_file)
    
    if True:
        json_path = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'config','rtk-gps-day3-june22-2021.json')
        configWriterFromLatLonElJSON(json_path,verbose=True)
    if False:
        #Can be used to convert legacy calibrations to json.
        sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
        import tools.info as info
        generateConfigFromDeployIndex(config_file_dir, 29)

    #The below was meant to update a calibration file generated with only enu data.     
    #configWriter(config_file_dir + 'deploy_30_updated.json', origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, update_latlonel=True, description="deploy_index=30 with updated coordinates.",force_write=True)

    if False:
        origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays = loadSampleData()

        test_write_file = config_file_dir + 'deploy_test_A.json'
        output_filename = configWriter(test_write_file, origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description="Testing the configuration writer.",force_write=True)
        auto_data = configReader(output_filename)
    else:
        print('Testing of configWriter disabled.')