import json
import sys
import os

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
config_file_dir = os.environ['BEACON_ANALYSIS_DIR'] + 'config/'

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

def configWriter(json_path, origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description="",force_write=True):
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
            file_root = '_'.join(json_path.split('_')[:-1])
            file_end = json_path.split('_')[-1]

            if file_end.replace('.json','').isdigit():
                index = int(file_end.replace('.json',''))
            else:
                file_root = file_root + '_%s'%(file_end.replace('.json',''))
                file_end = '0'
                index = 0

            while os.path.exists(file_root + '_%i.json'%index):
                index += 1
            json_path = file_root + '_%i.json'%index


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

        with open(json_path, 'w') as outfile:
            outfile.write(json.dumps(data, indent=4, sort_keys=False))
        return json_path
    else:
        print('No output file created.')

def configReader(json_path, return_mode='enu'):
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
    '''
    with open(json_path) as json_file:
        data = json.load(json_file)

    print('Calibration Description:')
    print(data['description'])
    origin = data['origin']['latlonel'] #Lat Lon Elevation, use for generating ENU from other latlonel values.     
    antennas_physical = {}
    antennas_phase_hpol = {}
    antennas_phase_vpol = {}
    cable_delays = {'hpol' : [0.0 , 0.0 , 0.0 , 0.0],'vpol' : [0.0 , 0.0 , 0.0 , 0.0]}

    for mast in range(4):
        antennas_physical[mast] = data['antennas']['ant%i'%mast]['physical']['enu']
        antennas_phase_hpol[mast] = data['antennas']['ant%i'%mast]['hpol']['enu']
        antennas_phase_vpol[mast] = data['antennas']['ant%i'%mast]['vpol']['enu']
        cable_delays['hpol'][mast] = data['antennas']['ant%i'%mast]['hpol']['cable_delay']
        cable_delays['vpol'][mast] = data['antennas']['ant%i'%mast]['vpol']['cable_delay']

    return origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays



if __name__ == '__main__':
    print('Running config_reader.py in debug mode:') #As apposed to loading functions for use elsewhere. 
    test_file = config_file_dir + 'deploy_30.json'
    manual_data = configReader(test_file)

    origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays = loadSampleData()

    test_write_file = config_file_dir + 'deploy_A.json'
    output_filename = configWriter(test_write_file, origin, antennas_physical, antennas_phase_hpol, antennas_phase_vpol, cable_delays, description="Testing the configuration writer.",force_write=True)
    auto_data = configReader(output_filename)