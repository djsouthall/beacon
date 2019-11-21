'''
This file is intended to hold any organizational information to centralize it to be imported by other scripts.

This file itself may not be the most organized by hopefully it allows others to stay moreso. 

A 'deploy_index' is used for many locations to denote the specific configuration of antennas and pulsers depending
upon which deployment you are in.

You can set what you want the default deployment to be by changing default_deploy at the top of this file.

deploy_index = 0:
    Before Oct 2019
deploy_index = 1:
    After Oct 2019
'''
import sys
import os
import inspect
import numpy
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.field_fox as ff
import pymap3d as pm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.interpolate
default_deploy = 1 #The deployment time to use as the default.


def pulserRuns():
    '''
    Returns
    -------
    pulser_runs : numpy.ndarray of ints
        This is the list of known pulser runs as determined by the matching_times.py script.
    '''
    pulser_runs = numpy.array([734,735,736,737,739,740,746,747,757,757,762,763,764,766,767,768,769,770,781,782,783,784,785,786,787,788,789,790,792,793,1504,1506,1507,1508,1509,15011]) 
    
    return pulser_runs

def loadAntennaZeroLocation(deploy_index=default_deploy):
    '''
    Loads antenna 0's location (which use used as the station location).
    Loads both the latitude, longtidue, elevation
    '''
    if deploy_index == 0:
        A0Location = (37.5893,-118.2381,3894.12)#latitude,longtidue,elevation  #ELEVATION GIVEN FROM GOOGLE EARTH given in m
    elif deploy_index == 1:
        A0Location = (37.589310, -118.237621, 3875.53)#latitude,longtidue,elevation #ELEVATION FROM GOOGLE EARTH given in m  
    return A0Location

def loadAntennaLocationsENU(deploy_index=default_deploy):
    '''
    Loads the antenna locations and phase locations as best they are known.
    These are given in ENU relative to Antenna 0.
    '''
    if deploy_index == 0:
        antennas_physical   = {0:(0.0,0.0,0.0),1:(-6.039,-1.618,2.275),2:(-1.272,-10.362,1.282),3:(3.411,-11.897,-0.432)} #ORIGINAL
        '''
        #These were determined using only run 793
        antennas_phase_hpol = {0:(  -0.02557475,   0.03116954,   0.09699316),1:(-6.07239516,  -1.57654064,   2.40102979),2:(-1.03349923, -10.66185761,   0.41323144),3:( 3.0254727 , -11.41386618,   1.08350273)}#ADJUSTED HPOL
        antennas_phase_vpol = {0:(-0.3113139 ,   0.37988811,   1.22224369),1:(-5.87779214,  -1.8179266 ,   1.68175401),2:(-1.57186065,  -9.98385335,   2.45102724),3:( 3.79236323, -12.37305718,  -1.80125484)}#ADJUSTED VPOL
        '''
        ''
        #These were determined in first attempt with day 5 and day 6 data with no bounds on phase positons.
        antennas_phase_hpol = {0:( -1.05036701,  -2.83990607,   5.7301439) , 1:(-5.04455409,   1.80238432,  -3.37157069), 2:(-0.70469931,  -9.35762227,  -1.46880603),  3:( 0.62819922, -18.85449124,  14.09627911)}#ADJUSTED HPOL
        antennas_phase_vpol = {0:( -1.97517555,  -4.78830899,  10.53874329), 1:( -5.26414199,   0.06191184,  -1.6073464), 2:( -1.17891238,  -8.69156208,   0.24012179), 3:(  4.23558404, -11.0023696 ,  -4.13418962)}#ADJUSTED VPOL
    elif deploy_index == 1:

        origin = loadAntennaZeroLocation(deploy_index = 1)
        antennas_physical_latlon = {0:origin,1:(37.5892, -118.2380, 3890.77),2:(37.588909, -118.237719, 3881.02),3:(37.5889210, -118.2379850, 3887.42)} #ORIGINAL
        antennas_physical = {}
        for key, location in antennas_physical_latlon.items():
            antennas_physical[key] = pm.geodetic2enu(location[0],location[1],location[2],origin[0],origin[1],origin[2])

        #Errors not currently used.  


    #antennas_phase_vpol = {0 : [0.000000, 0.000000, 0.000000], 1 : [-30.307267, -12.610417, 11.411196], 2 : [-10.464510, -46.217141, -0.229276], 3 : [-31.172820, -42.069610, 14.812669]}
    #antennas_phase_vpol_hesse = {0 : [0.000000, 0.000000, 0.000000], 1 : [0.042770, 0.047592, 0.232313], 2 : [0.069353, 0.034351, 0.297275], 3 : [0.065285, 0.047254, 0.249236]}
    #antennas_phase_hpol = {0 : [0.000000, 0.000000, 0.000000], 1 : [-29.908104, -12.713682, 11.649818], 2 : [-9.956527, -46.119743, 1.003367], 3 : [-31.175775, -41.743273, 15.241286]}
    #antennas_phase_hpol_hesse = {0 : [0.000000, 0.000000, 0.000000], 1 : [0.057717, 0.065283, 0.261789], 2 : [0.080427, 0.046084, 0.282382], 3 : [0.084447, 0.065313, 0.276901]}

    '''
    #These are the ones I got by flipping channel 2
    antennas_phase_hpol = {0 : [0.000000, 0.000000, 0.000000], 1 : [-28.231754, -13.562168, 9.068510], 2 : [-9.735529, -45.985775, -0.088781], 3 : [-31.298949, -42.571000, 13.874803]}                                                          
    antennas_phase_hpol_hesse = {0 : [0.000000, 0.000000, 0.000000], 1 : [0.066406, 0.054628, 0.148034], 2 : [0.033279, 0.044264, 0.117928], 3 : [0.034486, 0.038190, 0.096574]} 
    antennas_phase_vpol = {0 : [0.000000, 0.000000, 0.000000], 1 : [-30.466405, -12.754674, 11.112169], 2 : [-10.727788, -46.128595, -1.850623], 3 : [-31.569865, -42.169063, 13.027888]}                                                        
    antennas_phase_vpol_hesse = {0 : [0.000000, 0.000000, 0.000000], 1 : [0.024673, 0.039216, 0.162412], 2 : [0.041817, 0.028456, 0.215791], 3 : [0.038274, 0.035657, 0.180371]} 
    '''
    #These are the ones that I had prior to 11/21/2019
    antennas_phase_vpol = {0 : [0.000000, 0.000000, 0.000000], 1 : [-30.401760, -12.621386, 10.810656], 2 : [-10.372617, -46.301546, -0.381405], 3 : [-31.261194, -42.193276, 14.412270]}
    antennas_phase_vpol_hesse = {0 : [0.000000, 0.000000, 0.000000], 1 : [0.042055, 0.048239, 0.254382], 2 : [0.069051, 0.034678, 0.330481], 3 : [0.065562, 0.049100, 0.276208]}

    antennas_phase_hpol = {0 : [0.000000, 0.000000, 0.000000], 1 : [-29.979756, -12.405349, 9.822749], 2 : [-9.619360, -46.467924, 2.456841], 3 : [-31.138537, -41.778211, 14.606688]}
    antennas_phase_hpol_hesse = {0 : [0.000000, 0.000000, 0.000000], 1 : [0.051475, 0.064251, 0.278799], 2 : [0.078516, 0.049043, 0.293481], 3 : [0.085002, 0.073460, 0.303864]}
    return antennas_physical, antennas_phase_hpol, antennas_phase_vpol

def loadCableDelays(return_raw=False):
    '''
    This are calculated using group_delay.py via the group delay.  They correspond to the length of the LMR400
    cable that extends from the observatory to the antennas and accounts for the majority of systematic delay
    between signals.  This should be accounted for in interferometric uses.
    '''

    cable_delays =  {'hpol': numpy.array([423.37836156, 428.43979143, 415.47714969, 423.58803498]), \
                     'vpol': numpy.array([428.59277751, 430.16685915, 423.56765695, 423.50469285])}

    
    if return_raw == False:
        min_delay = min((min(cable_delays['hpol']),min(cable_delays['vpol'])))
        cable_delays['hpol'] -= min_delay
        cable_delays['vpol'] -= min_delay

    return cable_delays


def loadPulserPolarizations():
    '''
    Loads the polarizations used in each pulsing run.  Options are hpol, vpol, or both

    This won't make sense for data taken in the October 2019 pulsing run.   Will need higher
    resolution, i.e. time of day spans rather than run labels. 
    '''
    pulser_pol = {}

    #Trip 1
    #Day 1
    #Site 1 37.4671° N, 117.7525° W
    pulser_pol['run734'] = 'vpol'
    pulser_pol['run735'] = 'vpol'
    pulser_pol['run736'] = 'both'
    pulser_pol['run737'] = 'hpol'
    #Site 2 37° 34' 30.8" N, 117° 54' 31.7" W
    pulser_pol['run739'] = 'both'
    pulser_pol['run740'] = 'hpol'

    #Day 2 37° 34' 30.8" N 117° 54' 31.7" W
    pulser_pol['run746'] = 'both'
    pulser_pol['run747'] = 'vpol'

    #Day 3 37° 35’ 54.82” N 117° 59’ 37.97” W
    pulser_pol['run756'] = 'hpol'
    pulser_pol['run757'] = 'hpol'

    #Day 4
    #Site 1 37° 43' 36.40" N 118° 2' 3.40" W
    pulser_pol['run762'] = 'vpol'
    pulser_pol['run763'] = 'vpol'
    pulser_pol['run764'] = 'vpol'
    #Site2 37° 25' 32.85" N 117° 37' 57.55" W
    pulser_pol['run766'] = 'vpol'
    pulser_pol['run767'] = 'vpol'
    pulser_pol['run768'] = 'vpol'
    pulser_pol['run769'] = 'vpol'
    pulser_pol['run770'] = 'hpol'

    #Day 5 37° 35' 9.27" N 118° 14' 0.73" W
    pulser_pol['run781'] = 'hpol'
    pulser_pol['run782'] = 'hpol'
    pulser_pol['run783'] = 'hpol'
    pulser_pol['run784'] = 'hpol'
    pulser_pol['run785'] = 'hpol'
    pulser_pol['run786'] = 'hpol'
    pulser_pol['run787'] = 'hpol'
    pulser_pol['run788'] = 'hpol'
    pulser_pol['run789'] = 'hpol'
    pulser_pol['run790'] = 'vpol'

    #Day 6 37° 35.166' N 118° 13.990' W 
    pulser_pol['run792'] = 'vpol'
    pulser_pol['run793'] = 'vpol'

    #Trip 2
    #Site 1a 37.5859361° N 118.233841 W 
    pulser_pol['run1506'] = 'hpol'
    pulser_pol['run1507'] = 'hpol'

    #Site 2 37.58568583° N 118.225942 W 
    pulser_pol['run1508'] = 'both'
    pulser_pol['run1509'] = 'both'

    #Site 3 37.592001861° N 118.2354480278 W 
    pulser_pol['run1511'] = 'both'
    
    
    return pulser_pol   

def loadPulserLocations():
    '''
    Loads the latitude,longtidue,elevation locations of the antennas.
    See loadPulserLocationsENU for these locations converted to
    be relative to antenna 0.

    These are repeated if that pulser is used for multiply runs. 
    '''
    pulser_locations = {}

    #Day 1
    #Site 1 37.4671° N, 117.7525° W
    pulser_locations['run734'] = (37.4671,-117.7525,1763.0)
    pulser_locations['run735'] = (37.4671,-117.7525,1763.0)
    pulser_locations['run736'] = (37.4671,-117.7525,1763.0)
    pulser_locations['run737'] = (37.4671,-117.7525,1763.0)
    #Site 2 37° 34' 30.8" N, 117° 54' 31.7" W
    pulser_locations['run739'] = (37.575225,-117.908807,1646.0)
    pulser_locations['run740'] = (37.575225,-117.908807,1646.0)

    #Day 2 37° 34' 30.8" N 117° 54' 31.7" W
    pulser_locations['run746'] = (37.575225,-117.908807,1646.0)
    pulser_locations['run747'] = (37.575225,-117.908807,1646.0)

    #Day 3 37° 35’ 54.82” N 117° 59’ 37.97” W
    pulser_locations['run756'] = (37.598554,-117.993874,1501.0)
    pulser_locations['run757'] = (37.598554,-117.993874,1501.0)

    #Day 4
    #Site 1 37° 43' 36.40" N 118° 2' 3.40" W
    pulser_locations['run762'] = (37.726735,-118.034261,1542.0)
    pulser_locations['run763'] = (37.726735,-118.034261,1542.0)
    pulser_locations['run764'] = (37.726735,-118.034261,1542.0)
    #Site2 37° 25' 32.85" N 117° 37' 57.55" W
    pulser_locations['run766'] = (37.425788,-117.632653,2021.0)
    pulser_locations['run767'] = (37.425788,-117.632653,2021.0)
    pulser_locations['run768'] = (37.425788,-117.632653,2021.0)
    pulser_locations['run769'] = (37.425788,-117.632653,2021.0)
    pulser_locations['run770'] = (37.425788,-117.632653,2021.0)

    #Day 5 37° 35' 9.27" N 118° 14' 0.73" W
    pulser_locations['run781'] = (37.585912,-118.233535,3789)
    pulser_locations['run782'] = (37.585912,-118.233535,3789)
    pulser_locations['run783'] = (37.585912,-118.233535,3789)
    pulser_locations['run784'] = (37.585912,-118.233535,3789)
    pulser_locations['run785'] = (37.585912,-118.233535,3789)
    pulser_locations['run786'] = (37.585912,-118.233535,3789)
    pulser_locations['run787'] = (37.585912,-118.233535,3789)
    pulser_locations['run788'] = (37.585912,-118.233535,3789)
    pulser_locations['run789'] = (37.585912,-118.233535,3789)
    pulser_locations['run790'] = (37.585912,-118.233535,3789)

    #Day 6 37° 35.166' N 118° 13.990' W 
    pulser_locations['run792'] = (37.5861,-118.2332,3779.52)
    pulser_locations['run793'] = (37.5861,-118.2332,3779.52)


    #Trip 2
    #Site 1  37.5859361 N 118.233918056 W  (37.5859361, -118.233918056)
    #Alt: 3762.9m (GPS)  3789.32 m (MSL) Google Earth: Alt: 3796.284
    pulser_locations['run1504'] = (37.5859361, -118.233918056,3796.284)

    #Site 1a 37.58595472° N 118.233841 W 
    #Alt: 3763.1m (GPS)  3789.53 m (MSL) Google Earth: 3794.76
    pulser_locations['run1506'] = (37.58595472, -118.233841,3794.76)
    pulser_locations['run1507'] = (37.58595472, -118.233841,3794.76)

    #Site 2 37.58568583° N 118.225942 W 
    #Alt: 3690.70m (GPS)  3717.04m (MSL) Google Earth: 3729.228
    pulser_locations['run1508'] = (37.58568583, -118.225942,3729.228)
    pulser_locations['run1509'] = (37.58568583, -118.225942,3729.228)

    #Site 3 37.592001861° N 118.2354480278 W 
    #Alt: 3806.25m (GPS)  3832.55m (MSL) Google Earth: 3827.6784
    pulser_locations['run1511'] = (37.592001861, -118.2354480278,3827.6784)

    return pulser_locations    

def loadPulserLocationsENU():
    '''
    Loads the locations of the antennas converted to
    be relative to antenna 0.

    These are repeated if that pulser is used for multiple runs. 

    This is depricated and does not all for the antennas to have different 
    phase centers. loadPulserPhaseLocationsENU is better.
    '''
    pulser_locations_ENU = {}
    pulser_locations = loadPulserLocations()

    origin = loadAntennaZeroLocation()
    for key, location in pulser_locations.items():
        pulser_locations_ENU[key] = pm.geodetic2enu(location[0],location[1],location[2],origin[0],origin[1],origin[2])
    return pulser_locations_ENU

def loadPulserPhaseLocationsENU():
    '''
    Loads the locations of the antennas converted to
    be relative to antenna 0.

    These are repeated if that pulser is used for multiple runs.  

    The output will be a dictionary with keys 'physical','hpol', and 'vpol'
    corresponding to the best known physical locations, and current best fit
    for phase centers.
    '''
    pulser_locations_ENU = {}
    pulser_locations = loadPulserLocations()

    pulser_locations_ENU['physical'] = {}

    origin = loadAntennaZeroLocation()
    for key, location in pulser_locations.items():
        pulser_locations_ENU['physical'][key] = pm.geodetic2enu(location[0],location[1],location[2],origin[0],origin[1],origin[2])

    #pulser_locations_ENU['vpol'] = {'run1507':[275.708465, -372.224572, -82.673435], 'run1509':[1027.431897, -492.547030, -155.078725], 'run1511':[178.030733, 331.491744, -39.509867]}
    #pulser_locations_ENU['vpol_hesse_error'] = {'run1507':[1.106217, 1.246325, 2.503419], 'run1509':[5.552711, 2.469107, 5.305297], 'run1511':[1.324100, 2.146287, 2.310413]}
    
    '''
    #These are the ones I got by flipping channel 2
    pulser_locations_ENU['hpol'] = {'run1507':[253.799742, -355.808099, -117.460144], 'run1509':[1137.872525, -534.620695, -189.860634], 'run1511':[117.111896, 303.926298, -74.299903]}                                                         
    pulser_locations_ENU['hpol_hesse_error'] = {'run1507':[0.501879, 0.520719, 1.090585], 'run1509':[2.701943, 1.187494, 3.423798], 'run1511':[2.560162, 6.026819, 3.490432]}
    pulser_locations_ENU['vpol'] = {'run1507':[263.002766, -355.657916, -80.473571], 'run1509':[1067.737570, -511.201739, -152.867512], 'run1511':[179.261666, 338.184460, -37.316588]}
    pulser_locations_ENU['vpol_hesse_error'] = {'run1507':[0.848562, 0.978776, 1.976002], 'run1509':[2.842634, 1.265834, 3.353987], 'run1511':[1.327852, 2.182353, 2.249852]}
    '''

    #These are the ones I got prior to 11/21/2019
    pulser_locations_ENU['hpol'] = {'run1507':[259.417378, -353.989882, -84.468321], 'run1509':[1129.874543, -528.948053, -156.869667], 'run1511':[189.018118, 338.618832, -41.302878]}
    pulser_locations_ENU['hpol_hesse_error'] = {'run1507':[1.335408, 1.471356, 3.081678], 'run1509':[9.068532, 3.966986, 8.780782], 'run1511':[2.157532, 3.556144, 4.533434]}
    pulser_locations_ENU['vpol'] = {'run1507':[271.225702, -362.785806, -75.741218], 'run1509':[1063.720642, -504.948969, -148.130744], 'run1511':[181.798419, 331.680671, -32.583995]}
    pulser_locations_ENU['vpol_hesse_error'] = {'run1507':[1.080508, 1.216558, 2.517439], 'run1509':[5.776634, 2.552221, 5.590785], 'run1511':[1.319455, 2.150750, 2.325213]}

    #pulser_locations_ENU['hpol'] = {'run1507':[265.441241, -366.161638, -91.314401], 'run1509':[1065.399706, -508.295499, -163.706607], 'run1511':[178.535899, 344.684624, -48.158593]}
    #pulser_locations_ENU['hpol_hesse_error'] = {'run1507':[1.384200, 1.534385, 3.015226], 'run1509':[8.512122, 3.788872, 7.223959], 'run1511':[2.282271, 3.801014, 4.190135]}

    return pulser_locations_ENU



def plotStationAndPulsers(plot_phase=False):
    '''
    Currently only intended to plot the most recent station with the three pulsers that we used for it.
    '''
    antennas_physical, antennas_phase_hpol, antennas_phase_vpol = loadAntennaLocationsENU(deploy_index=1)

    colors = ['b','g','r','c']
    pulser_colors = ['m','y','k']

    fig = plt.figure()
    fig.canvas.set_window_title('Antenna Locations')
    ax = fig.add_subplot(111, projection='3d')

    for i, a in antennas_physical.items():
        ax.scatter(a[0], a[1], a[2], marker='o',color=colors[i],label='Physical %i'%i,alpha=0.8)

    if plot_phase == True:
        for i, a in antennas_phase_hpol.items():
            ax.plot([antennas_physical[i][0],antennas_phase_hpol[i][0]],[antennas_physical[i][1],antennas_phase_hpol[i][1]],[antennas_physical[i][2],antennas_phase_hpol[i][2]],color=colors[i],linestyle='--',alpha=0.5)
            ax.scatter(a[0], a[1], a[2], marker='*',color=colors[i],label='%s Phase Center %i'%('Hpol', i),alpha=0.8)
        for i, a in antennas_phase_vpol.items():
            ax.plot([antennas_physical[i][0],antennas_phase_vpol[i][0]],[antennas_physical[i][1],antennas_phase_vpol[i][1]],[antennas_physical[i][2],antennas_phase_vpol[i][2]],color=colors[i],linestyle='--',alpha=0.5)
            ax.scatter(a[0], a[1], a[2], marker='^',color=colors[i],label='%s Phase Center %i'%('Vpol', i),alpha=0.8)




    pulser_locations = loadPulserPhaseLocationsENU()
    for site, key in enumerate(['run1507','run1509','run1511']):
        site += 1
        ax.scatter(pulser_locations['physical'][key][0], pulser_locations['physical'][key][1], pulser_locations['physical'][key][2], color=pulser_colors[site-1], marker='o',label='Physical Pulser Site %i'%site,alpha=0.8)

    if plot_phase == True:
        ax.plot([pulser_locations['hpol']['run1507'][0],pulser_locations['physical']['run1507'][0]],[pulser_locations['hpol']['run1507'][1],pulser_locations['physical']['run1507'][1]],[pulser_locations['hpol']['run1507'][2],pulser_locations['physical']['run1507'][2]],color=pulser_colors[0],linestyle='--',alpha=0.5)
        ax.scatter( pulser_locations['hpol']['run1507'][0] , pulser_locations['hpol']['run1507'][1] , pulser_locations['hpol']['run1507'][2] , color=pulser_colors[0] , marker='*',alpha=0.8)

        ax.plot([pulser_locations['hpol']['run1509'][0],pulser_locations['physical']['run1509'][0]],[pulser_locations['hpol']['run1509'][1],pulser_locations['physical']['run1509'][1]],[pulser_locations['hpol']['run1509'][2],pulser_locations['physical']['run1509'][2]],color=pulser_colors[1],linestyle='--',alpha=0.5)
        ax.scatter( pulser_locations['hpol']['run1509'][0] , pulser_locations['hpol']['run1509'][1] , pulser_locations['hpol']['run1509'][2] , color=pulser_colors[1] , marker='*',alpha=0.8)

        ax.plot([pulser_locations['hpol']['run1511'][0],pulser_locations['physical']['run1511'][0]],[pulser_locations['hpol']['run1511'][1],pulser_locations['physical']['run1511'][1]],[pulser_locations['hpol']['run1511'][2],pulser_locations['physical']['run1511'][2]],color=pulser_colors[2],linestyle='--',alpha=0.5)
        ax.scatter( pulser_locations['hpol']['run1511'][0] , pulser_locations['hpol']['run1511'][1] , pulser_locations['hpol']['run1511'][2] , color=pulser_colors[2] , marker='*',alpha=0.8)

        ax.plot([pulser_locations['vpol']['run1507'][0],pulser_locations['physical']['run1507'][0]],[pulser_locations['vpol']['run1507'][1],pulser_locations['physical']['run1507'][1]],[pulser_locations['vpol']['run1507'][2],pulser_locations['physical']['run1507'][2]],color=pulser_colors[0],linestyle='--',alpha=0.5)
        ax.scatter( pulser_locations['vpol']['run1507'][0] , pulser_locations['vpol']['run1507'][1] , pulser_locations['vpol']['run1507'][2] , color=pulser_colors[0] , marker='^',alpha=0.8)

        ax.plot([pulser_locations['vpol']['run1509'][0],pulser_locations['physical']['run1509'][0]],[pulser_locations['vpol']['run1509'][1],pulser_locations['physical']['run1509'][1]],[pulser_locations['vpol']['run1509'][2],pulser_locations['physical']['run1509'][2]],color=pulser_colors[1],linestyle='--',alpha=0.5)
        ax.scatter( pulser_locations['vpol']['run1509'][0] , pulser_locations['vpol']['run1509'][1] , pulser_locations['vpol']['run1509'][2] , color=pulser_colors[1] , marker='^',alpha=0.8)

        ax.plot([pulser_locations['vpol']['run1511'][0],pulser_locations['physical']['run1511'][0]],[pulser_locations['vpol']['run1511'][1],pulser_locations['physical']['run1511'][1]],[pulser_locations['vpol']['run1511'][2],pulser_locations['physical']['run1511'][2]],color=pulser_colors[2],linestyle='--',alpha=0.5)
        ax.scatter( pulser_locations['vpol']['run1511'][0] , pulser_locations['vpol']['run1511'][1] , pulser_locations['vpol']['run1511'][2] , color=pulser_colors[2] , marker='^',alpha=0.8)


    ax.set_xlabel('E (m)')
    ax.set_ylabel('N (m)')
    ax.set_zlabel('Relative Elevation (m)')
    plt.legend()



def loadClockRates():
    '''
    Loads a dictionary containing the known clock rates as calculated using the clock_correct.py scipt.
    These are given in Hz.
    '''
    clock_rates = {
    'run782'    :31249808.91966798,
    'run783'    :31249808.948130235,
    'run784'    :31249809.35802664,
    'run785'    :31249809.82779526,
    'run788'    :31249807.839061476,
    'run789'    :31249809.895620257,
    'run792'    :31249812.04283368,
    'run793'    :31249809.22371152,
    'run1506'   :31249822.962542757,
    'run1507'   :31249815.193117745,
    'run1508'   :31249811.59632718,
    'run1509'   :31249810.666976035,
    'run1511'   :31249840.967325963}
    clock_rates['default'] = numpy.mean([v for key,v in clock_rates.items()])
    return clock_rates


def loadIgnorableEventids():
    '''
    This function loads dictionaries containing eventids that one may want to ignore.

    For instance eventids that are known pulser event ids but ones you want to ignore
    when making a template. 
    '''

    ignore_eventids = {}
    ignore_eventids['run793'] = numpy.array([ 96607,  96657,  96820,  96875,  98125,  98588,  99208, 100531,\
                           101328, 101470, 101616, 101640, 101667, 102159, 102326, 102625,\
                           103235, 103646, 103842, 103895, 103977, 104118, 104545, 105226,\
                           105695, 105999, 106227, 106476, 106622, 106754, 106786, 106813,\
                           106845, 107022, 107814, 108162, 110074, 110534, 110858, 111098,\
                           111197, 111311, 111542, 111902, 111941, 112675, 112713, 112864,\
                           112887, 113062, 113194, 113392, 113476, 113957, 114069, 114084,\
                           114295, 114719, 114738, 114755, 114942, 115055, 115413, 115442,\
                           115465, 115491, 115612, 116065])
    #For some reason there is a shift in arrival times that makes these different than later events?
    ignore_eventids['run1509'] = numpy.array([  2473, 2475, 2477, 2479, 2481, 2483, 2485, 2487, 2489, 2491, 2493,\
                                               2495, 2497, 2499, 2501, 2503, 2505, 2507, 2509, 2511, 2513, 2515,\
                                               2517, 2519, 2521, 2523, 2525, 2527, 2529, 2531, 2533, 2535, 2537,\
                                               2539, 2541, 2543, 2545, 2547, 2549, 2551, 2553, 2555, 2557, 2559,\
                                               2561, 2563, 2565, 2567, 2569, 2571, 2573, 2575, 2577, 2579, 2581,\
                                               2583, 2585, 2587, 2589, 2591, 2593, 2595, 2597, 2599, 2601, 2603,\
                                               2605, 2607, 2609, 2611, 2613, 2615, 2617, 2619, 2621, 2623, 2625,\
                                               2627, 2629, 2631, 2633, 2635, 2637, 2639, 2641, 2643, 2645, 2647,\
                                               2649, 2651, 2653, 2655, 2657, 2659, 2661, 2663, 2665, 2667, 2669,\
                                               2671, 2673, 2675, 2677, 2679, 2681, 2683, 2685, 2687, 2689, 2691,\
                                               2693, 2695, 2697, 2699, 2701, 2703, 2705, 2707, 2709, 2711, 2713,\
                                               2715, 2717, 2719, 2721, 2723, 2725, 2727, 2729, 2731, 2733, 2735,\
                                               2737, 2739, 2741, 2743, 2745, 2747, 2749, 2751, 2753, 2755, 2757,\
                                               2759, 2761, 2763, 2765, 2767, 2769, 2771, 2773, 2775, 2777, 2779,\
                                               2781, 2783, 2785, 2787, 2789, 2791, 2793, 2795, 2797, 2799, 2801,\
                                               2803, 2805, 2807, 2809, 2811, 2813, 2815, 2817, 2819, 2821, 2823,\
                                               2825, 2827, 2829, 2831, 2833, 2835, 2837, 2839, 2841, 2843, 2845,\
                                               2847, 2849, 2851, 2853, 2855, 2857, 2859])
    ignore_eventids['run1511'] = numpy.array([1052, 1162, 1198, 1224, 1230, 1232, 1242, 1244, 1262, 1264, 1286,\
                                        1310, 1328, 1330, 1346, 1350, 1370, 1372, 1382, 1416, 1420, 1426,\
                                        1428, 1434, 1442, 1458, 1462, 1464, 1470, 1482, 1492, 1494, 1502,\
                                        1504, 1506, 1528, 1536, 1538, 1574, 1592, 1614, 1636, 1644, 1654,\
                                        1668, 1670, 1672, 1680, 1684, 1686, 1694, 1698, 1710, 1726, 1734,\
                                        1736, 1738, 1742, 1774, 1776, 1782, 1786, 1794, 1798, 1804, 1826,\
                                        1832, 1838, 1848, 1860, 1862, 1868, 1876, 1880, 1882, 1888, 1890,\
                                        1892, 1894, 1908, 1912, 1950, 1960, 1962, 1964, 1974, 1992, 2004,\
                                        2012, 2014, 2018, 2030, 2034, 2048, 2050, 2054, 2058, 2066, 2068,\
                                        2070, 2076, 2078, 2096, 2100, 2108, 2120, 2124, 2126, 2134, 2146,\
                                        2152, 2162, 2176, 2182, 2184, 2186, 2196, 2200, 2202, 2216, 2220,\
                                        2230, 2236, 2242, 2248, 2258, 2266, 2270, 2280, 2302, 2304, 2316,\
                                        2330, 2348, 2352, 2356, 2360, 2362, 2368, 2372, 2376, 2388, 2390,\
                                        2392, 2394, 2396, 2398, 2400, 2406, 2408, 2416, 2426, 2428, 2442,\
                                        2452, 2458, 2466, 2470, 2472, 2476, 2484, 2486, 2498, 2504, 2516,\
                                        2530, 2532, 2534, 2542, 2546, 2570, 2592, 2594, 2596, 2598, 2600,\
                                        2602, 2608, 2620, 2622, 2624, 2636, 2638, 2650, 2660, 2664, 2670,\
                                        2686, 2692, 2698, 2700, 2702, 2706, 2716, 2718, 2740, 2750, 2754])
    ignore_eventids['run1507'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1507_pulser_ignoreids.csv',delimiter=',').astype(int)
    ignore_eventids['run1509'] = numpy.sort(numpy.append(ignore_eventids['run1509'],numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1509_pulser_ignoreids.csv',delimiter=',').astype(int)))
    ignore_eventids['run1511'] = numpy.sort(numpy.append(ignore_eventids['run1511'],numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1511_pulser_ignoreids.csv',delimiter=',').astype(int)))
    return ignore_eventids

def loadPulserEventids(remove_ignored=False):
    '''
    Loads a dictionary containing the known eventids for pulsers.

    If subsets of runs are known to be different, this dictionary may contain
    an additional layer of keys seperating the events.  The code that uses this should
    known how to handle this. 
    '''
    known_pulser_ids = {}
    known_pulser_ids['run781'] = numpy.array([])
    known_pulser_ids['run782'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run782_pulser_eventids.csv',delimiter=',').astype(int)
    known_pulser_ids['run783'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run783_pulser_eventids.csv',delimiter=',').astype(int)
    known_pulser_ids['run784'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run784_pulser_eventids.csv',delimiter=',').astype(int)
    known_pulser_ids['run785'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run785_pulser_eventids.csv',delimiter=',').astype(int)
    known_pulser_ids['run786'] = numpy.array([])
    known_pulser_ids['run787'] = numpy.array([])
    known_pulser_ids['run788'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run788_pulser_eventids.csv',delimiter=',').astype(int)
    known_pulser_ids['run789'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run789_pulser_eventids.csv',delimiter=',').astype(int)

    known_pulser_ids['run792'] = \
        numpy.array([115156, 115228, 115256, 115276, 115283, 115315, 115330, 115371,\
        115447, 115612, 115872, 116230, 116262, 116462, 116473, 116479,\
        116486, 116511, 116524, 116603, 116619, 116624, 116633, 116760,\
        116790, 116816, 117026, 117050, 117175, 117195, 117237, 117247,\
        117258, 117315, 117378, 117540, 117837, 117858, 117874, 117933,\
        117949, 118116, 118139, 118167, 118208, 118219, 118227, 118241,\
        118256, 118267, 118295, 118364, 118423, 118461, 118497, 118518,\
        118644, 118662, 118676, 118685, 118719, 118752, 118856, 118872,\
        118889, 118908, 118930, 118946, 118994, 119038, 119053, 119064,\
        119070, 119094, 119150, 119161, 119177, 119208, 119223, 119304,\
        119315, 119339, 119346, 119371, 119390, 119401, 119408, 119414,\
        119431, 119434, 119458, 119472, 119478, 119508, 119517, 119555,\
        119578, 119598, 119629, 119636, 119648, 119660, 119671, 119844,\
        120009, 120107, 120115, 120202, 120225, 120241, 120249, 120263,\
        120276, 120281, 120292, 120374, 120587, 120607, 120613, 120628,\
        120632, 120905, 120910, 120916, 120925, 120941, 121019, 121081,\
        121170, 121318, 121382, 121460, 121489, 121510, 121725, 121736,\
        121741, 121751, 121765, 121769, 121803, 121876, 121981, 122001,\
        122014, 122021, 122053, 122073, 122093, 122166, 122293, 122311,\
        122403, 122455, 122508, 122551, 122560, 122579, 122723, 122761,\
        122797])
    known_pulser_ids['run793'] = \
        numpy.array([    96607,  96632,  96657,  96684,  96762,  96820,  96875,  96962,\
        97532,  97550,  97583,  97623,  97636,  97661,  97681,  97698,\
        97720,  97739,  97761,  97782,  97803,  97824,  97846,  97876,\
        97932,  97954,  97979,  98006,  98030,  98050,  98075,  98125,\
        98148,  98163,  98190,  98207,  98277,  98431,  98450,  98472,\
        98507,  98545,  98561,  98577,  98587,  98588,  98631,  98657,\
        98674,  98687,  98707,  98731,  98799,  98815,  99040,  99086,\
        99110,  99158,  99208,  99227,  99245,  99264,  99288,  99309,\
        99340,  99353,  99375,  99398,  99423,  99440,  99454,  99477,\
        99493,  99513,  99530,  99548,  99911,  99942,  99951,  99985,\
        100002, 100019, 100035, 100055, 100073, 100096, 100114, 100153,\
        100189, 100294, 100424, 100442, 100531, 100591, 100748, 100767,\
        100899, 100979, 101000, 101011, 101025, 101129, 101146, 101161,\
        101177, 101191, 101212, 101227, 101261, 101281, 101297, 101311,\
        101328, 101363, 101378, 101457, 101470, 101485, 101500, 101527,\
        101540, 101556, 101578, 101616, 101640, 101667, 101736, 101760,\
        101819, 102100, 102116, 102136, 102159, 102178, 102194, 102215,\
        102239, 102255, 102274, 102309, 102326, 102364, 102382, 102398,\
        102417, 102443, 102464, 102484, 102516, 102529, 102551, 102562,\
        102574, 102587, 102606, 102625, 102648, 102667, 102693, 102713,\
        102733, 102758, 102775, 102796, 102811, 102830, 102847, 102870,\
        102883, 102904, 102924, 102944, 102965, 102982, 102997, 103017,\
        103035, 103054, 103075, 103097, 103116, 103135, 103156, 103176,\
        103195, 103214, 103235, 103249, 103264, 103283, 103301, 103323,\
        103340, 103390, 103407, 103419, 103438, 103456, 103468, 103479,\
        103497, 103512, 103528, 103540, 103555, 103578, 103593, 103617,\
        103627, 103646, 103665, 103679, 103697, 103715, 103731, 103747,\
        103761, 103774, 103800, 103818, 103842, 103880, 103895, 103921,\
        103965, 103977, 103995, 104008, 104025, 104055, 104073, 104118,\
        104142, 104152, 104174, 104191, 104204, 104220, 104255, 104279,\
        104340, 104398, 104430, 104487, 104515, 104545, 104572, 104606,\
        104632, 104656, 104721, 104745, 104779, 104812, 104836, 105082,\
        105119, 105147, 105191, 105226, 105304, 105329, 105352, 105407,\
        105429, 105454, 105477, 105510, 105530, 105560, 105586, 105620,\
        105641, 105667, 105695, 105723, 105749, 105779, 105804, 105832,\
        105881, 105897, 105967, 105999, 106017, 106043, 106063, 106093,\
        106152, 106227, 106397, 106421, 106461, 106476, 106516, 106538,\
        106559, 106581, 106622, 106680, 106730, 106754, 106765, 106786,\
        106813, 106845, 106869, 106891, 106916, 106942, 106966, 107022,\
        107052, 107070, 107088, 107114, 107126, 107153, 107203, 107221,\
        107249, 107275, 107302, 107325, 107341, 107356, 107382, 107407,\
        107433, 107461, 107489, 107499, 107522, 107546, 107571, 107596,\
        107620, 107646, 107672, 107692, 107718, 107744, 107764, 107790,\
        107814, 107835, 107856, 107881, 107911, 107940, 108115, 108131,\
        108162, 108184, 108209, 108233, 108275, 108294, 108319, 108373,\
        108827, 108878, 108926, 108969, 108984, 109012, 109054, 109087,\
        109106, 109121, 109139, 109161, 109185, 109212, 109261, 110029,\
        110074, 110100, 110126, 110142, 110163, 110181, 110203, 110221,\
        110235, 110258, 110274, 110429, 110442, 110471, 110534, 110580,\
        110599, 110624, 110643, 110661, 110684, 110713, 110741, 110777,\
        110795, 110858, 110884, 110900, 110917, 110970, 110993, 111005,\
        111035, 111056, 111083, 111098, 111126, 111145, 111183, 111197,\
        111238, 111274, 111293, 111311, 111331, 111368, 111389, 111415,\
        111440, 111456, 111481, 111504, 111522, 111542, 111584, 111600,\
        111640, 111702, 111714, 111729, 111750, 111796, 111823, 111841,\
        111855, 111873, 111885, 111902, 111919, 111941, 111956, 111980,\
        111991, 112010, 112025, 112035, 112051, 112068, 112080, 112092,\
        112115, 112140, 112160, 112177, 112196, 112213, 112258, 112294,\
        112315, 112610, 112626, 112656, 112675, 112701, 112713, 112730,\
        112749, 112765, 112812, 112844, 112864, 112887, 112907, 112934,\
        112952, 112972, 113038, 113062, 113156, 113178, 113194, 113235,\
        113259, 113275, 113295, 113312, 113333, 113357, 113375, 113392,\
        113414, 113476, 113496, 113519, 113889, 113930, 113957, 114004,\
        114048, 114069, 114084, 114127, 114147, 114173, 114196, 114226,\
        114266, 114295, 114313, 114331, 114356, 114374, 114399, 114428,\
        114457, 114500, 114525, 114569, 114589, 114633, 114655, 114677,\
        114703, 114719, 114738, 114755, 114777, 114789, 114801, 114852,\
        114879, 114900, 114942, 114960, 114996, 115019, 115055, 115095,\
        115115, 115130, 115197, 115217, 115236, 115275, 115283, 115303,\
        115321, 115337, 115377, 115413, 115442, 115465, 115491, 115535,\
        115554, 115570, 115584, 115612, 115630, 115644, 115662, 115675,\
        115689, 115708, 115721, 115735, 115759, 115787, 115806, 115823,\
        115844, 115870, 115888, 115912, 115935, 115963, 115976, 115996,\
        116019, 116044, 116065, 116082, 116101, 116115, 116155, 116173,\
        116184])

    known_pulser_ids['run1507'] = {}
    known_pulser_ids['run1507']['hpol'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1507_pulser_eventids_site_1a_bicone_hpol_16dB.csv',delimiter=',').astype(int)
    known_pulser_ids['run1507']['vpol'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1507_pulser_eventids_site_1a_bicone_vpol_16dB.csv',delimiter=',').astype(int)

    known_pulser_ids['run1509'] = {}
    known_pulser_ids['run1509']['hpol'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1509_pulser_eventids_site_2_bicone_hpol_22dB.csv',delimiter=',').astype(int)
    known_pulser_ids['run1509']['vpol'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1509_pulser_eventids_site_2_bicone_vpol_17dB.csv',delimiter=',').astype(int)

    known_pulser_ids['run1511'] = {}
    known_pulser_ids['run1511']['hpol'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1511_pulser_eventids_site_3_bicone_hpol_20dB.csv',delimiter=',').astype(int)
    known_pulser_ids['run1511']['vpol'] = numpy.loadtxt(os.environ['BEACON_ANALYSIS_DIR'] + 'tools/eventids/run1511_pulser_eventids_site_3_bicone_vpol_20dB.csv',delimiter=',').astype(int)

    if remove_ignored == True:
        ignore_events = loadIgnorableEventids()
        for key in numpy.array(list(known_pulser_ids.keys()))[numpy.isin(numpy.array(list(known_pulser_ids.keys())),numpy.array(list(ignore_events.keys())))]:
            if type(known_pulser_ids[key]) is dict:
                for kkey,val in known_pulser_ids[key].items():
                    known_pulser_ids[key][kkey] = known_pulser_ids[key][kkey][~numpy.isin(known_pulser_ids[key][kkey],ignore_events[key])]
            else:
                known_pulser_ids[key] = known_pulser_ids[key][~numpy.isin(known_pulser_ids[key],ignore_events[key])]
    #import pdb; pdb.set_trace()
    return known_pulser_ids



'''
MAKE AN EXPECTED PULSER TIME DELAY FUNCTION
'''

if __name__ == '__main__':
    try:
        print('Loaded run info dictionaries.')
        plt.ion()

    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)

