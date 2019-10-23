'''
This script use time delays measured by frequency_domain_time_delays to estimate the phase
center locations of the antennas.  
'''

import numpy
import itertools
import os
import sys
import csv
from iminuit import Minuit

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import objects.station as bc
import tools.info as info

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()

pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))

antennas_physical = info.loadAntennaLocationsENU(deploy_index=1)[0]
antenna_locs = numpy.array([antennas_physical[0],antennas_physical[1],antennas_physical[2],antennas_physical[3]])
antenna_locs_3d = antenna_locs[pairs]

A0Location = info.loadAntennaZeroLocation(deploy_index=1)#latitude,longtidue,elevation
pulser_locations = info.loadPulserLocationsENU() #ENU

site1_pulser_location = pulser_locations['run1507']
site2_pulser_location = pulser_locations['run1509']
site3_pulser_location = pulser_locations['run1511']

#12 variables (x2 because of polarizations for output variables)
ant0_physical_x = antennas_physical[0][0]
ant0_physical_y = antennas_physical[0][1]
ant0_physical_z = antennas_physical[0][2]

ant1_physical_x = antennas_physical[1][0]
ant1_physical_y = antennas_physical[1][1]
ant1_physical_z = antennas_physical[1][2]

ant2_physical_x = antennas_physical[2][0]
ant2_physical_y = antennas_physical[2][1]
ant2_physical_z = antennas_physical[2][2]

ant3_physical_x = antennas_physical[3][0]
ant3_physical_y = antennas_physical[3][1]
ant3_physical_z = antennas_physical[3][2]

#Limits 
guess_range = None #Limit to how far from physical center the phase centers can be.  

if guess_range is not None:
    ant0_physical_limits_x = (antennas_physical[0][0] - guess_range ,antennas_physical[0][0] + guess_range)
    ant0_physical_limits_y = (antennas_physical[0][1] - guess_range ,antennas_physical[0][1] + guess_range)
    ant0_physical_limits_z = (antennas_physical[0][2] - guess_range ,antennas_physical[0][2] + guess_range)

    ant1_physical_limits_x = (antennas_physical[1][0] - guess_range ,antennas_physical[1][0] + guess_range)
    ant1_physical_limits_y = (antennas_physical[1][1] - guess_range ,antennas_physical[1][1] + guess_range)
    ant1_physical_limits_z = (antennas_physical[1][2] - guess_range ,antennas_physical[1][2] + guess_range)

    ant2_physical_limits_x = (antennas_physical[2][0] - guess_range ,antennas_physical[2][0] + guess_range)
    ant2_physical_limits_y = (antennas_physical[2][1] - guess_range ,antennas_physical[2][1] + guess_range)
    ant2_physical_limits_z = (antennas_physical[2][2] - guess_range ,antennas_physical[2][2] + guess_range)

    ant3_physical_limits_x = (antennas_physical[3][0] - guess_range ,antennas_physical[3][0] + guess_range)
    ant3_physical_limits_y = (antennas_physical[3][1] - guess_range ,antennas_physical[3][1] + guess_range)
    ant3_physical_limits_z = (antennas_physical[3][2] - guess_range ,antennas_physical[3][2] + guess_range)
    
    puls1_z_limits          = (site1_pulser_location[2] - guess_range ,site1_pulser_location[2] + guess_range)
    puls2_z_limits          = (site2_pulser_location[2] - guess_range ,site2_pulser_location[2] + guess_range)
    puls3_z_limits          = (site3_pulser_location[2] - guess_range ,site3_pulser_location[2] + guess_range)

else:
    #ant0_physical_limits_x = None 
    #ant0_physical_limits_y = None
    #ant0_physical_limits_z = None

    ant1_physical_limits_x = None
    ant1_physical_limits_y = None
    ant1_physical_limits_z = (0.5,None)

    ant2_physical_limits_x = None
    ant2_physical_limits_y = None
    ant2_physical_limits_z = (0.5,None)

    ant3_physical_limits_x = None
    ant3_physical_limits_y = None
    ant3_physical_limits_z = (0.5,None)

    puls1_z_limits          = (None, 0.0)#(site1_pulser_location[2] - 300, site1_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)
    puls2_z_limits          = (None, 0.0)#(site2_pulser_location[2] - 300, site2_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)
    puls3_z_limits          = (None, 0.0)#(site3_pulser_location[2] - 300, site3_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)


n = 1.0003 #Index of refraction of air
c = 299792458/n #m/s


#Measureables (6 baselines x 3 sites x 2 polarizations = 36)
'''
site1_measured_time_delays_hpol = [((0, 1), -31.50596658225331), ((0, 2), 112.29204025938702), ((0, 3), 32.248928504073355), ((1, 2), 141.11091746227595), ((1, 3), 69.13153350845035), ((2, 3), -74.99795442193404)]
site1_measured_time_delays_errors_hpol = [((0, 1), 0.5820445560999024), ((0, 2), 0.40247139497220014), ((0, 3), 0.42569738988685707), ((1, 2), 0.39082979328625916), ((1, 3), 0.7415134220554188), ((2, 3), 0.5877255148788703)]

site1_measured_time_delays_vpol = [((0, 1), -37.335566098781776), ((0, 2), 103.14898432141268), ((0, 3), 39.19824604415767), ((1, 2), 139.6802987989492), ((1, 3), 77.16843791624561), ((2, 3), -63.362824592555654)]
site1_measured_time_delays_errors_vpol = [((0, 1), 0.45538449491888333), ((0, 2), 0.40495765028137637), ((0, 3), 0.41540945533255474), ((1, 2), 0.456427491280387), ((1, 3), 0.39227205895617356), ((2, 3), 0.4360430978356431)]

#used argmin
site2_measured_time_delays_hpol = [((0, 1), -72.57254966760061), ((0, 2), 43.731996686600134), ((0, 3), -46.33463302321706), ((1, 2), 114.85538839910733), ((1, 3), 34.70043293082059), ((2, 3), -82.85458351753215)]
site2_measured_time_delays_errors_hpol = [((0, 1), 1.1717465180235311), ((0, 2), 0.6786179165840319), ((0, 3), 0.36293305414459476), ((1, 2), 0.3558863533818111), ((1, 3), 0.27042190659167975), ((2, 3), 0.7935424234041731)]

site2_measured_time_delays_vpol = [((0, 1), -81.30840774663484), ((0, 2), 33.29805960105865), ((0, 3), -37.12525410979598), ((1, 2), 113.72069195864087), ((1, 3), 45.42737454563861), ((2, 3), -70.33621702538534)]
site2_measured_time_delays_errors_vpol = [((0, 1), 0.6132982582319269), ((0, 2), 0.539600854639544), ((0, 3), 0.47499134267800347), ((1, 2), 0.5903966622377067), ((1, 3), 0.6131526958621443), ((2, 3), 0.5524757089532757)]

site3_measured_time_delays_hpol = [((0, 1), -83.94528392503807), ((0, 2), -133.81057343891848), ((0, 3), -170.3800596579772), ((1, 2), -51.541646794480094), ((1, 3), -81.94080033449774), ((2, 3), -34.34775837342382)]
site3_measured_time_delays_errors_hpol = [((0, 1), 1.1394211310301217), ((0, 2), 0.7388983029470965), ((0, 3), 0.4739267498989166), ((1, 2), 1.1151042745185153), ((1, 3), 0.26310842496261105), ((2, 3), 0.85338337359965)]

site3_measured_time_delays_vpol = [((0, 1), -92.90894952599034), ((0, 2), -146.753746628771), ((0, 3), -168.69628317644583), ((1, 2), -54.27219772136395), ((1, 3), -75.38221290158118), ((2, 3), -21.586881578133063)]
site3_measured_time_delays_errors_vpol = [((0, 1), 0.4098254478375821), ((0, 2), 0.38743367386411853), ((0, 3), 0.4443820434508988), ((1, 2), 0.40067940060880697), ((1, 3), 0.4256517070789963), ((2, 3), 0.4175103282819117)]
'''
'''
site1_measured_time_delays_hpol = [((0, 1), -46.74361492556689), ((0, 2), 99.3655771932916), ((0, 3), 28.12886674113619), ((1, 2), 145.62691144105065), ((1, 3), 75.00828664661606), ((2, 3), -70.49396538776276)]
site1_measured_time_delays_errors_hpol = [((0, 1), 1.004617619088465), ((0, 2), 0.5600890744591248), ((0, 3), 0.5654699954719372), ((1, 2), 0.6993129436497062), ((1, 3), 0.8840990755387067), ((2, 3), 0.5449783963164334)]

site1_measured_time_delays_vpol = [((0, 1), -39.6501515898923), ((0, 2), 104.00705194659903), ((0, 3), 34.99880369633982), ((1, 2), 143.95645138714508), ((1, 3), 74.58377194219477), ((2, 3), -68.8109260444564)]
site1_measured_time_delays_errors_vpol = [((0, 1), 0.5139310788454517), ((0, 2), 0.5444560732664553), ((0, 3), 0.6618302220560341), ((1, 2), 0.7273671115826105), ((1, 3), 1.4044000212141627), ((2, 3), 0.5731623616756774)]


site2_measured_time_delays_hpol = [((0, 1), -85.6362865721698), ((0, 2), 33.35216757390571), ((0, 3), -47.247803495092114), ((1, 2), 121.68926776596173), ((1, 3), 38.089013155197364), ((2, 3), -79.77489660476738)]
site2_measured_time_delays_errors_hpol = [((0, 1), 1.3991855251282468), ((0, 2), 2.0506323287800075), ((0, 3), 0.578684129791201), ((1, 2), 1.2607841691075659), ((1, 3), 0.9630827169040093), ((2, 3), 1.7899950766134167)]

site2_measured_time_delays_vpol = [((0, 1), -81.87169517607926), ((0, 2), 36.319617300484914), ((0, 3), -40.57365690370628), ((1, 2), 119.22949518976401), ((1, 3), 44.772064570869865), ((2, 3), -76.35524912975656)]
site2_measured_time_delays_errors_vpol = [((0, 1), 0.4131751215838118), ((0, 2), 0.5200782620321024), ((0, 3), 1.004092788414462), ((1, 2), 0.9981381866793425), ((1, 3), 1.1949612507487604), ((2, 3), 0.7809621181636988)]

site3_measured_time_delays_hpol = [((0, 1), -98.7292921127855), ((0, 2), -148.05651578297704), ((0, 3), -174.4545392939933), ((1, 2), -49.175762208847864), ((1, 3), -76.57822348706328), ((2, 3), -26.44206302397775)]
site3_measured_time_delays_errors_hpol = [((0, 1), 1.3242371602917185), ((0, 2), 1.4917511712018567), ((0, 3), 1.2562783900243546), ((1, 2), 1.0256158780825804), ((1, 3), 1.005302612737222), ((2, 3), 1.7162891733977599)]

site3_measured_time_delays_vpol = [((0, 1), -94.42573114632032), ((0, 2), -146.11242851886647), ((0, 3), -172.67984487643378), ((1, 2), -51.38556198060898), ((1, 3), -77.3233634764919), ((2, 3), -26.75595617693424)]
site3_measured_time_delays_errors_vpol = [((0, 1), 0.3806854370258467), ((0, 2), 0.4789613211404254), ((0, 3), 0.817391338181048), ((1, 2), 0.6205804905412381), ((1, 3), 0.7876550285730572), ((2, 3), 0.6274695249295562)]
'''

#hilbert before corrstime_differences_hpol

site1_measured_time_delays_hpol = [((0, 1), -48.18506157056843), ((0, 2), 97.07768637505505), ((0, 3), 29.43802765143754), ((1, 2), 145.2030661555888), ((1, 3), 77.90470767771336), ((2, 3), -67.44082764554243)]
site1_measured_time_delays_errors_hpol = [((0, 1), 0.6853659720802519), ((0, 2), 0.5676052954832373), ((0, 3), 0.47633793467451263), ((1, 2), 0.551831987060609), ((1, 3), 0.7145727137782395), ((2, 3), 0.5659149940932722)]

site1_measured_time_delays_vpol = [((0, 1), -41.89855316325409), ((0, 2), 104.20363218904961), ((0, 3), 34.042796359617434), ((1, 2), 146.01822822446695), ((1, 3), 75.69195437829129), ((2, 3), -70.11195348021838)]
site1_measured_time_delays_errors_vpol = [((0, 1), 0.6221622543691303), ((0, 2), 0.5128126551777908), ((0, 3), 0.5465249344579302), ((1, 2), 0.6524150476402931), ((1, 3), 0.5610624009421362), ((2, 3), 0.6072021832980956)]

site2_measured_time_delays_hpol = [((0, 1), -90.30781423773114), ((0, 2), 31.322379889153908), ((0, 3), -46.09013764757296), ((1, 2), 121.8024966929029), ((1, 3), 45.020646502368315), ((2, 3), -76.9780720810267)]
site2_measured_time_delays_errors_hpol = [((0, 1), 1.1225603543618814), ((0, 2), 0.8022755343378248), ((0, 3), 0.6772874098082104), ((1, 2), 0.9903960712986218), ((1, 3), 1.2195303653660767), ((2, 3), 0.7104952922475969)]

site2_measured_time_delays_vpol = [((0, 1), -85.66026484243226), ((0, 2), 37.05620785870472), ((0, 3), -41.23094882656212), ((1, 2), 123.29675054889603), ((1, 3), 44.07893813824877), ((2, 3), -78.04200779637598)]
site2_measured_time_delays_errors_vpol = [((0, 1), 0.911190870636156), ((0, 2), 0.6483444314312705), ((0, 3), 0.7864503381843414), ((1, 2), 0.9574044979296679), ((1, 3), 0.7886242316671788), ((2, 3), 0.8327133174629702)]

site3_measured_time_delays_hpol = [((0, 1), -98.30927070427768), ((0, 2), -147.83771560106516), ((0, 3), -171.43203611958108), ((1, 2), -49.66997012895028), ((1, 3), -73.82557815939602), ((2, 3), -24.425656089572428)]
site3_measured_time_delays_errors_hpol = [((0, 1), 0.755976869714551), ((0, 2), 1.0587427537912906), ((0, 3), 0.6722806468604524), ((1, 2), 0.8868692409047823), ((1, 3), 0.7981106501884698), ((2, 3), 1.2353048369623272)]

site3_measured_time_delays_vpol = [((0, 1), -96.16179161978721), ((0, 2), -145.5425126728109), ((0, 3), -174.10368940726966), ((1, 2), -49.336495960395276), ((1, 3), -78.21679649612219), ((2, 3), -28.468930209762764)]
site3_measured_time_delays_errors_vpol = [((0, 1), 0.5519803564345046), ((0, 2), 0.4939047658701588), ((0, 3), 0.6956400754396588), ((1, 2), 0.548392710774062), ((1, 3), 0.5596150011008781), ((2, 3), 0.6527907218323572)]



def f(ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z, puls1_z, puls2_z, puls3_z):
    '''
    To generalize, look into from_array_func Minuit initializer.  
    '''
    #fixing the locations of antenna zero.
    ant0_x = 0.0
    ant0_y = 0.0
    ant0_z = 0.0

    #Calculate distances from pulser to each antenna
    site1_d0 = numpy.sqrt((site1_pulser_location[0] - ant0_x)**2 + (site1_pulser_location[1] - ant0_y)**2 + (puls1_z - ant0_z)**2 )
    site1_d1 = numpy.sqrt((site1_pulser_location[0] - ant1_x)**2 + (site1_pulser_location[1] - ant1_y)**2 + (puls1_z - ant1_z)**2 )
    site1_d2 = numpy.sqrt((site1_pulser_location[0] - ant2_x)**2 + (site1_pulser_location[1] - ant2_y)**2 + (puls1_z - ant2_z)**2 )
    site1_d3 = numpy.sqrt((site1_pulser_location[0] - ant3_x)**2 + (site1_pulser_location[1] - ant3_y)**2 + (puls1_z - ant3_z)**2 )

    site2_d0 = numpy.sqrt((site2_pulser_location[0] - ant0_x)**2 + (site2_pulser_location[1] - ant0_y)**2 + (puls2_z - ant0_z)**2 )
    site2_d1 = numpy.sqrt((site2_pulser_location[0] - ant1_x)**2 + (site2_pulser_location[1] - ant1_y)**2 + (puls2_z - ant1_z)**2 )
    site2_d2 = numpy.sqrt((site2_pulser_location[0] - ant2_x)**2 + (site2_pulser_location[1] - ant2_y)**2 + (puls2_z - ant2_z)**2 )
    site2_d3 = numpy.sqrt((site2_pulser_location[0] - ant3_x)**2 + (site2_pulser_location[1] - ant3_y)**2 + (puls2_z - ant3_z)**2 )

    site3_d0 = numpy.sqrt((site3_pulser_location[0] - ant0_x)**2 + (site3_pulser_location[1] - ant0_y)**2 + (puls3_z - ant0_z)**2 )
    site3_d1 = numpy.sqrt((site3_pulser_location[0] - ant1_x)**2 + (site3_pulser_location[1] - ant1_y)**2 + (puls3_z - ant1_z)**2 )
    site3_d2 = numpy.sqrt((site3_pulser_location[0] - ant2_x)**2 + (site3_pulser_location[1] - ant2_y)**2 + (puls3_z - ant2_z)**2 )
    site3_d3 = numpy.sqrt((site3_pulser_location[0] - ant3_x)**2 + (site3_pulser_location[1] - ant3_y)**2 + (puls3_z - ant3_z)**2 )

    #calculate differences in distances travelled and convert to expected time delays for known pairs.
    #Ant 0 - Ant 1
    site1_p0 = ((site1_d0 - site1_d1)/c)*1.0e9 #ns
    site2_p0 = ((site2_d0 - site2_d1)/c)*1.0e9 #ns
    site3_p0 = ((site3_d0 - site3_d1)/c)*1.0e9 #ns

    #Ant 0 - Ant 2
    site1_p1 = ((site1_d0 - site1_d2)/c)*1.0e9 #ns
    site2_p1 = ((site2_d0 - site2_d2)/c)*1.0e9 #ns
    site3_p1 = ((site3_d0 - site3_d2)/c)*1.0e9 #ns

    #Ant 0 - Ant 3
    site1_p2 = ((site1_d0 - site1_d3)/c)*1.0e9 #ns
    site2_p2 = ((site2_d0 - site2_d3)/c)*1.0e9 #ns
    site3_p2 = ((site3_d0 - site3_d3)/c)*1.0e9 #ns

    #Ant 1 - Ant 2
    site1_p3 = ((site1_d1 - site1_d2)/c)*1.0e9 #ns
    site2_p3 = ((site2_d1 - site2_d2)/c)*1.0e9 #ns
    site3_p3 = ((site3_d1 - site3_d2)/c)*1.0e9 #ns

    #Ant 1 - Ant 3
    site1_p4 = ((site1_d1 - site1_d3)/c)*1.0e9 #ns
    site2_p4 = ((site2_d1 - site2_d3)/c)*1.0e9 #ns
    site3_p4 = ((site3_d1 - site3_d3)/c)*1.0e9 #ns

    #Ant 2 - Ant 3
    site1_p5 = ((site1_d2 - site1_d3)/c)*1.0e9 #ns
    site2_p5 = ((site2_d2 - site2_d3)/c)*1.0e9 #ns
    site3_p5 = ((site3_d2 - site3_d3)/c)*1.0e9 #ns

    
    chi_2 =     ((site1_p0 - site1_measured_time_delays[0][1])**2)/site1_measured_time_delays_errors[0][1]**2 + \
                ((site2_p0 - site2_measured_time_delays[0][1])**2)/site2_measured_time_delays_errors[0][1]**2 + \
                ((site3_p0 - site3_measured_time_delays[0][1])**2)/site3_measured_time_delays_errors[0][1]**2 + \
                \
                ((site1_p1 - site1_measured_time_delays[1][1])**2)/site1_measured_time_delays_errors[1][1]**2 + \
                ((site2_p1 - site2_measured_time_delays[1][1])**2)/site2_measured_time_delays_errors[1][1]**2 + \
                ((site3_p1 - site3_measured_time_delays[1][1])**2)/site3_measured_time_delays_errors[1][1]**2 + \
                \
                ((site1_p2 - site1_measured_time_delays[2][1])**2)/site1_measured_time_delays_errors[2][1]**2 + \
                ((site2_p2 - site2_measured_time_delays[2][1])**2)/site2_measured_time_delays_errors[2][1]**2 + \
                ((site3_p2 - site3_measured_time_delays[2][1])**2)/site3_measured_time_delays_errors[2][1]**2 + \
                \
                ((site1_p3 - site1_measured_time_delays[3][1])**2)/site1_measured_time_delays_errors[3][1]**2 + \
                ((site2_p3 - site2_measured_time_delays[3][1])**2)/site2_measured_time_delays_errors[3][1]**2 + \
                ((site3_p3 - site3_measured_time_delays[3][1])**2)/site3_measured_time_delays_errors[3][1]**2 + \
                \
                ((site1_p4 - site1_measured_time_delays[4][1])**2)/site1_measured_time_delays_errors[4][1]**2 + \
                ((site2_p4 - site2_measured_time_delays[4][1])**2)/site2_measured_time_delays_errors[4][1]**2 + \
                ((site3_p4 - site3_measured_time_delays[4][1])**2)/site3_measured_time_delays_errors[4][1]**2 + \
                \
                ((site1_p5 - site1_measured_time_delays[5][1])**2)/site1_measured_time_delays_errors[5][1]**2 + \
                ((site2_p5 - site2_measured_time_delays[5][1])**2)/site2_measured_time_delays_errors[5][1]**2 + \
                ((site3_p5 - site3_measured_time_delays[5][1])**2)/site3_measured_time_delays_errors[5][1]**2
    
    return chi_2


if __name__ == '__main__':
    try:

        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to vpol')
                mode = 'vpol'
        else:
            print('No mode given.  Defaulting to vpol')
            mode = 'vpol'

        print('Performing calculations for %s'%mode)
        if mode == 'hpol':
            site1_measured_time_delays = site1_measured_time_delays_hpol
            site2_measured_time_delays = site2_measured_time_delays_hpol
            site3_measured_time_delays = site3_measured_time_delays_hpol

            site1_measured_time_delays_errors = site1_measured_time_delays_errors_hpol
            site2_measured_time_delays_errors = site2_measured_time_delays_errors_hpol
            site3_measured_time_delays_errors = site3_measured_time_delays_errors_hpol

        elif mode == 'vpol':
            site1_measured_time_delays = site1_measured_time_delays_vpol
            site2_measured_time_delays = site2_measured_time_delays_vpol
            site3_measured_time_delays = site3_measured_time_delays_vpol

            site1_measured_time_delays_errors = site1_measured_time_delays_errors_vpol
            site2_measured_time_delays_errors = site2_measured_time_delays_errors_vpol
            site3_measured_time_delays_errors = site3_measured_time_delays_errors_vpol

        initial_step = 1.0 #m

        m = Minuit(     f,\
                        ant1_x=ant1_physical_x,\
                        ant1_y=ant1_physical_y,\
                        ant1_z=ant1_physical_z,\
                        ant2_x=ant2_physical_x,\
                        ant2_y=ant2_physical_y,\
                        ant2_z=ant2_physical_z,\
                        ant3_x=ant3_physical_x,\
                        ant3_y=ant3_physical_y,\
                        ant3_z=ant3_physical_z,\
                        puls1_z=site1_pulser_location[2],\
                        puls2_z=site2_pulser_location[2],\
                        puls3_z=site3_pulser_location[2],\
                        error_ant1_x=initial_step,\
                        error_ant1_y=initial_step,\
                        error_ant1_z=initial_step,\
                        error_ant2_x=initial_step,\
                        error_ant2_y=initial_step,\
                        error_ant2_z=initial_step,\
                        error_ant3_x=initial_step,\
                        error_ant3_y=initial_step,\
                        error_ant3_z=initial_step,\
                        error_puls1_z=initial_step*5,\
                        error_puls2_z=initial_step*5,\
                        error_puls3_z=initial_step*5,\
                        errordef = 1.0,\
                        limit_ant1_x=ant1_physical_limits_x,\
                        limit_ant1_y=ant1_physical_limits_y,\
                        limit_ant1_z=ant1_physical_limits_z,\
                        limit_ant2_x=ant2_physical_limits_x,\
                        limit_ant2_y=ant2_physical_limits_y,\
                        limit_ant2_z=ant2_physical_limits_z,\
                        limit_ant3_x=ant3_physical_limits_x,\
                        limit_ant3_y=ant3_physical_limits_y,\
                        limit_ant3_z=ant3_physical_limits_z,\
                        limit_puls1_z=puls1_z_limits,\
                        limit_puls2_z=puls2_z_limits,\
                        limit_puls3_z=puls3_z_limits)
        result = m.migrad()
        m.hesse()
        m.minos()
        pprint(m.get_fmin())
        print(result)

        #12 variables
        ant0_phase_x = 0.0#m.values['ant0_x']
        ant0_phase_y = 0.0#m.values['ant0_y']
        ant0_phase_z = 0.0#m.values['ant0_z']

        ant1_phase_x = m.values['ant1_x']
        ant1_phase_y = m.values['ant1_y']
        ant1_phase_z = m.values['ant1_z']

        ant2_phase_x = m.values['ant2_x']
        ant2_phase_y = m.values['ant2_y']
        ant2_phase_z = m.values['ant2_z']

        ant3_phase_x = m.values['ant3_x']
        ant3_phase_y = m.values['ant3_y']
        ant3_phase_z = m.values['ant3_z']

        puls1_phase_z = m.values['puls1_z']
        puls2_phase_z = m.values['puls2_z']
        puls3_phase_z = m.values['puls3_z']
        pulser_phase_locs = numpy.array([puls1_phase_z, puls2_phase_z, puls3_phase_z])
        pulser_physical_locs = numpy.array([site1_pulser_location[2], site2_pulser_location[2], site3_pulser_location[2]])
        phase_locs = numpy.array([[ant0_phase_x,ant0_phase_y,ant0_phase_z],[ant1_phase_x,ant1_phase_y,ant1_phase_z],[ant2_phase_x,ant2_phase_y,ant2_phase_z],[ant3_phase_x,ant3_phase_y,ant3_phase_z]])

        print('Antenna Locations: \n%s'%str(antenna_locs))
        print('Phase Locations: \n%s'%str(phase_locs))

        print('\nDifference (antenna_locs - phase_locs): \n%s'%str(antenna_locs - phase_locs))

        print('New z values for pulsers:')
        print(pulser_phase_locs)
        print('Physical z values for pulsers:')
        print(pulser_physical_locs)
        info.plotStationAndPulsers() 

        measured_baselines = {'01':129*0.3048,
                              '02':163*0.3048,
                              '03':181*0.3048,
                              '12':151*0.3048,
                              '13':102*0.3048,
                              '23':85 *0.3048}
        baselines = {}
        print('Measured Baseline  -  Phase Baseline  =  ')
        for pair in pairs:
            #print('Measured Baseline = ', measured_baselines[str(min(pair))+str(max(pair))])
            baselines[str(min(pair))+str(max(pair))] = numpy.sqrt((phase_locs[min(pair)][0] - phase_locs[max(pair)][0])**2 + (phase_locs[min(pair)][1] - phase_locs[max(pair)][1])**2 + (phase_locs[min(pair)][2] - phase_locs[max(pair)][2])**2)
            #print('Phase Baseline = ', baselines[str(min(pair))+str(max(pair))])
            print('%0.3f  -  %0.3f  =  %0.3f'%(measured_baselines[str(min(pair))+str(max(pair))], baselines[str(min(pair))+str(max(pair))], measured_baselines[str(min(pair))+str(max(pair))]-baselines[str(min(pair))+str(max(pair))]))



    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






