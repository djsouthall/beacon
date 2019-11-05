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
        cable_delays = info.loadCableDelays()[mode]

        pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))

        antennas_physical = info.loadAntennaLocationsENU(deploy_index=1)[0]
        antenna_locs = numpy.array([antennas_physical[0],antennas_physical[1],antennas_physical[2],antennas_physical[3]])
        antenna_locs_3d = antenna_locs[pairs]

        A0Location = info.loadAntennaZeroLocation(deploy_index=1)#latitude,longtidue,elevation
        all_pulser_locations = info.loadPulserPhaseLocationsENU() #ENU

        site1_pulser_location = all_pulser_locations[mode]['run1507']
        site2_pulser_location = all_pulser_locations[mode]['run1509']
        site3_pulser_location = all_pulser_locations[mode]['run1511']

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
            
            puls1_x_limits          = (site1_pulser_location[0] - guess_range ,site1_pulser_location[0] + guess_range)
            puls1_y_limits          = (site1_pulser_location[1] - guess_range ,site1_pulser_location[1] + guess_range)
            puls1_z_limits          = (site1_pulser_location[2] - guess_range ,site1_pulser_location[2] + guess_range)

            puls2_x_limits          = (site2_pulser_location[0] - guess_range ,site2_pulser_location[0] + guess_range)
            puls2_y_limits          = (site2_pulser_location[1] - guess_range ,site2_pulser_location[1] + guess_range)
            puls2_z_limits          = (site2_pulser_location[2] - guess_range ,site2_pulser_location[2] + guess_range)

            puls3_x_limits          = (site3_pulser_location[0] - guess_range ,site3_pulser_location[0] + guess_range)
            puls3_y_limits          = (site3_pulser_location[1] - guess_range ,site3_pulser_location[1] + guess_range)
            puls3_z_limits          = (site3_pulser_location[2] - guess_range ,site3_pulser_location[2] + guess_range)


        else:
            #ant0_physical_limits_x = None 
            #ant0_physical_limits_y = None
            #ant0_physical_limits_z = None

            ant1_physical_limits_x = None#(None,0.0)
            ant1_physical_limits_y = None#None
            ant1_physical_limits_z = None#(10.0,None)

            ant2_physical_limits_x = None#None
            ant2_physical_limits_y = None#(None,0.0)
            ant2_physical_limits_z = None#None

            ant3_physical_limits_x = None#(None,0.0)
            ant3_physical_limits_y = None#(None,0.0)
            ant3_physical_limits_z = None#(0.0,None)

            puls1_x_limits          = (0.0, None)
            puls1_y_limits          = (None, 0.0)
            puls1_z_limits          = (None, -10.0)#(site1_pulser_location[2] - 300, site1_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)

            puls2_x_limits          = (0.0, None)
            puls2_y_limits          = (None, 0.0)
            puls2_z_limits          = (None, -10.0)#(site2_pulser_location[2] - 300, site2_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)

            puls3_x_limits          = (0.0, None)
            puls3_y_limits          = (0.0, None)
            puls3_z_limits          = (None, -10.0)#(None, 0.0)#(site3_pulser_location[2] - 300, site3_pulser_location[2] + 20)#(site1_pulser_location[2] - 20 ,site1_pulser_location[2] + 20)



        n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
        c = 299792458/n #m/s

        #calculated with align_method=8, crit_freq_low_pass_MHz = 70,  low_pass_filter_order = 4, crit_freq_high_pass_MHz = 30, high_pass_filter_order = 8

        site1_measured_time_delays_hpol =  [((0, 1), -40.440209498791944), ((0, 2), 105.33057271920997), ((0, 3), 30.018460817451853), ((1, 2), 145.63622318088764), ((1, 3), 70.52602999210539), ((2, 3), -75.24411224605574)]
        site1_measured_time_delays_errors_hpol =  [((0, 1), 0.06193089185745354), ((0, 2), 0.044864022122539454), ((0, 3), 0.04952495740572169), ((1, 2), 0.032093018610684435), ((1, 3), 0.040718907022717674), ((2, 3), 0.034526519878503886)]
        site1_measured_time_delays_vpol =  [((0, 1), -38.063287651759545), ((0, 2), 101.70119938927523), ((0, 3), 36.2876186848624), ((1, 2), 139.78669639622473), ((1, 3), 74.44365865334211), ((2, 3), -65.4064310607008)]
        site1_measured_time_delays_errors_vpol =  [((0, 1), 0.034109221590889145), ((0, 2), 0.037184625480844086), ((0, 3), 0.03654373624493399), ((1, 2), 0.03670835970229502), ((1, 3), 0.036672366959658816), ((2, 3), 0.03612413830246497)]

        site2_measured_time_delays_hpol =  [((0, 1), -82.47404046303845), ((0, 2), 40.38479946003285), ((0, 3), -44.629047343370836), ((1, 2), 122.73256779826421), ((1, 3), 37.87316090544776), ((2, 3), -85.11104338149791)]
        site2_measured_time_delays_errors_hpol =  [((0, 1), 0.0864609952977043), ((0, 2), 0.07274475963064986), ((0, 3), 0.0713954810774026), ((1, 2), 0.06971946054189211), ((1, 3), 0.08147148533844578), ((2, 3), 0.05332486122892472)]
        site2_measured_time_delays_vpol =  [((0, 1), -79.99394882591717), ((0, 2), 36.8744110148191), ((0, 3), -38.44576089035465), ((1, 2), 116.79802405897858), ((1, 3), 41.586253349115516), ((2, 3), -75.31440095977592)]
        site2_measured_time_delays_errors_vpol =  [((0, 1), 0.04814645685384285), ((0, 2), 0.04899790590798649), ((0, 3), 0.04485407838445902), ((1, 2), 0.055336457553921935), ((1, 3), 0.04319429132920376), ((2, 3), 0.04378830878879002)]

        site3_measured_time_delays_hpol =  [((0, 1), -94.78566258324585), ((0, 2), -143.86637982757858), ((0, 3), -177.29778130445408), ((1, 2), -49.16552101183997), ((1, 3), -82.47598224055454), ((2, 3), -33.25646246237567)]
        site3_measured_time_delays_errors_hpol =  [((0, 1), 0.07569115457117638), ((0, 2), 0.09531589022367223), ((0, 3), 0.08541377337608434), ((1, 2), 0.05342651878917397), ((1, 3), 0.05980764593661971), ((2, 3), 0.07215334859645393)]
        site3_measured_time_delays_vpol =  [((0, 1), -92.07069739611525), ((0, 2), -147.16128683129963), ((0, 3), -172.5843775008375), ((1, 2), -55.11372813134522), ((1, 3), -80.44018591398685), ((2, 3), -25.427294958856226)]
        site3_measured_time_delays_errors_vpol =  [((0, 1), 0.03150407914721276), ((0, 2), 0.03635356075580606), ((0, 3), 0.03701344081619418), ((1, 2), 0.033491276265193035), ((1, 3), 0.0330274920797608), ((2, 3), 0.03873217143470215)]

        '''
        #calculated with align_method=8, crit_freq_low_pass_MHz = None,  low_pass_filter_order = None, crit_freq_high_pass_MHz = None, high_pass_filter_order = None

        site1_measured_time_delays_hpol =  [((0, 1), -40.240076807156136), ((0, 2), 105.457675325627), ((0, 3), 30.076736217174066), ((1, 2), 145.54283218855645), ((1, 3), 70.41366528582363), ((2, 3), -75.28322303467672)]
        site1_measured_time_delays_errors_hpol =  [((0, 1), 0.048761746380142044), ((0, 2), 0.04473990112587433), ((0, 3), 0.04513070971648978), ((1, 2), 0.02628383406571089), ((1, 3), 0.03171072931428555), ((2, 3), 0.027736216311380662)]
        
        site1_measured_time_delays_vpol =  [((0, 1), -38.0149400429526), ((0, 2), 101.72079010578945), ((0, 3), 36.40035830577931), ((1, 2), 139.74530559730763), ((1, 3), 74.47618262573134), ((2, 3), -65.34900418050427)]
        site1_measured_time_delays_errors_vpol =  [((0, 1), 0.031574715268129536), ((0, 2), 0.03232533820461333), ((0, 3), 0.033624846630922245), ((1, 2), 0.030584296248440366), ((1, 3), 0.030443644126805343), ((2, 3), 0.029663500399466344)]

        site2_measured_time_delays_hpol =  [((0, 1), -82.26750033217739), ((0, 2), 40.49534830587726), ((0, 3), -44.73107895359357), ((1, 2), 122.46769600413636), ((1, 3), 37.533475883782565), ((2, 3), -85.15957080637239)]
        site2_measured_time_delays_errors_hpol =  [((0, 1), 0.06626537113772676), ((0, 2), 0.0605894101258289), ((0, 3), 0.05535657501306924), ((1, 2), 0.05243438491556112), ((1, 3), 0.06531308537218636), ((2, 3), 0.045171421324951354)]
        site2_measured_time_delays_vpol =  [((0, 1), -79.97358630455436), ((0, 2), 36.76593441860734), ((0, 3), -38.39806412660886), ((1, 2), 116.6607881315279), ((1, 3), 41.640660788711294), ((2, 3), -75.17986472157682)]
        site2_measured_time_delays_errors_vpol =  [((0, 1), 0.04363464145842657), ((0, 2), 0.040989900510301214), ((0, 3), 0.03842368509609836), ((1, 2), 0.04521476575166656), ((1, 3), 0.038623922210662937), ((2, 3), 0.03724034295130426)]

        site3_measured_time_delays_hpol =  [((0, 1), -94.51699527044052), ((0, 2), -143.6667441296591), ((0, 3), -177.17646123686023), ((1, 2), -49.22617845571326), ((1, 3), -82.46858030352125), ((2, 3), -33.338366700114044)]
        site3_measured_time_delays_errors_hpol =  [((0, 1), 0.054582230632404874), ((0, 2), 0.06460489511707884), ((0, 3), 0.06160715068110298), ((1, 2), 0.04543656576070127), ((1, 3), 0.03481691345989192), ((2, 3), 0.057998526520626224)]
        site3_measured_time_delays_vpol =  [((0, 1), -92.0513714108267), ((0, 2), -147.1564311181347), ((0, 3), -172.50196583126936), ((1, 2), -55.11465753318667), ((1, 3), -80.3397876709674), ((2, 3), -25.32381320651826)]
        site3_measured_time_delays_errors_vpol =  [((0, 1), 0.026897117112708667), ((0, 2), 0.02887653028683511), ((0, 3), 0.03132265439075829), ((1, 2), 0.028798552999521852), ((1, 3), 0.028530159953231538), ((2, 3), 0.030121461663435427)]
        '''
        '''
        #calculated with align_method=8, crit_freq_low_pass_MHz = 70,  low_pass_filter_order = 8, crit_freq_high_pass_MHz = 50, high_pass_filter_order = 8

        site1_measured_time_delays_hpol =  [((0, 1), -40.64470271129501), ((0, 2), 105.17202548994948), ((0, 3), 30.015387723943626), ((1, 2), 145.76157978332864), ((1, 3), 70.6835945103127), ((2, 3), -75.11809843392194)]
        site1_measured_time_delays_errors_hpol =  [((0, 1), 0.06652513914933796), ((0, 2), 0.04991956226803804), ((0, 3), 0.05394634016956528), ((1, 2), 0.04089447090443683), ((1, 3), 0.04457263570069339), ((2, 3), 0.03553503604238321)]
        
        site1_measured_time_delays_vpol =  [((0, 1), -38.15419336471346), ((0, 2), 101.63794584377128), ((0, 3), 36.15297977838468), ((1, 2), 139.83605046719745), ((1, 3), 74.36412171334415), ((2, 3), -65.47412245573732)]
        site1_measured_time_delays_errors_vpol =  [((0, 1), 0.04365047581848327), ((0, 2), 0.04697512258711787), ((0, 3), 0.04314581431960822), ((1, 2), 0.04546178559648083), ((1, 3), 0.04142305114089252), ((2, 3), 0.04336387256579426)]

        site2_measured_time_delays_hpol =  [((0, 1), -82.69580770804092), ((0, 2), 40.33229422087878), ((0, 3), -44.584324984573776), ((1, 2), 122.98724797853622), ((1, 3), 38.08569980232328), ((2, 3), -85.05283610195872)]
        site2_measured_time_delays_errors_hpol =  [((0, 1), 0.1007304292236345), ((0, 2), 0.09132911743169496), ((0, 3), 0.07212350110860007), ((1, 2), 0.08897094131959918), ((1, 3), 0.08027531820869632), ((2, 3), 0.07069673388498222)]
        
        site2_measured_time_delays_vpol =  [((0, 1), -80.06091300989463), ((0, 2), 36.99033105411573), ((0, 3), -38.511769661866715), ((1, 2), 116.98448265111364), ((1, 3), 41.548981911008596), ((2, 3), -75.47794774082209)]
        site2_measured_time_delays_errors_vpol =  [((0, 1), 0.06558739475388563), ((0, 2), 0.059824303634571675), ((0, 3), 0.05996289174814816), ((1, 2), 0.06836316445154562), ((1, 3), 0.05882383283632012), ((2, 3), 0.05891026601528802)]

        site3_measured_time_delays_hpol =  [((0, 1), -95.04063772895786), ((0, 2), -144.04683233971625), ((0, 3), -177.34148089651308), ((1, 2), -49.09394878959559), ((1, 3), -82.41868553569927), ((2, 3), -33.22519819742752)]
        site3_measured_time_delays_errors_hpol =  [((0, 1), 0.0868973954088791), ((0, 2), 0.14636586500696974), ((0, 3), 0.10949984709887549), ((1, 2), 0.08430841262643438), ((1, 3), 0.08760405827012555), ((2, 3), 0.12743004079228484)]
        
        site3_measured_time_delays_vpol =  [((0, 1), -92.08426367941851), ((0, 2), -147.14022265944442), ((0, 3), -172.68787637066822), ((1, 2), -55.140284778144434), ((1, 3), -80.58398842247041), ((2, 3), -25.627797510968616)]
        site3_measured_time_delays_errors_vpol =  [((0, 1), 0.055082970967158015), ((0, 2), 0.05229886519696005), ((0, 3), 0.05870610955185572), ((1, 2), 0.05249038244970584), ((1, 3), 0.05507228394866774), ((2, 3), 0.057538550987021965)]
        '''

        '''
        #Made with averaged waveforms, given false errors. 
        site1_measured_time_delays_hpol =  [((0, 1), -40.20651888606884), ((0, 2), 105.42486911650398), ((0, 3), 30.076727191600487), ((1, 2), 145.47506405667053), ((1, 3), 70.40830523439116), ((2, 3), -75.22308276818167)]
        site1_measured_time_delays_errors_hpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]
        
        site1_measured_time_delays_vpol =  [((0, 1), -38.04924843261725), ((0, 2), 101.73562399320996), ((0, 3), 36.36094981687252), ((1, 2), 139.78487242582722), ((1, 3), 74.47272782785069), ((2, 3), -65.37467417633744)]
        site1_measured_time_delays_errors_vpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]


        site2_measured_time_delays_hpol =  [((0, 1), -82.30008730167084), ((0, 2), 40.63430060469886), ((0, 3), -44.66647351085744), ((1, 2), 122.62181636325664), ((1, 3), 37.72738525374733), ((2, 3), -85.20700265262238)]
        site2_measured_time_delays_errors_hpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]
        
        site2_measured_time_delays_vpol =  [((0, 1), -79.95580072832284), ((0, 2), 36.75841347009681), ((0, 3), -38.41504264859608), ((1, 2), 116.62044273548572), ((1, 3), 41.603272388349374), ((2, 3), -75.1734561186929)]
        site2_measured_time_delays_errors_vpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]


        site3_measured_time_delays_hpol =  [((0, 1), -94.49037748308051), ((0, 2), -143.62662406045482), ((0, 3), -177.25932209942096), ((1, 2), -49.29253234893085), ((1, 3), -82.51888738184999), ((2, 3), -33.38264080447568)]
        site3_measured_time_delays_errors_hpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]
        
        site3_measured_time_delays_vpol =  [((0, 1), -92.02106229248727), ((0, 2), -147.1274253433212), ((0, 3), -172.50823464410232), ((1, 2), -55.10636305083391), ((1, 3), -80.39340088868113), ((2, 3), -25.380809300781134)]
        site3_measured_time_delays_errors_vpol =  [((0, 1), 0.2), ((0, 2), 0.2), ((0, 3), 0.2), ((1, 2), 0.2), ((1, 3), 0.2), ((2, 3), 0.2)]

        '''
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


        def f(ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z, puls1_x, puls1_y, puls1_z, puls2_x, puls2_y, puls2_z, puls3_x, puls3_y, puls3_z):
            #To generalize, look into from_array_func Minuit initializer.  
            try:
                #fixing the locations of antenna zero.
                ant0_x = 0.0
                ant0_y = 0.0
                ant0_z = 0.0

                #Calculate distances (already converted to ns) from pulser to each antenna

                site1_d0 = (numpy.sqrt((puls1_x - ant0_x)**2 + (puls1_y - ant0_y)**2 + (puls1_z - ant0_z)**2 )/c)*1.0e9 #ns
                site1_d1 = (numpy.sqrt((puls1_x - ant1_x)**2 + (puls1_y - ant1_y)**2 + (puls1_z - ant1_z)**2 )/c)*1.0e9 #ns
                site1_d2 = (numpy.sqrt((puls1_x - ant2_x)**2 + (puls1_y - ant2_y)**2 + (puls1_z - ant2_z)**2 )/c)*1.0e9 #ns
                site1_d3 = (numpy.sqrt((puls1_x - ant3_x)**2 + (puls1_y - ant3_y)**2 + (puls1_z - ant3_z)**2 )/c)*1.0e9 #ns

                site2_d0 = (numpy.sqrt((puls2_x - ant0_x)**2 + (puls2_y - ant0_y)**2 + (puls2_z - ant0_z)**2 )/c)*1.0e9 #ns
                site2_d1 = (numpy.sqrt((puls2_x - ant1_x)**2 + (puls2_y - ant1_y)**2 + (puls2_z - ant1_z)**2 )/c)*1.0e9 #ns
                site2_d2 = (numpy.sqrt((puls2_x - ant2_x)**2 + (puls2_y - ant2_y)**2 + (puls2_z - ant2_z)**2 )/c)*1.0e9 #ns
                site2_d3 = (numpy.sqrt((puls2_x - ant3_x)**2 + (puls2_y - ant3_y)**2 + (puls2_z - ant3_z)**2 )/c)*1.0e9 #ns

                site3_d0 = (numpy.sqrt((puls3_x - ant0_x)**2 + (puls3_y - ant0_y)**2 + (puls3_z - ant0_z)**2 )/c)*1.0e9 #ns
                site3_d1 = (numpy.sqrt((puls3_x - ant1_x)**2 + (puls3_y - ant1_y)**2 + (puls3_z - ant1_z)**2 )/c)*1.0e9 #ns
                site3_d2 = (numpy.sqrt((puls3_x - ant2_x)**2 + (puls3_y - ant2_y)**2 + (puls3_z - ant2_z)**2 )/c)*1.0e9 #ns
                site3_d3 = (numpy.sqrt((puls3_x - ant3_x)**2 + (puls3_y - ant3_y)**2 + (puls3_z - ant3_z)**2 )/c)*1.0e9 #ns



                #calculate differences in distances (already converted to ns) travelled and convert to expected time delays for known pairs.
                #Here I add cable delays, because I am looking at physical distance + delay to measurement to match the measured time differences.
                #Ant 0 - Ant 1
                site1_p0 = (site1_d0 + cable_delays[0]) - (site1_d1 + cable_delays[1])
                site2_p0 = (site2_d0 + cable_delays[0]) - (site2_d1 + cable_delays[1])
                site3_p0 = (site3_d0 + cable_delays[0]) - (site3_d1 + cable_delays[1])

                #Ant 0 - Ant 2
                site1_p1 = (site1_d0 + cable_delays[0]) - (site1_d2 + cable_delays[2])
                site2_p1 = (site2_d0 + cable_delays[0]) - (site2_d2 + cable_delays[2])
                site3_p1 = (site3_d0 + cable_delays[0]) - (site3_d2 + cable_delays[2])

                #Ant 0 - Ant 3
                site1_p2 = (site1_d0 + cable_delays[0]) - (site1_d3 + cable_delays[3])
                site2_p2 = (site2_d0 + cable_delays[0]) - (site2_d3 + cable_delays[3])
                site3_p2 = (site3_d0 + cable_delays[0]) - (site3_d3 + cable_delays[3])

                #Ant 1 - Ant 2
                site1_p3 = (site1_d1 + cable_delays[1]) - (site1_d2 + cable_delays[2])
                site2_p3 = (site2_d1 + cable_delays[1]) - (site2_d2 + cable_delays[2])
                site3_p3 = (site3_d1 + cable_delays[1]) - (site3_d2 + cable_delays[2])

                #Ant 1 - Ant 3
                site1_p4 = (site1_d1 + cable_delays[1]) - (site1_d3 + cable_delays[3])
                site2_p4 = (site2_d1 + cable_delays[1]) - (site2_d3 + cable_delays[3])
                site3_p4 = (site3_d1 + cable_delays[1]) - (site3_d3 + cable_delays[3])

                #Ant 2 - Ant 3
                site1_p5 = (site1_d2 + cable_delays[2]) - (site1_d3 + cable_delays[3])
                site2_p5 = (site2_d2 + cable_delays[2]) - (site2_d3 + cable_delays[3])
                site3_p5 = (site3_d2 + cable_delays[2]) - (site3_d3 + cable_delays[3])


                if False:
                    print('\n')
                    print('site1_p0 = %0.3f \t site1_measured_time_delays[0][1] = %0.3f \t diff = %0.3f'%(site1_p0, site1_measured_time_delays[0][1], site1_p0 - site1_measured_time_delays[0][1]))
                    print('site2_p0 = %0.3f \t site2_measured_time_delays[0][1] = %0.3f \t diff = %0.3f'%(site2_p0, site2_measured_time_delays[0][1], site2_p0 - site2_measured_time_delays[0][1]))
                    print('site3_p0 = %0.3f \t site3_measured_time_delays[0][1] = %0.3f \t diff = %0.3f'%(site3_p0, site3_measured_time_delays[0][1], site3_p0 - site3_measured_time_delays[0][1]))
                    print('site1_p1 = %0.3f \t site1_measured_time_delays[1][1] = %0.3f \t diff = %0.3f'%(site1_p1, site1_measured_time_delays[1][1], site1_p1 - site1_measured_time_delays[1][1]))
                    print('site2_p1 = %0.3f \t site2_measured_time_delays[1][1] = %0.3f \t diff = %0.3f'%(site2_p1, site2_measured_time_delays[1][1], site2_p1 - site2_measured_time_delays[1][1]))
                    print('site3_p1 = %0.3f \t site3_measured_time_delays[1][1] = %0.3f \t diff = %0.3f'%(site3_p1, site3_measured_time_delays[1][1], site3_p1 - site3_measured_time_delays[1][1]))
                    print('site1_p2 = %0.3f \t site1_measured_time_delays[2][1] = %0.3f \t diff = %0.3f'%(site1_p2, site1_measured_time_delays[2][1], site1_p2 - site1_measured_time_delays[2][1]))
                    print('site2_p2 = %0.3f \t site2_measured_time_delays[2][1] = %0.3f \t diff = %0.3f'%(site2_p2, site2_measured_time_delays[2][1], site2_p2 - site2_measured_time_delays[2][1]))
                    print('site3_p2 = %0.3f \t site3_measured_time_delays[2][1] = %0.3f \t diff = %0.3f'%(site3_p2, site3_measured_time_delays[2][1], site3_p2 - site3_measured_time_delays[2][1]))
                    print('site1_p3 = %0.3f \t site1_measured_time_delays[3][1] = %0.3f \t diff = %0.3f'%(site1_p3, site1_measured_time_delays[3][1], site1_p3 - site1_measured_time_delays[3][1]))
                    print('site2_p3 = %0.3f \t site2_measured_time_delays[3][1] = %0.3f \t diff = %0.3f'%(site2_p3, site2_measured_time_delays[3][1], site2_p3 - site2_measured_time_delays[3][1]))
                    print('site3_p3 = %0.3f \t site3_measured_time_delays[3][1] = %0.3f \t diff = %0.3f'%(site3_p3, site3_measured_time_delays[3][1], site3_p3 - site3_measured_time_delays[3][1]))
                    print('site1_p4 = %0.3f \t site1_measured_time_delays[4][1] = %0.3f \t diff = %0.3f'%(site1_p4, site1_measured_time_delays[4][1], site1_p4 - site1_measured_time_delays[4][1]))
                    print('site2_p4 = %0.3f \t site2_measured_time_delays[4][1] = %0.3f \t diff = %0.3f'%(site2_p4, site2_measured_time_delays[4][1], site2_p4 - site2_measured_time_delays[4][1]))
                    print('site3_p4 = %0.3f \t site3_measured_time_delays[4][1] = %0.3f \t diff = %0.3f'%(site3_p4, site3_measured_time_delays[4][1], site3_p4 - site3_measured_time_delays[4][1]))
                    print('site1_p5 = %0.3f \t site1_measured_time_delays[5][1] = %0.3f \t diff = %0.3f'%(site1_p5, site1_measured_time_delays[5][1], site1_p5 - site1_measured_time_delays[5][1]))
                    print('site2_p5 = %0.3f \t site2_measured_time_delays[5][1] = %0.3f \t diff = %0.3f'%(site2_p5, site2_measured_time_delays[5][1], site2_p5 - site2_measured_time_delays[5][1]))
                    print('site3_p5 = %0.3f \t site3_measured_time_delays[5][1] = %0.3f \t diff = %0.3f'%(site3_p5, site3_measured_time_delays[5][1], site3_p5 - site3_measured_time_delays[5][1]))

                #This will weight against differences that result in longer baselines than measured.   I.e. smaller number if current baseline > measured.  Large for current < measured. 
                w = lambda measured, current : numpy.exp(measured - current)**2

                measured_baselines = {'01':129*0.3048,
                                      '02':163*0.3048,
                                      '03':181*0.3048,
                                      '12':151*0.3048,
                                      '13':102*0.3048,
                                      '23':85 *0.3048}

                current_baselines = {   '01':numpy.sqrt((ant0_x - ant1_x)**2 + (ant0_y - ant1_y)**2 + (ant0_z - ant1_z)**2),\
                                        '02':numpy.sqrt((ant0_x - ant2_x)**2 + (ant0_y - ant2_y)**2 + (ant0_z - ant2_z)**2),\
                                        '03':numpy.sqrt((ant0_x - ant3_x)**2 + (ant0_y - ant3_y)**2 + (ant0_z - ant3_z)**2),\
                                        '12':numpy.sqrt((ant1_x - ant2_x)**2 + (ant1_y - ant2_y)**2 + (ant1_z - ant2_z)**2),\
                                        '13':numpy.sqrt((ant1_x - ant3_x)**2 + (ant1_y - ant3_y)**2 + (ant1_z - ant3_z)**2),\
                                        '23':numpy.sqrt((ant2_x - ant3_x)**2 + (ant2_y - ant3_y)**2 + (ant2_z - ant3_z)**2)}

                baseline_weights = {  '01':w(measured_baselines['01'], current_baselines['01']),
                                      '02':w(measured_baselines['02'], current_baselines['02']),
                                      '03':w(measured_baselines['03'], current_baselines['03']),
                                      '12':w(measured_baselines['12'], current_baselines['12']),
                                      '13':w(measured_baselines['13'], current_baselines['13']),
                                      '23':w(measured_baselines['23'], current_baselines['23'])}

                #These are the measured height differences in antennas based on the GPS used for pulsing.
                #I am not using their absolute values, but I hope their relative differences will be representative of
                #reality.  There is no measurements with the same gps for the array so i'll just weight pulsers. 

                measured_pulser_relative_heights = {'1-2':3763.1-3690.7,\
                                                    '2-3':3690.7-3806.25,\
                                                    '1-3':3763.1-3806.25}
                measured_pulser_relative_heights_error = 5.0 #m

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
                            ((site3_p5 - site3_measured_time_delays[5][1])**2)/site3_measured_time_delays_errors[5][1]**2 + \
                            \
                            ((current_baselines['01'] - measured_baselines['01'])**2)/baseline_weights['01'] + \
                            ((current_baselines['02'] - measured_baselines['02'])**2)/baseline_weights['02'] + \
                            ((current_baselines['03'] - measured_baselines['03'])**2)/baseline_weights['03'] + \
                            ((current_baselines['12'] - measured_baselines['12'])**2)/baseline_weights['12'] + \
                            ((current_baselines['13'] - measured_baselines['13'])**2)/baseline_weights['13'] + \
                            ((current_baselines['23'] - measured_baselines['23'])**2)/baseline_weights['23'] + \
                            \
                            (((puls1_z - puls2_z) - (measured_pulser_relative_heights['1-2']))**2.0) / measured_pulser_relative_heights_error**2.0 +\
                            (((puls2_z - puls3_z) - (measured_pulser_relative_heights['2-3']))**2.0) / measured_pulser_relative_heights_error**2.0 +\
                            (((puls1_z - puls3_z) - (measured_pulser_relative_heights['1-3']))**2.0) / measured_pulser_relative_heights_error**2.0

                return chi_2
            except Exception as e:
                print('Error in f')
                print(e)
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(exc_type, fname, exc_tb.tb_lineno)


        initial_chi_2 = f(ant1_physical_x,ant1_physical_y,ant1_physical_z,ant2_physical_x,ant2_physical_y,ant2_physical_z,ant3_physical_x,ant3_physical_y,ant3_physical_z,site1_pulser_location[0],site1_pulser_location[1],site1_pulser_location[2],site2_pulser_location[0],site2_pulser_location[1],site2_pulser_location[2],site3_pulser_location[0],site3_pulser_location[1],site3_pulser_location[2])
        print('Initial Chi^2 is %0.3f'%(initial_chi_2))

        initial_step = 0.01 #m
        #-12 ft on pulser locations relative to antennas to account for additional mast elevation.
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
                        puls1_x=site1_pulser_location[0],\
                        puls1_y=site1_pulser_location[1],\
                        puls1_z=site1_pulser_location[2] - 12.0*0.3048,\
                        puls2_x=site2_pulser_location[0],\
                        puls2_y=site2_pulser_location[1],\
                        puls2_z=site2_pulser_location[2] - 12.0*0.3048,\
                        puls3_x=site3_pulser_location[0],\
                        puls3_y=site3_pulser_location[1],\
                        puls3_z=site3_pulser_location[2] - 12.0*0.3048,\
                        error_ant1_x=initial_step,\
                        error_ant1_y=initial_step,\
                        error_ant1_z=initial_step/2.0,\
                        error_ant2_x=initial_step,\
                        error_ant2_y=initial_step,\
                        error_ant2_z=initial_step/2.0,\
                        error_ant3_x=initial_step,\
                        error_ant3_y=initial_step,\
                        error_ant3_z=initial_step/2.0,\
                        error_puls1_x=initial_step,\
                        error_puls1_y=initial_step,\
                        error_puls1_z=initial_step,\
                        error_puls2_x=initial_step,\
                        error_puls2_y=initial_step,\
                        error_puls2_z=initial_step,\
                        error_puls3_x=initial_step,\
                        error_puls3_y=initial_step,\
                        error_puls3_z=initial_step,\
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
                        limit_puls1_x=puls1_x_limits,\
                        limit_puls1_y=puls1_y_limits,\
                        limit_puls1_z=puls1_z_limits,\
                        limit_puls2_x=puls2_x_limits,\
                        limit_puls2_y=puls2_y_limits,\
                        limit_puls2_z=puls2_z_limits,\
                        limit_puls3_x=puls3_x_limits,\
                        limit_puls3_y=puls3_y_limits,\
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

        puls1_phase_x = m.values['puls1_x']
        puls1_phase_y = m.values['puls1_y']
        puls1_phase_z = m.values['puls1_z']
        puls2_phase_x = m.values['puls2_x']
        puls2_phase_y = m.values['puls2_y']
        puls2_phase_z = m.values['puls2_z']
        puls3_phase_x = m.values['puls3_x']
        puls3_phase_y = m.values['puls3_y']
        puls3_phase_z = m.values['puls3_z']

        phase_locs = numpy.array([[ant0_phase_x,ant0_phase_y,ant0_phase_z],[ant1_phase_x,ant1_phase_y,ant1_phase_z],[ant2_phase_x,ant2_phase_y,ant2_phase_z],[ant3_phase_x,ant3_phase_y,ant3_phase_z]])

        print('Antenna Locations: \n%s'%str(antenna_locs))
        print('Phase Locations: \n%s'%str(phase_locs))

        print('\nDifference (antenna_locs - phase_locs): \n%s'%str(antenna_locs - phase_locs))

        print('\nSite 1 Physical Location: \n%s'%str((site1_pulser_location[0], site1_pulser_location[1], site1_pulser_location[2])))
        print('Site 1 Phase Location: \n%s'%str((puls1_phase_x, puls1_phase_y, puls1_phase_z)))

        print('\nSite 2 Physical Location: \n%s'%str((site2_pulser_location[0], site2_pulser_location[1], site2_pulser_location[2])))
        print('Site 2 Phase Location: \n%s'%str((puls2_phase_x, puls2_phase_y, puls2_phase_z)))

        print('\nSite 3 Physical Location: \n%s'%str((site3_pulser_location[0], site3_pulser_location[1], site3_pulser_location[2])))
        print('Site 3 Phase Location: \n%s'%str((puls3_phase_x, puls3_phase_y, puls3_phase_z)))

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


        print('Copy-Paste Prints:\n------------')
        print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, phase_locs[0][0],phase_locs[0][1],phase_locs[0][2],  phase_locs[1][0],phase_locs[1][1],phase_locs[1][2],  phase_locs[2][0],phase_locs[2][1],phase_locs[2][2],  phase_locs[3][0],phase_locs[3][1],phase_locs[3][2]))
        print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, 0.0 ,0.0 ,0.0 ,  m.errors['ant1_x'],m.errors['ant1_y'],m.errors['ant1_z'],  m.errors['ant2_x'],m.errors['ant2_y'],m.errors['ant2_z'],  m.errors['ant3_x'],m.errors['ant3_y'],m.errors['ant3_z']))

        print('pulser_locations_ENU[\'%s\'] = {\'run1507\':[%f, %f, %f], \'run1509\':[%f, %f, %f], \'run1511\':[%f, %f, %f]}'%(mode,m.values['puls1_x'], m.values['puls1_y'], m.values['puls1_z'], m.values['puls2_x'], m.values['puls2_y'], m.values['puls2_z'], m.values['puls3_x'], m.values['puls3_y'], m.values['puls3_z']))
        print('pulser_locations_ENU[\'%s_hesse_error\'] = {\'run1507\':[%f, %f, %f], \'run1509\':[%f, %f, %f], \'run1511\':[%f, %f, %f]}'%(mode,m.errors['puls1_x'], m.errors['puls1_y'], m.errors['puls1_z'], m.errors['puls2_x'], m.errors['puls2_y'], m.errors['puls2_z'], m.errors['puls3_x'], m.errors['puls3_y'], m.errors['puls3_z']))



        if True:
            colors = ['g','r','b','m']
            antennas_physical = info.loadAntennaLocationsENU(deploy_index=1)[0]

            fig = plt.figure()
            fig.canvas.set_window_title('Antenna Locations')
            ax = fig.add_subplot(111, projection='3d')

            for i, a in antennas_physical.items():
                ax.scatter(a[0], a[1], a[2], marker='o',color=colors[i],label='Physical %i'%i)

            for i, a in enumerate(phase_locs):
                ax.scatter(a[0], a[1], a[2], marker='*',color=colors[i],label='%s Phase Center %i'%(mode, i))

            pulser_locations = info.loadPulserPhaseLocationsENU()['physical']


            for site, key in enumerate(['run1507','run1509','run1511']):
                site += 1
                ax.scatter(pulser_locations[key][0], pulser_locations[key][1], pulser_locations[key][2], color='k', marker='o',label='Physical Pulser Site %i'%site)

            ax.plot([pulser_locations['run1507'][0],puls1_phase_x],[pulser_locations['run1507'][1],puls1_phase_y],[pulser_locations['run1507'][2],puls1_phase_z],linestyle='--')
            ax.scatter(puls1_phase_x, puls1_phase_y, puls1_phase_z, color='k', marker='*',label='Physical Pulser Site %i'%1)

            ax.plot([pulser_locations['run1509'][0],puls2_phase_x],[pulser_locations['run1509'][1],puls2_phase_y],[pulser_locations['run1509'][2],puls2_phase_z],linestyle='--')
            ax.scatter(puls2_phase_x, puls2_phase_y, puls2_phase_z, color='k', marker='*',label='Physical Pulser Site %i'%2)

            ax.plot([pulser_locations['run1511'][0],puls3_phase_x],[pulser_locations['run1511'][1],puls3_phase_y],[pulser_locations['run1511'][2],puls3_phase_z],linestyle='--')
            ax.scatter(puls3_phase_x, puls3_phase_y, puls3_phase_z, color='k', marker='*',label='Physical Pulser Site %i'%3)

            ax.set_xlabel('E (m)')
            ax.set_ylabel('N (m)')
            ax.set_zlabel('Relative Elevation (m)')
            plt.legend()


    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






