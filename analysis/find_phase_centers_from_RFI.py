'''
This script assumes the antenna positions have already roughly been set/determined
using find_phase_centers.py  .  It will assume these as starting points of the antennas
and then vary all 4 of their locations to match measured time delays from RFI sources in the valley. 
'''

#General Imports
import numpy
import itertools
import os
import sys
import csv
import scipy
import scipy.interpolate
import pymap3d as pm
from iminuit import Minuit
import inspect
import h5py

#Personal Imports
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.
sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
from tools.data_handler import createFile, getTimes
import tools.station as bc
import tools.info as info
import tools.get_plane_tracks as pt
from tools.fftmath import TimeDelayCalculator
from tools.data_slicer import dataSlicerSingleRun
from tools.correlator import Correlator

#Plotting Imports
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D


#Settings
from pprint import pprint
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
plt.ion()
datapath = os.environ['BEACON_DATA']

n = 1.0003 #Index of refraction of air  #Should use https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.453-11-201507-S!!PDF-E.pdf 
c = 299792458/n #m/s

#Settings for the current hilbert and "normal filtered waveforms"

#Pre measured time delays from pulsing sources.
#All use the following filters
# final_corr_length = 2**17 #Should be a factor of 2 for fastest performance
# apply_phase_response = True

# crit_freq_low_pass_MHz = [100,75,75,75,75,75,75,75]
# low_pass_filter_order = [8,12,14,14,14,14,14,14]

# crit_freq_high_pass_MHz = None#30#None#50
# high_pass_filter_order = None#5#None#8

# sine_subtract = True
# sine_subtract_min_freq_GHz = 0.03
# sine_subtract_max_freq_GHz = 0.09
# sine_subtract_percent = 0.03

# plot_filters = False
# plot_multiple = False

# hilbert = [VARIES, LABELED BELOW]
# align_method = 0

# shorten_signals = True
# shorten_thresh = 0.7
# shorten_delay = 10.0
# shorten_length = 90.0

#Settings for by eye and for when using vpol to determine hpol:

# crit_freq_low_pass_MHz = [80,70,70,70,70,70,60,70] #Filters here are attempting to correct for differences in signals from pulsers.
# low_pass_filter_order = [0,8,8,8,10,8,3,8]

# crit_freq_high_pass_MHz = 65
# high_pass_filter_order = 12

# sine_subtract = True
# sine_subtract_min_freq_GHz = 0.03
# sine_subtract_max_freq_GHz = 0.13
# sine_subtract_percent = 0.03


print('IF USING TIME DELAYS FOR PULSERS, I TRUST CURRENT BY EYE VERSION')

# Site 1, Run 1507
#Using normal filtered waveforms
# site1_measured_time_delays_hpol =  [((0, 1), -54.73863562760832), ((0, 2), 91.15858599870768), ((0, 3), 2.2128385205326464), ((1, 2), 145.8794202783432), ((1, 3), 70.83219696885855), ((2, 3), -75.0806062508086)]
# site1_measured_time_delays_errors_hpol =  [((0, 1), 0.06502425258486934), ((0, 2), 0.04578149270315245), ((0, 3), 0.05552697804026172), ((1, 2), 0.04285045065819871), ((1, 3), 0.0502413434940142), ((2, 3), 0.03943499232368893)]
# site1_measured_time_delays_vpol =  [((0, 1), -41.34298309480882), ((0, 2), 98.41416470888946), ((0, 3), 32.9834991717602), ((1, 2), 139.76496652601807), ((1, 3), 74.35728021223598), ((2, 3), -65.41722156408288)]
# site1_measured_time_delays_errors_vpol =  [((0, 1), 0.042884363603573825), ((0, 2), 0.0441559772370423), ((0, 3), 0.03848571442688579), ((1, 2), 0.04704966892964731), ((1, 3), 0.04630379764018927), ((2, 3), 0.045227795080871123)]

#By eye with heavy filtering
# site1_measured_time_delays_hpol =  [((0, 1), -54.969833107118845), ((0, 2), 87.14009967681076), ((0, 3), 24.00177447738297), ((1, 2), 142.14047250017052), ((1, 3), 79.05998159730048), ((2, 3), -63.10231911641529)]
# site1_measured_time_delays_errors_hpol =  [((0, 1), 0.05397614123500611), ((0, 2), 0.04512309121971968), ((0, 3), 0.05204283012142891), ((1, 2), 0.03087221688563602), ((1, 3), 0.032053827917602244), ((2, 3), 0.028770235190528414)]
# site1_measured_time_delays_vpol =  [((0, 1), -38.0395259095481), ((0, 2), 101.66276202065728), ((0, 3), 36.28366947068722), ((1, 2), 139.71981822328297), ((1, 3), 74.38132870251398), ((2, 3), -65.37132308735707)]
# site1_measured_time_delays_errors_vpol =  [((0, 1), 0.03821252883631024), ((0, 2), 0.03683462360960683), ((0, 3), 0.0371499730741353), ((1, 2), 0.04214783336728045), ((1, 3), 0.03760640106500705), ((2, 3), 0.03780978324309525)]
#By eye unfiltered
site1_measured_time_delays_hpol =  [((0, 1), -40.31873095761775), ((0, 2), 105.45784514533067), ((0, 3), 30.135050738939075), ((1, 2), 145.61788076163322), ((1, 3), 70.58430198762721), ((2, 3), -75.18303704634458)]
site1_measured_time_delays_errors_hpol =  [((0, 1), 0.05390602875058418), ((0, 2), 0.044986501385526165), ((0, 3), 0.04766074318950998), ((1, 2), 0.028887890879186494), ((1, 3), 0.03252021044637105), ((2, 3), 0.03225428496045381)]
site1_measured_time_delays_vpol =  [((0, 1), -37.997470132440384), ((0, 2), 101.68917969316817), ((0, 3), 36.40724396999149), ((1, 2), 139.68684427937058), ((1, 3), 74.50112508391489), ((2, 3), -65.25487641338805)]
site1_measured_time_delays_errors_vpol =  [((0, 1), 0.03011827820152272), ((0, 2), 0.03019736343475925), ((0, 3), 0.031375107623472516), ((1, 2), 0.032447697682718275), ((1, 3), 0.03109667967812645), ((2, 3), 0.031271228554630597)]

# Site 2, Run 1509
#Using normal filtered waveforms
# site2_measured_time_delays_hpol =  [((0, 1), -96.52211468189127), ((0, 2), 26.44864606474854), ((0, 3), -72.54565055659056), ((1, 2), 123.1494860790839), ((1, 3), 38.24363584514595), ((2, 3), -85.05175803438193)]
# site2_measured_time_delays_errors_hpol =  [((0, 1), 0.08249898683406354), ((0, 2), 0.07575714414002656), ((0, 3), 0.07207545260984918), ((1, 2), 0.07601506755721471), ((1, 3), 0.08100775284944606), ((2, 3), 0.04962571327194129)]
# site2_measured_time_delays_vpol =  [((0, 1), -83.19953242338167), ((0, 2), 33.788067132019236), ((0, 3), -41.64972472448962), ((1, 2), 116.94909535917708), ((1, 3), 41.5594035217089), ((2, 3), -75.40642816197486)]
# site2_measured_time_delays_errors_vpol =  [((0, 1), 0.0587727059708651), ((0, 2), 0.06416229949778407), ((0, 3), 0.057996504388612015), ((1, 2), 0.06429783537463728), ((1, 3), 0.05940387444310559), ((2, 3), 0.05479525148189039)]

#Starting with vpol time delays for hpol
# site2_measured_time_delays_hpol =  [((0, 1), -83.4350299273764), ((0, 2), 35.98735001057835), ((0, 3), -37.227818605847155), ((1, 2), 119.48454452242105), ((1, 3), 46.314350011875135), ((2, 3), -73.19628700828557)]
# site2_measured_time_delays_errors_hpol =  [((0, 1), 0.08100363991429034), ((0, 2), 0.08840029523525751), ((0, 3), 0.07553940454510318), ((1, 2), 0.06814144543927382), ((1, 3), 0.05585941670256171), ((2, 3), 0.04175515290975236)]
# site2_measured_time_delays_vpol =  [((0, 1), -79.9254869986498), ((0, 2), 36.95016873421155), ((0, 3), -38.37130874987727), ((1, 2), 116.82930650572263), ((1, 3), 41.60022844836235), ((2, 3), -75.29534858596858)]
# site2_measured_time_delays_errors_vpol =  [((0, 1), 0.05470476738778757), ((0, 2), 0.05120618401181174), ((0, 3), 0.04941877318633543), ((1, 2), 0.061857288229529614), ((1, 3), 0.05130304646372132), ((2, 3), 0.04567102070600878)]
#By eye with heavy filtering
# site2_measured_time_delays_hpol =  [((0, 1), -96.97522838255696), ((0, 2), 8.739083570682284), ((0, 3), -50.80788386992672), ((1, 2), 105.83015717556633), ((1, 3), 46.314350011875135), ((2, 3), -73.19628700828557)]
# site2_measured_time_delays_errors_hpol =  [((0, 1), 0.06347605396637605), ((0, 2), 0.06380476954335818), ((0, 3), 0.06187583322961239), ((1, 2), 0.06269502641508909), ((1, 3), 0.05585941670256171), ((2, 3), 0.04175515290975236)]
# site2_measured_time_delays_vpol =  [((0, 1), -79.92548699865343), ((0, 2), 36.950168734211644), ((0, 3), -38.37130874987719), ((1, 2), 116.82930650572267), ((1, 3), 41.60022844836235), ((2, 3), -75.29534858596858)]
# site2_measured_time_delays_errors_vpol =  [((0, 1), 0.05470476739678275), ((0, 2), 0.05120618401166019), ((0, 3), 0.04941877318949003), ((1, 2), 0.061857288229529614), ((1, 3), 0.05130304646372132), ((2, 3), 0.04567102070600878)]
#By eye with no filtering
site2_measured_time_delays_hpol =  [((0, 1), -82.36300952477696), ((0, 2), 40.532241178254594), ((0, 3), -44.62321034858617), ((1, 2), 122.63983003115614), ((1, 3), 37.84008150548918), ((2, 3), -85.10254711335975)]
site2_measured_time_delays_errors_hpol =  [((0, 1), 0.06805633356582455), ((0, 2), 0.06374115194036371), ((0, 3), 0.061932184861914215), ((1, 2), 0.05385988241825421), ((1, 3), 0.06333087652874915), ((2, 3), 0.04576242749045018)]
site2_measured_time_delays_vpol =  [((0, 1), -79.95386249566438), ((0, 2), 36.76801256475748), ((0, 3), -38.345785064895686), ((1, 2), 116.64786424329235), ((1, 3), 41.7113458359634), ((2, 3), -75.08797122311996)]
site2_measured_time_delays_errors_vpol =  [((0, 1), 0.042075284218763505), ((0, 2), 0.04115999677119886), ((0, 3), 0.03731577983391688), ((1, 2), 0.04697815701549882), ((1, 3), 0.037159371366881265), ((2, 3), 0.03363016431350404)]

# Site 3, Run 1511
site3_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.03654878810011606), ((0, 2), 0.044076823274307404), ((0, 3), 0.04115323593715623), ((1, 2), 0.039281585194972476), ((1, 3), 0.04240838492773853), ((2, 3), 0.04449409255674249)]

#Using normal filtered waveforms
# site3_measured_time_delays_hpol =  [((0, 1), -109.18511106059067), ((0, 2), -158.10857451624145), ((0, 3), -191.41010115062258), ((1, 2), -48.875126824337244), ((1, 3), -82.20492349641286), ((2, 3), -33.23554377320981)]
# site3_measured_time_delays_errors_hpol =  [((0, 1), 0.052428389377522186), ((0, 2), 0.07196071815111829), ((0, 3), 0.057841508523156095), ((1, 2), 0.05836839659836438), ((1, 3), 0.06144960035833733), ((2, 3), 0.08874905433961774)]
# site3_measured_time_delays_vpol =  [((0, 1), -95.30265032203047), ((0, 2), -150.3612627337264), ((0, 3), -175.88155062409197), ((1, 2), -55.06965132126639), ((1, 3), -80.55266965411217), ((2, 3), -25.524227125491926)]
# site3_measured_time_delays_errors_vpol =  [((0, 1), 0.03654878810011606), ((0, 2), 0.044076823274307404), ((0, 3), 0.04115323593715623), ((1, 2), 0.039281585194972476), ((1, 3), 0.04240838492773853), ((2, 3), 0.04449411155669218)]

#By eye with heavy filtering
# site3_measured_time_delays_hpol =  [((0, 1), -109.34183150535519), ((0, 2), -162.03789621141365), ((0, 3), -183.21756927506613), ((1, 2), -52.62823300992422), ((1, 3), -73.79391408829574), ((2, 3), -21.183744364934398)]
# site3_measured_time_delays_errors_hpol =  [((0, 1), 0.04265253188274841), ((0, 2), 0.06423203676109482), ((0, 3), 0.04702481712409536), ((1, 2), 0.04856625123399987), ((1, 3), 0.04080526421433228), ((2, 3), 0.05989498787455408)]
# site3_measured_time_delays_vpol =  [((0, 1), -92.01039256757632), ((0, 2), -147.11633863997915), ((0, 3), -172.54667944061194), ((1, 2), -55.10074275670341), ((1, 3), -80.47338613712525), ((2, 3), -25.426023438675795)]
# site3_measured_time_delays_errors_vpol =  [((0, 1), 0.03342246038492944), ((0, 2), 0.03848447654781584), ((0, 3), 0.039030703182496955), ((1, 2), 0.036584465848776934), ((1, 3), 0.037736362321036596), ((2, 3), 0.038955535978780334)]

#By eye with no filtering
site3_measured_time_delays_hpol =  [((0, 1), -94.60387463826103), ((0, 2), -143.57675835202508), ((0, 3), -177.08153009135634), ((1, 2), -49.07507885269845), ((1, 3), -82.25996187697748), ((2, 3), -33.23172107457927)]
site3_measured_time_delays_errors_hpol =  [((0, 1), 0.050722621463835826), ((0, 2), 0.0668571644522086), ((0, 3), 0.054427178270967415), ((1, 2), 0.0460384615661204), ((1, 3), 0.03794985949273057), ((2, 3), 0.06075860733801835)]
site3_measured_time_delays_vpol =  [((0, 1), -92.0411188138037), ((0, 2), -147.12574991663104), ((0, 3), -172.41237810521784), ((1, 2), -55.093716852263405), ((1, 3), -80.26283429217344), ((2, 3), -25.26536153179327)]
site3_measured_time_delays_errors_vpol =  [((0, 1), 0.026686488662462645), ((0, 2), 0.029657613098382718), ((0, 3), 0.03176185523762114), ((1, 2), 0.029839183454581412), ((1, 3), 0.02891810009497307), ((2, 3), 0.030919678248785214)]

#Starting with vpol time delays for hpol
# site3_measured_time_delays_hpol =  [((0, 1), -96.09672657178147), ((0, 2), -148.73076067727396), ((0, 3), -169.93956133124348), ((1, 2), -52.62823300992422), ((1, 3), -73.79391408829574), ((2, 3), -21.183744364934398)]
# site3_measured_time_delays_errors_hpol =  [((0, 1), 0.039552441264012354), ((0, 2), 0.06278263246667376), ((0, 3), 0.047063796245871106), ((1, 2), 0.04856625123399987), ((1, 3), 0.04080526421433228), ((2, 3), 0.05989498787455408)]
# site3_measured_time_delays_vpol =  [((0, 1), -92.0103925675772), ((0, 2), -147.11633863997926), ((0, 3), -172.54667944061526), ((1, 2), -55.10074275670341), ((1, 3), -80.47338613712525), ((2, 3), -25.426023438675795)]
# site3_measured_time_delays_errors_vpol =  [((0, 1), 0.03342246036574039), ((0, 2), 0.03848447653883523), ((0, 3), 0.03903070321188336), ((1, 2), 0.036584465848776934), ((1, 3), 0.037736362321036596), ((2, 3), 0.038955535978780334)]





print('IF USING HILBERT TIME DELAYS FOR PULSERS, I DONT TRUST CURRENT VERSION')

# Site 1, Run 1507
#Using hilbert enevelopes:
site1_measured_time_delays_hilbert_hpol =  [((0, 1), -58.987476762558636), ((0, 2), 85.65448689743364), ((0, 3), 8.320951597921127), ((1, 2), 144.48438341182535), ((1, 3), 67.16676381542445), ((2, 3), -77.3644536438967)]
site1_measured_time_delays_hilbert_errors_hpol =  [((0, 1), 0.8377448773505135), ((0, 2), 0.3594185551190815), ((0, 3), 0.594635989964824), ((1, 2), 0.7881229445755795), ((1, 3), 0.655179094065839), ((2, 3), 0.5535397424928563)]
site1_measured_time_delays_hilbert_vpol =  [((0, 1), -41.7904293292381), ((0, 2), 97.99566415134592), ((0, 3), 32.98832571271674), ((1, 2), 139.79330184195436), ((1, 3), 74.75836427079234), ((2, 3), -65.04301010744241)]
site1_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.4735699881401701), ((0, 2), 0.4590523544465075), ((0, 3), 0.42013177070803437), ((1, 2), 0.6012971718190752), ((1, 3), 0.47273363718795347), ((2, 3), 0.4704350387698477)]

# Site 2, Run 1509
#Using hilbert envelopes:
site2_measured_time_delays_hilbert_hpol =  [((0, 1), -100.60440248833073), ((0, 2), 18.252306266584625), ((0, 3), -68.1176724682709), ((1, 2), 118.43971845333104), ((1, 3), 32.27161618415478), ((2, 3), -86.74537359631644)]
site2_measured_time_delays_hilbert_errors_hpol =  [((0, 1), 0.8957178121617457), ((0, 2), 1.016165870957765), ((0, 3), 0.7663702873722121), ((1, 2), 1.400179720774765), ((1, 3), 1.0146094407051107), ((2, 3), 0.9757739190044133)]
site2_measured_time_delays_hilbert_vpol =  [((0, 1), -83.84057479839511), ((0, 2), 29.220940734890714), ((0, 3), -42.61904475811045), ((1, 2), 113.34806469515324), ((1, 3), 41.375706449073355), ((2, 3), -71.86012025765733)]
site2_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.6408498537052976), ((0, 2), 0.6485697868214796), ((0, 3), 0.5952727184008998), ((1, 2), 0.7658668502667069), ((1, 3), 0.6880201767149771), ((2, 3), 0.6428620412909642)]

# Site 3, Run 1511
#Using hilbert envelopes:
site3_measured_time_delays_hilbert_hpol =  [((0, 1), -109.18511106059067), ((0, 2), -158.10857451624145), ((0, 3), -191.41010115062258), ((1, 2), -48.875126824337244), ((1, 3), -82.20492349641286), ((2, 3), -33.23548270662396)]
site3_measured_time_delays_hilbert_errors_hpol =  [((0, 1), 0.052428389377522186), ((0, 2), 0.07196071815111829), ((0, 3), 0.057841508523156095), ((1, 2), 0.05836839659836438), ((1, 3), 0.06144960035833733), ((2, 3), 0.08874875693196994)]
site3_measured_time_delays_hilbert_vpol =  [((0, 1), -95.30265032203047), ((0, 2), -150.3612627337264), ((0, 3), -175.88155062409197), ((1, 2), -55.06965132126639), ((1, 3), -80.55266965411217), ((2, 3), -25.524379719980214)]
site3_measured_time_delays_hilbert_errors_vpol =  [((0, 1), 0.03654878810011606), ((0, 2), 0.044076823274307404), ((0, 3), 0.04115323593715623), ((1, 2), 0.039281585194972476), ((1, 3), 0.04240838492773853), ((2, 3), 0.04449409255674249)]







if __name__ == '__main__':
    try:
        plt.close('all')
        if len(sys.argv) == 2:
            if str(sys.argv[1]) in ['vpol', 'hpol']:
                mode = str(sys.argv[1])
            else:
                print('Given mode not in options.  Defaulting to hpol')
                mode = 'hpol'
        else:
            print('No mode given.  Defaulting to hpol')
            mode = 'hpol'

        #palmetto is a reflection?
        #Need to label which ones work for vpol and hpol
        #use_sources = ['Solar Plant','Quarry Substation','Tonopah KTPH','Dyer Cell Tower','Beatty Airport Vortac','Palmetto Cell Tower','Cedar Peak','Goldfield Hill Tower','Goldield Town Tower','Goldfield KGFN-FM','Silver Peak Town Antenna','Silver Peak Lithium Mine','Past SP Substation','Silver Peak Substation']
        
        # 'Solar Plant Substation'
        # Some crosspol. definitely mostly hpol

        # 'Dyer or Tonopah'
        # Very crosspol, good for both calibrations

        # 'Beatty Airport VORTAC'
        # Should work for vpol

        # 'Silver Peak or Distant Substation'
        # Basically no vpol

        # 'Palmetto Tower'
        # Cross pol, not the most impulsive

        # 'Goldfield Radio'
        # Impulsive cross pol, better for hpol though
        valley_source_run = 1650
        if False:
            pulser_weight = 1.0 #Each pulsing site worth this % as much as a valley source.
            # Southern Calibration
            unknown_source_dir_valley = False #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            if mode == 'hpol':
                use_sources = ['Solar Plant']#,'Beatty Substation','Palmetto Cell Tower', 'Cedar Peak'] #Southern Calibration
                included_pulsers = ['run1507','run1509','run1511'] #Only included if include_pulsers == True
            elif mode == 'vpol':
                use_sources = ['East Dyer Substation','Goldfield KGFN-FM']
                included_pulsers = ['run1507','run1509','run1511'] #Only included if include_pulsers == True
        elif False:
            pulser_weight = 1.0 #Each pulsing site worth this % as much as a valley source.
            # Southern Calibration
            unknown_source_dir_valley = False #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            if mode == 'hpol':
                use_sources = ['East Dyer Substation','Goldfield KGFN-FM','Silver Peak Substation']#,'Beatty Substation','Palmetto Cell Tower', 'Cedar Peak'] #Southern Calibration
                included_pulsers = ['run1507','run1509'] #Only included if include_pulsers == True
            elif mode == 'vpol':
                use_sources = ['East Dyer Substation','Goldfield KGFN-FM']
                included_pulsers = ['run1507','run1509'] #Only included if include_pulsers == True
        elif False:
            pulser_weight = 1.0 #Each pulsing site worth this % as much as a valley source.
            unknown_source_dir_valley = False #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            if mode == 'hpol':
                # Northern Calibration
                #use_sources = ['Tonopah KTPH','Solar Plant','Silver Peak Substation']#,'Beatty Substation','Palmetto Cell Tower', 'Cedar Peak'] #Southern Calibration
                use_sources = ['Tonopah KTPH','Solar Plant']
                included_pulsers = ['run1507','run1509']#['run1511'] #Only included if include_pulsers == True
            elif mode == 'vpol':
                use_sources = ['Tonopah KTPH','Solar Plant']
                included_pulsers = ['run1507','run1509']#['run1511'] #Only included if include_pulsers == True
        
        elif True:
            pulser_weight = 0.0 #Each pulsing site worth this % as much as a valley source.
            unknown_source_dir_valley = False #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            if mode == 'hpol':
                use_sources = ['Tonopah Vortac','Tonopah AFS GATR Site']#['East Dyer Substation','Goldfield KGFN-FM','Tonopah KTPH','Solar Plant','Silver Peak Substation']#['Tonopah KTPH','Solar Plant','Silver Peak Substation']#'East Dyer Substation',
                included_pulsers = ['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1509']#['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1507','run1509','run1511'] #Only included if include_pulsers == True
            elif mode == 'vpol':
                use_sources = ['Tonopah Vortac','Tonopah AFS GATR Site']#'East Dyer Substation',
                included_pulsers = ['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1507','run1509','run1511'] #Only included if include_pulsers == True
        elif False:
            pulser_weight = 1.0 #Each pulsing site worth this % as much as a valley source.
            unknown_source_dir_valley = True #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            unique_cut_subset = ['Solar Plant','Tonopah KTPH','West Dyer Substation','East Dyer Substation','Beatty Mountain Cell Tower','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation'] #these are all uniquely identified clusters.  Their names are not certain sources
            if mode == 'hpol':
                use_sources = ['Solar Plant','Tonopah KTPH','West Dyer Substation','East Dyer Substation','Beatty Mountain Cell Tower','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation']#['Solar Plant','Tonopah KTPH','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation']#['East Dyer Substation','Goldfield KGFN-FM','Tonopah KTPH','Solar Plant','Silver Peak Substation']#['Tonopah KTPH','Solar Plant','Silver Peak Substation']#'East Dyer Substation',
                included_pulsers = ['run1507','run1509','run1511']#['run1509']#['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1507','run1509','run1511'] #Only included if include_pulsers == True
            elif mode == 'vpol':
                use_sources = ['East Dyer Substation']#'East Dyer Substation',
                included_pulsers = ['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1507','run1509','run1511'] #Only included if include_pulsers == True
        elif False:
            pulser_weight = 1.0 #Each pulsing site worth this % as much as a valley source.
            unknown_source_dir_valley = False #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            unique_cut_subset = ['Solar Plant','Tonopah KTPH','West Dyer Substation','East Dyer Substation','Beatty Mountain Cell Tower','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation'] #these are all uniquely identified clusters.  Their names are not certain sources
            if mode == 'hpol':
                use_sources = ['Solar Plant']#['Solar Plant','Tonopah KTPH','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation']#['East Dyer Substation','Goldfield KGFN-FM','Tonopah KTPH','Solar Plant','Silver Peak Substation']#['Tonopah KTPH','Solar Plant','Silver Peak Substation']#'East Dyer Substation',
                included_pulsers = ['run1507','run1509','run1511']#['run1509']#['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1507','run1509','run1511'] #Only included if include_pulsers == True
            elif mode == 'vpol':
                use_sources = ['Solar Plant']#'East Dyer Substation',
                included_pulsers = ['run1507','run1509','run1511']#['run1507','run1509','run1511']#['run1507','run1509','run1511'] #Only included if include_pulsers == True
        else:
            pulser_weight = 1.0 #Each pulsing site worth this % as much as a valley source.
            unknown_source_dir_valley = False #If true then the chi^2 will not assume known arrival directions, but will instead just attempt to get overlap ANYWHERE for all selected populations.
            use_sources = [] #iterating through potential sources until something makes sense
            included_pulsers = ['run1507','run1509','run1511']#,'run1511' #Only included if include_pulsers == True

        #only_plot = use_sources
        
        only_plot   = [ 'Tonopah AFS GATR Site',\
                        'Tonopah Vortac']

        only_plot   = [ 'Tonopah AFS GATR Site',\
                        'Tonopah Vortac',\
                        'Dyer Cell Tower',\
                        'West Dyer Substation',\
                        'East Dyer Substation',\
                        'Beatty Substation',\
                        'Palmetto Cell Tower',\
                        'Black Mountain',\
                        'Cedar Peak',\
                        'Goldfield Hill Tower',\
                        'Silver Peak Substation']

        # only_plot   = [ 'Solar Plant',\
        #                 'Tonopah KTPH',\
        #                 'Dyer Cell Tower',\
        #                 'West Dyer Substation',\
        #                 'East Dyer Substation',\
        #                 'Beatty Airport Vortac',\
        #                 'Beatty Substation',\
        #                 'Palmetto Cell Tower',\
        #                 'Black Mountain',\
        #                 'Cedar Peak',\
        #                 'Goldfield Hill Tower',\
        #                 'Goldield Town Tower',\
        #                 'Goldfield KGFN-FM',\
        #                 'Silver Peak Town Antenna',\
        #                 'Silver Peak Lithium Mine',\
        #                 'Past SP Substation',\
        #                 'Silver Peak Substation']
        #only_plot = use_sources
        
        #only_plot.append('Beatty Mountain Cell Tower')
        # only_plot.append('Goldfield Hill Tower')
        # only_plot.append('Goldield Town Tower')
        # only_plot.append('Goldfield KGFN-FM')
        # only_plot.append('Beatty Airport Vortac')
        # only_plot.append('Beatty Airport Antenna')
        # only_plot.append('Beatty Substation')

        # Azimuths
        # Northern Cell Tower : 49.393
        # Quarry Substation : 46.095
        # Solar Plant : 43.441
        # Tonopah KTPH : 30.265
        # Dyer Cell Tower : 29.118
        # Tonopah Vortac : 25.165
        # Silver Peak Lithium Mine : 19.259
        # Silver Peak Substation : 19.112
        # Silver Peak Town Antenna : 18.844
        # Past SP Substation : 18.444
        # Goldfield Hill Tower : 10.018
        # Goldield Town Tower : 8.967
        # Goldfield KGFN-FM : 8.803
        # Cedar Peak : 4.992
        # West Dyer Substation : 3.050
        # Black Mountain : -13.073
        # Palmetto Cell Tower : -13.323
        # East Dyer Substation : -17.366
        # Beatty Substation : -29.850
        # Beatty Airport Antenna : -31.380
        # Beatty Airport Vortac : -33.040



        plot_time_delay_calculations = False
        plot_time_delays_on_maps = True
        plot_expected_direction = True
        limit_events = 10 #Number of events use for time delay calculation
        #only_plot = ['East Dyer Substation','West Dyer Substation','Northern Cell Tower','Solar Plant','Quarry Substation','Tonopah KTPH','Dyer Cell Tower','Tonopah Vortac','Beatty Airport Vortac','Palmetto Cell Tower','Cedar Peak','Goldfield Hill Tower','Goldield Town Tower','Goldfield KGFN-FM','Silver Peak Town Antenna','Silver Peak Lithium Mine','Past SP Substation','Silver Peak Substation']#['Solar Plant','Tonopah KTPH','Beatty Airport Vortac','Palmetto Cell Tower','Goldfield Hill Tower','Silver Peak Substation']
        impulsivity_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        time_delays_dset_key = 'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_0-corlen_65536-align_0-shortensignals-0-shortenthresh-0.70-shortendelay-10.00-shortenlength-90.00-sinesubtract_1'
        map_direction_dset_key = 'LPf_70.0-LPo_4-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0'#'LPf_100.0-LPo_8-HPf_None-HPo_None-Phase_1-Hilb_1-upsample_32768-maxmethod_0-sinesubtract_1'

        plot_residuals = False
        plot_histograms = False
        iterate_sub_baselines = 3 #The lower this is the higher the time it will take to plot.  Does combinatoric subsets of baselines with this length. 
        #plot_time_delays_on_maps = False

        final_corr_length = 2**17
        cor_upsample = final_corr_length

        crit_freq_low_pass_MHz = None#[80,70,70,70,70,70,60,70]#90#
        low_pass_filter_order = None#[0,8,8,8,10,8,3,8]#8#

        crit_freq_high_pass_MHz = None#70#None#60
        high_pass_filter_order = None#6#None#8

        sine_subtract = False
        sine_subtract_min_freq_GHz = 0.03
        sine_subtract_max_freq_GHz = 0.13
        sine_subtract_percent = 0.03

        waveform_index_range = (None,None)

        apply_phase_response = True
        hilbert = False

        include_pulsers = True 
        include_baseline_measurements = False
        baseline_measurement_uncertainty_m = 3 #Assuming a 3m spread in our data.  This is very approximate.
        time_delay_measurement_uncertainty_ns = 25 #ns, The time window used to as error in chi^2 for time delay.  If you are assuming that the time delays are 100% accurate then this is usually sub ns.  But if you think it is slipping cycles you could give this a larger value. 
        include_sanity = False #Slow
        plot_predicted_time_shifts = False
        random_offset_amount = 0.25 #m (every antenna will be stepped randomly by this amount.  Set to 0 if you don't want this. ), Note that this is applied to 
        included_antennas_lumped = [0,1,2,3] #If an antenna is not in this list then it will not be included in the chi^2 (regardless of if it is fixed or not)  Lumped here imlies that antenna 0 in this list means BOTH channels 0 and 1 (H and V of crossed dipole antenna 0).
        included_antennas_channels = numpy.concatenate([[2*i,2*i+1] for i in included_antennas_lumped])
        include_baselines = [0,1,2,3,4,5] #Basically sets the starting condition of which baselines to include, then the lumped channels and antennas will cut out further from that.  The above options of excluding antennas will override this to exclude baselines, but if both antennas are included but the baseline is not then it will not be included.  Overwritten when antennas removed.
        plot_overlap = True #Will plot the overlap map for time delays from each source.
        overlap_window_ns = 50 #ns The time window used to define sufficient overlap. 
        overlap_goal = overlap_window_ns*len(included_antennas_channels)*len(use_sources) #This shouldn't be varied, vary the error if anything.  This is the portion of chi^2 coming from overlapping valley source time delays.  The measured map max will be subtracted from this in a chi^2 calculation.  
        overlap_error = overlap_goal/50 #The error portion of chi^2 coming from overlapping valley source time delays will be devided by this number.
        limit_array_plane_azimuth_range = False #Should be seen as a temporary test.  Doesn't use any errors and isn't in standard chi^2 format.
        allowed_array_plane_azimuth_range = 20 #plus or minus this from East is not impacted by weighting. 

        #Limits 
        initial_step_x = 0.25#75 #m
        initial_step_y = 0.75#75 #m
        initial_step_z = 0.75#5 #m
        initial_step_cable_delay = 1.0 #ns
        cable_delay_guess_range = 10 #ns
        antenna_position_guess_range_x = 1#2#4 #Limit to how far from input phase locations to limit the parameter space to
        antenna_position_guess_range_y = 1#2#7 #Limit to how far from input phase locations to limit the parameter space to
        antenna_position_guess_range_z = 1#3 #Limit to how far from input phase locations to limit the parameter space to

        #Manually shifting input of antenna 0 around so that I can find a fit that has all of its baselines visible for valley sources. 
        manual_offset_ant0_x = 14#6
        manual_offset_ant0_y = -2.7#15
        manual_offset_ant0_z = -17#-4

        fix_ant0_x = False
        fix_ant0_y = False
        fix_ant0_z = False
        fix_ant1_x = False
        fix_ant1_y = False
        fix_ant1_z = False
        fix_ant2_x = False
        fix_ant2_y = False
        fix_ant2_z = False
        fix_ant3_x = False
        fix_ant3_y = False
        fix_ant3_z = False
        fix_cable_delay0 = False
        fix_cable_delay1 = False
        fix_cable_delay2 = False
        fix_cable_delay3 = False

        #Force antennas not to be included to be fixed.  
        if not(0 in included_antennas_lumped):
            fix_ant0_x = True
            fix_ant0_y = True
            fix_ant0_z = True
        if not(1 in included_antennas_lumped):
            fix_ant1_x = True
            fix_ant1_y = True
            fix_ant1_z = True
        if not(2 in included_antennas_lumped):
            fix_ant2_x = True
            fix_ant2_y = True
            fix_ant2_z = True
        if not(3 in included_antennas_lumped):
            fix_ant3_x = True
            fix_ant3_y = True
            fix_ant3_z = True


        #I think adding an absolute time offset for each antenna and letting that vary could be interesting.  It could be used to adjust the cable delays.
        cable_delays = info.loadCableDelays()[mode]
        print('Potential RFI Source Locations.')
        sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU()


        print('Source Directions based on ENU, sorted North to South are:')

        keys = list(sources_ENU.keys())
        azimuths = []

        for source_key in keys:
            azimuths.append(numpy.rad2deg(numpy.arctan2(sources_ENU[source_key][1],sources_ENU[source_key][0])))

        sort_cut = numpy.argsort(azimuths)[::-1]
        for index in sort_cut:
            print('%s : %0.3f'%(keys[index], azimuths[index]))

        pulser_locations_ENU = info.loadPulserLocationsENU()
        known_pulser_ids = info.loadPulserEventids(remove_ignored=True)
        if include_pulsers:
            pulser_time_delay_dict = {}
            pulser_time_delay_error_dict = {}
            if mode == 'hpol':
                if hilbert == True:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_hilbert_hpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_hilbert_errors_hpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_hilbert_hpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_hilbert_errors_hpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_hilbert_hpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_hilbert_errors_hpol
                else:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_hpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_errors_hpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_hpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_errors_hpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_hpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_errors_hpol
            else:
                if hilbert == True:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_hilbert_vpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_hilbert_errors_vpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_hilbert_vpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_hilbert_errors_vpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_hilbert_vpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_hilbert_errors_vpol
                else:
                    pulser_time_delay_dict['run1507'] = site1_measured_time_delays_vpol
                    pulser_time_delay_error_dict['run1507'] = site1_measured_time_delays_errors_vpol

                    pulser_time_delay_dict['run1509'] = site2_measured_time_delays_vpol
                    pulser_time_delay_error_dict['run1509'] = site2_measured_time_delays_errors_vpol

                    pulser_time_delay_dict['run1511'] = site3_measured_time_delays_vpol
                    pulser_time_delay_error_dict['run1511'] = site3_measured_time_delays_errors_vpol
        

            if plot_predicted_time_shifts:
                for run in [1507,1509,1511]:
                    if run == 1507:
                            _waveform_index_range = (1500,2000) #Looking at the later bit of the waveform only, 10000 will cap off.  
                    elif run == 1509:
                            _waveform_index_range = (2500,3000) #Looking at the later bit of the waveform only, 10000 will cap off.  
                    elif run == 1511:
                            _waveform_index_range = (1250,1750) #Looking at the later bit of the waveform only, 10000 will cap off.  
                    reader = Reader(datapath,run)
                    tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=_waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
                    _time_delays = numpy.append(numpy.array(pulser_time_delay_dict['run%i'%run])[:,1],numpy.array(pulser_time_delay_dict['run%i'%run])[:,1]) #Just duplicating, I am only plotting the ones for this polarization anyways.
                    if mode == 'hpol':
                        _eventid = numpy.random.choice(known_pulser_ids['run%i'%run]['hpol'])
                        _fig, _ax = tdc.plotEvent(_eventid, channels=[0,2,4,6], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='Sample Event shifted by input time delays', time_delays=_time_delays)
                    else:
                        _eventid = numpy.random.choice(known_pulser_ids['run%i'%run]['vpol'])
                        _fig, _ax = tdc.plotEvent(_eventid, channels=[1,3,5,7], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='Sample Event shifted by input time delays', time_delays=_time_delays)
        if True:
            #Note the above ROI assumes you have already cut out events that are below a certain correlation with a template.
            # ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
            # ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
            # #Done for OR condition
            # _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))
            # roi_eventids = numpy.intersect1d(ds.getCutsFromROI(roi_key),_eventids)
            antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU()

            if True:
                if mode == 'hpol':
                    antennas_phase_start = antennas_phase_hpol
                else:
                    antennas_phase_start = antennas_phase_vpol
            else:
                print('WARNING, USING PHYSICAL LOCATIONS TO START')
                antennas_phase_start = antennas_physical            

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

            #DETERMINE THE TIME DELAYS TO BE USED IN THE ACTUAL CALCULATION

            print('Calculating time delays from info.py')
            reader = Reader(datapath,valley_source_run)
            tdc = TimeDelayCalculator(reader, final_corr_length=final_corr_length, crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order,waveform_index_range=waveform_index_range,plot_filters=False,apply_phase_response=apply_phase_response)
            if sine_subtract:
                tdc.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
            

            ds = dataSlicerSingleRun(reader, impulsivity_dset_key, time_delays_dset_key, map_direction_dset_key,\
                        curve_choice=0, trigger_types=[2],included_antennas=included_antennas_channels,include_test_roi=False,\
                        cr_template_n_bins_h=200,cr_template_n_bins_v=200,\
                        impulsivity_n_bins_h=200,impulsivity_n_bins_v=200,\
                        time_delays_n_bins_h=150,time_delays_n_bins_v=150,min_time_delays_val=-200,max_time_delays_val=200,\
                        std_n_bins_h=200,std_n_bins_v=200,max_std_val=9,\
                        p2p_n_bins_h=128,p2p_n_bins_v=128,max_p2p_val=128,\
                        snr_n_bins_h=200,snr_n_bins_v=200,max_snr_val=35)

            ds.addROI('Simple Template V > 0.7',{'cr_template_search_v':[0.7,1.0]})# Adding 2 ROI in different rows and appending as below allows for "OR" instead of "AND"
            ds.addROI('Simple Template H > 0.7',{'cr_template_search_h':[0.7,1.0]})
            #Done for OR condition
            _eventids = numpy.sort(numpy.unique(numpy.append(ds.getCutsFromROI('Simple Template H > 0.7',load=False,save=False),ds.getCutsFromROI('Simple Template V > 0.7',load=False,save=False))))

            time_delay_dict = {}
            for source_key, cut_dict in data_slicer_cut_dict.items():
                if plot_time_delays_on_maps == False:
                    #Only need some time delays
                    if not(source_key in use_sources):
                        continue #Skipping calculating that one.
                else:
                    if numpy.logical_and(not(source_key in use_sources),not(source_key in only_plot)):
                        continue #Skipping calculating that one.

                ds.addROI(source_key,cut_dict)
                roi_eventids = numpy.intersect1d(ds.getCutsFromROI(source_key),_eventids)
                roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
                roi_impulsivity_sort = numpy.argsort(roi_impulsivity)[::-1] #Reverse sorting so high numbers are first.
                
                if len(roi_eventids) > limit_events:
                    print('LIMITING TIME DELAY CALCULATION TO %i MOST IMPULSIVE EVENTS'%limit_events)
                    roi_eventids = numpy.sort(roi_eventids[roi_impulsivity_sort[0:limit_events]])
            
                print('Calculating time delays for %s'%source_key)
                time_shifts, corrs, pairs = tdc.calculateMultipleTimeDelays(roi_eventids,align_method=0,hilbert=hilbert,plot=numpy.logical_and(source_key in use_sources,plot_time_delay_calculations), sine_subtract=sine_subtract)

                if mode == 'hpol':
                    # for event_index, eventid in enumerate(roi_eventids):
                    #     print(eventid)
                    #     print(time_shifts[0:6,event_index])
                    time_delay_dict[source_key] = numpy.mean(time_shifts[0:6,:],axis=1) #perhaps I should be fitting.  Assuming only using sources with consistent time delays
                else:
                    time_delay_dict[source_key] = numpy.mean(time_shifts[6:12,:],axis=1)
                
                all_tds = numpy.mean(time_shifts,axis=1)

                if plot_predicted_time_shifts:
                        _eventid = numpy.random.choice(roi_eventids)
                        if mode == 'hpol':
                            _fig, _ax = tdc.plotEvent(_eventid, channels=[0,2,4,6], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='%s Sample Event shifted by input time delays'%source_key, time_delays=all_tds)
                        else:
                            _fig, _ax = tdc.plotEvent(_eventid, channels=[1,3,5,7], apply_filter=True, hilbert=hilbert, sine_subtract=True, apply_tukey=None, additional_title_text='%s Sample Event shifted by input time delays'%source_key, time_delays=all_tds)
                print('time_delay_dict[%s] = '%source_key)
                print(time_delay_dict[source_key])
                
            for key in list(sources_ENU.keys()):
                if key in use_sources:
                    print('Using key %s'%key)
                    continue
                else:
                    del sources_ENU[key]
                    del data_slicer_cut_dict[key]

            if cable_delay_guess_range is not None:

                limit_cable_delay0 = (cable_delays[0] - cable_delay_guess_range , cable_delays[0] + cable_delay_guess_range)
                limit_cable_delay1 = (cable_delays[1] - cable_delay_guess_range , cable_delays[1] + cable_delay_guess_range)
                limit_cable_delay2 = (cable_delays[2] - cable_delay_guess_range , cable_delays[2] + cable_delay_guess_range)
                limit_cable_delay3 = (cable_delays[3] - cable_delay_guess_range , cable_delays[3] + cable_delay_guess_range)
            else:
                limit_cable_delay0 = None#(cable_delays[0] , None)
                limit_cable_delay1 = None#(cable_delays[1] , None)
                limit_cable_delay2 = None#(cable_delays[2] , None)
                limit_cable_delay3 = None#(cable_delays[3] , None)

            if antenna_position_guess_range_x is not None:
                ant0_physical_limits_x = (antennas_phase_start[0][0] + manual_offset_ant0_x - antenna_position_guess_range_x ,antennas_phase_start[0][0] + manual_offset_ant0_x + antenna_position_guess_range_x)
                ant1_physical_limits_x = (antennas_phase_start[1][0] - antenna_position_guess_range_x ,antennas_phase_start[1][0] + antenna_position_guess_range_x)
                ant2_physical_limits_x = (antennas_phase_start[2][0] - antenna_position_guess_range_x ,antennas_phase_start[2][0] + antenna_position_guess_range_x)
                ant3_physical_limits_x = (antennas_phase_start[3][0] - antenna_position_guess_range_x ,antennas_phase_start[3][0] + antenna_position_guess_range_x)
            else:
                ant0_physical_limits_x = None#None 
                ant1_physical_limits_x = None#(None,0.0) #Forced west of 0
                ant2_physical_limits_x = None#None
                ant3_physical_limits_x = None#(None,0.0) #Forced West of 0

            if antenna_position_guess_range_y is not None:
                ant0_physical_limits_y = (antennas_phase_start[0][1] + manual_offset_ant0_y - antenna_position_guess_range_y ,antennas_phase_start[0][1] + manual_offset_ant0_y + antenna_position_guess_range_y)
                ant1_physical_limits_y = (antennas_phase_start[1][1] - antenna_position_guess_range_y ,antennas_phase_start[1][1] + antenna_position_guess_range_y)
                ant2_physical_limits_y = (antennas_phase_start[2][1] - antenna_position_guess_range_y ,antennas_phase_start[2][1] + antenna_position_guess_range_y)
                ant3_physical_limits_y = (antennas_phase_start[3][1] - antenna_position_guess_range_y ,antennas_phase_start[3][1] + antenna_position_guess_range_y)
            else:
                ant0_physical_limits_y = None#None 
                ant1_physical_limits_y = None#(None,0.0) #Forced west of 0
                ant2_physical_limits_y = None#None
                ant3_physical_limits_y = None#(None,0.0) #Forced West of 0

            if antenna_position_guess_range_z is not None:
                ant0_physical_limits_z = (antennas_phase_start[0][2] + manual_offset_ant0_z - antenna_position_guess_range_z ,antennas_phase_start[0][2] + manual_offset_ant0_z + antenna_position_guess_range_z)
                ant1_physical_limits_z = (antennas_phase_start[1][2] - antenna_position_guess_range_z ,antennas_phase_start[1][2] + antenna_position_guess_range_z)
                ant2_physical_limits_z = (antennas_phase_start[2][2] - antenna_position_guess_range_z ,antennas_phase_start[2][2] + antenna_position_guess_range_z)
                ant3_physical_limits_z = (antennas_phase_start[3][2] - antenna_position_guess_range_z ,antennas_phase_start[3][2] + antenna_position_guess_range_z)
            else:
                ant0_physical_limits_z = None#None 
                ant1_physical_limits_z = None#(None,0.0) #Forced west of 0
                ant2_physical_limits_z = None#None  $(antennas_phase_start[2][2] - 2.5 ,antennas_phase_start[2][2] + 2.5)#None#None
                ant3_physical_limits_z = None#(None,0.0) #Forced West of 0

            if random_offset_amount > 0:
                print('RANDOMLY SHIFTING INPUT POSITIONS BY SPECIFIED AMOUNT:%0.2f m'%random_offset_amount)

            initial_ant0_x = manual_offset_ant0_x + antennas_phase_start[0][0] + float(not fix_ant0_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant0_y = manual_offset_ant0_y + antennas_phase_start[0][1] + float(not fix_ant0_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant0_z = manual_offset_ant0_z + antennas_phase_start[0][2] + float(not fix_ant0_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant1_x = antennas_phase_start[1][0] + float(not fix_ant1_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant1_y = antennas_phase_start[1][1] + float(not fix_ant1_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant1_z = antennas_phase_start[1][2] + float(not fix_ant1_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant2_x = antennas_phase_start[2][0] + float(not fix_ant2_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant2_y = antennas_phase_start[2][1] + float(not fix_ant2_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant2_z = antennas_phase_start[2][2] + float(not fix_ant2_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant3_x = antennas_phase_start[3][0] + float(not fix_ant3_x)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant3_y = antennas_phase_start[3][1] + float(not fix_ant3_y)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]
            initial_ant3_z = antennas_phase_start[3][2] + float(not fix_ant3_z)*numpy.random.uniform(low=-random_offset_amount, high=random_offset_amount,size=1)[0]

            initial_ant0_ENU = numpy.array([initial_ant0_x, initial_ant0_y, initial_ant0_z])
            initial_ant1_ENU = numpy.array([initial_ant1_x, initial_ant1_y, initial_ant1_z])
            initial_ant2_ENU = numpy.array([initial_ant2_x, initial_ant2_y, initial_ant2_z])
            initial_ant3_ENU = numpy.array([initial_ant3_x, initial_ant3_y, initial_ant3_z])

            ##########
            # Define Chi^2
            ##########

            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Initial Positions')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            chi2_ax.scatter(initial_ant0_x, initial_ant0_y, initial_ant0_z,c='r',alpha=0.5,label='Initial Ant0')
            chi2_ax.scatter(initial_ant1_x, initial_ant1_y, initial_ant1_z,c='g',alpha=0.5,label='Initial Ant1')
            chi2_ax.scatter(initial_ant2_x, initial_ant2_y, initial_ant2_z,c='b',alpha=0.5,label='Initial Ant2')
            chi2_ax.scatter(initial_ant3_x, initial_ant3_y, initial_ant3_z,c='m',alpha=0.5,label='Initial Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            
            if False:
                for key, enu in sources_ENU.items():
                    chi2_ax.scatter(enu[0], enu[1], enu[2],alpha=0.5,label=key)
            else:
                chi2_ax.dist = 10
            plt.legend()


            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Both')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            chi2_ax.scatter(initial_ant0_x, initial_ant0_y, initial_ant0_z,c='r',alpha=0.5,label='Initial Ant0')
            chi2_ax.scatter(initial_ant1_x, initial_ant1_y, initial_ant1_z,c='g',alpha=0.5,label='Initial Ant1')
            chi2_ax.scatter(initial_ant2_x, initial_ant2_y, initial_ant2_z,c='b',alpha=0.5,label='Initial Ant2')
            chi2_ax.scatter(initial_ant3_x, initial_ant3_y, initial_ant3_z,c='m',alpha=0.5,label='Initial Ant3')


            if include_baseline_measurements:
                #This will weight against differences that result in longer baselines than measured.   I.e. smaller number if current baseline > measured.  Large for current < measured. 
                #w = lambda measured, current : numpy.exp(measured - current)**2
                measured_baselines = {'01':129*0.3048,
                                      '02':163*0.3048,
                                      '03':181*0.3048,
                                      '12':151*0.3048,
                                      '13':102*0.3048,
                                      '23':85 *0.3048}

            #This math is to set the pairs to include in the calculation.  Typically it will be all of them, but if the option is enabled to remove some
            #from the calculation then this will allow for that to be done.
            pairs = numpy.array(list(itertools.combinations((0,1,2,3), 2)))
            pairs_cut = []
            for pair_index, pair in enumerate(numpy.array(list(itertools.combinations((0,1,2,3), 2)))):
                pairs_cut.append(numpy.logical_and(numpy.all(numpy.isin(numpy.array(pair),included_antennas_lumped)), pair_index in include_baselines)) #include_baselines Overwritten when antennas removed.

            include_baselines = numpy.where(pairs_cut)[0] #Effectively the same as the pairs_cut but index based for baselines.
            print('Including baseline pairs:')
            print(pairs[pairs_cut])

            if True:
                #Initial baselines just to be printed out
                initial_baselines = {   '01':numpy.sqrt((initial_ant0_x - initial_ant1_x)**2 + (initial_ant0_y - initial_ant1_y)**2 + (initial_ant0_z - initial_ant1_z)**2),\
                                        '02':numpy.sqrt((initial_ant0_x - initial_ant2_x)**2 + (initial_ant0_y - initial_ant2_y)**2 + (initial_ant0_z - initial_ant2_z)**2),\
                                        '03':numpy.sqrt((initial_ant0_x - initial_ant3_x)**2 + (initial_ant0_y - initial_ant3_y)**2 + (initial_ant0_z - initial_ant3_z)**2),\
                                        '12':numpy.sqrt((initial_ant1_x - initial_ant2_x)**2 + (initial_ant1_y - initial_ant2_y)**2 + (initial_ant1_z - initial_ant2_z)**2),\
                                        '13':numpy.sqrt((initial_ant1_x - initial_ant3_x)**2 + (initial_ant1_y - initial_ant3_y)**2 + (initial_ant1_z - initial_ant3_z)**2),\
                                        '23':numpy.sqrt((initial_ant2_x - initial_ant3_x)**2 + (initial_ant2_y - initial_ant3_y)**2 + (initial_ant2_z - initial_ant3_z)**2)}
                print('The initial baselines (specified by deploy_index = %i) with random offsets and manual adjustsments in meters are:'%(info.returnDefaultDeploy()))
                print(initial_baselines)

                for key in included_pulsers:
                    d0 = numpy.sqrt((pulser_locations_ENU[key][0] - initial_ant0_x)**2 + (pulser_locations_ENU[key][1] - initial_ant0_y)**2 + (pulser_locations_ENU[key][2] - initial_ant0_z)**2 ) #m
                    print('Pulser %s is %0.2f m away from Antenna 0'%(key, d0))

            if unknown_source_dir_valley or include_sanity:
                # Need correlator map class to use in chi^2.
                chi_2_reader = Reader(datapath,valley_source_run)
                chi_2_cor = Correlator(chi_2_reader,  upsample=cor_upsample, n_phi=180, n_theta=180, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True)
                chi_2_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                td_dicts = {}
                for source_key in use_sources:
                    td_dicts[source_key] = {mode:{'[0, 1]' :  [time_delay_dict[source_key][0]], '[0, 2]' : [time_delay_dict[source_key][1]], '[0, 3]' : [time_delay_dict[source_key][2]], '[1, 2]' : [time_delay_dict[source_key][3]], '[1, 3]' : [time_delay_dict[source_key][4]], '[2, 3]' : [time_delay_dict[source_key][5]]}}

            n_func = lambda v0, v1, v2 : - numpy.cross( v1 - v0 , v2 - v1 )/numpy.linalg.norm(- numpy.cross( v1 - v0 , v2 - v1 )) #Calculates normal vector of plane defined by 3 vectors.  
            def rawChi2(ant0_x, ant0_y, ant0_z, ant1_x, ant1_y, ant1_z, ant2_x, ant2_y, ant2_z, ant3_x, ant3_y, ant3_z, cable_delay0, cable_delay1, cable_delay2, cable_delay3):
                '''
                This is a chi^2 that loops over locations from potential RFI, calculating expected time delays for those locations.  Then
                it will compares those to the calculated time delays for suspected corresponding events.  
                '''
                try:
                    #Calculate distances (already converted to ns) from pulser to each antenna
                    chi_2 = 0.0
                    _cable_delays = [cable_delay0,cable_delay1,cable_delay2,cable_delay3]

                    if include_sanity or unknown_source_dir_valley:
                        #For each of these I need updated maps
                        ant0_ENU = numpy.array([ant0_x, ant0_y, ant0_z])
                        ant1_ENU = numpy.array([ant1_x, ant1_y, ant1_z])
                        ant2_ENU = numpy.array([ant2_x, ant2_y, ant2_z])
                        ant3_ENU = numpy.array([ant3_x, ant3_y, ant3_z])
                        if mode == 'hpol':
                            chi_2_cor.overwriteCableDelays(cable_delay0, chi_2_cor.cable_delays[1], cable_delay1, chi_2_cor.cable_delays[3], cable_delay2, chi_2_cor.cable_delays[5], cable_delay3, chi_2_cor.cable_delays[7],verbose=False, suppress_time_delay_calculations=True)#WARNING, SUPRESSING CALCULATION OF TIME DELAY TABLE HERE, MAKE SURE IT HAPPENS WHEN ANTENNAS CHANGE
                            chi_2_cor.overwriteAntennaLocations(chi_2_cor.A0_physical,chi_2_cor.A1_physical,chi_2_cor.A2_physical,chi_2_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,chi_2_cor.A0_vpol,chi_2_cor.A1_vpol,chi_2_cor.A2_vpol,chi_2_cor.A3_vpol,verbose=False, suppress_time_delay_calculations=False)
                        else:
                            chi_2_cor.overwriteCableDelays(chi_2_cor.cable_delays[0], cable_delay0, chi_2_cor.cable_delays[2], cable_delay1, chi_2_cor.cable_delays[4], cable_delay2, chi_2_cor.cable_delays[6], cable_delay3,verbose=False, suppress_time_delay_calculations=True)#WARNING, SUPRESSING CALCULATION OF TIME DELAY TABLE HERE, MAKE SURE IT HAPPENS WHEN ANTENNAS CHANGE
                            chi_2_cor.overwriteAntennaLocations(chi_2_cor.A0_physical,chi_2_cor.A1_physical,chi_2_cor.A2_physical,chi_2_cor.A3_physical,chi_2_cor.A0_hpol,chi_2_cor.A1_hpol,chi_2_cor.A2_hpol,chi_2_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False, suppress_time_delay_calculations=False)


                    if unknown_source_dir_valley == False:
                        #Assuming you know the source, calcualting time delays assuming that source direction.
                        for key in use_sources:
                                d0 = (numpy.sqrt((sources_ENU[key][0] - ant0_x)**2 + (sources_ENU[key][1] - ant0_y)**2 + (sources_ENU[key][2] - ant0_z)**2 )/c)*1.0e9 #ns
                                d1 = (numpy.sqrt((sources_ENU[key][0] - ant1_x)**2 + (sources_ENU[key][1] - ant1_y)**2 + (sources_ENU[key][2] - ant1_z)**2 )/c)*1.0e9 #ns
                                d2 = (numpy.sqrt((sources_ENU[key][0] - ant2_x)**2 + (sources_ENU[key][1] - ant2_y)**2 + (sources_ENU[key][2] - ant2_z)**2 )/c)*1.0e9 #ns
                                d3 = (numpy.sqrt((sources_ENU[key][0] - ant3_x)**2 + (sources_ENU[key][1] - ant3_y)**2 + (sources_ENU[key][2] - ant3_z)**2 )/c)*1.0e9 #ns

                                d = [d0,d1,d2,d3]

                                for pair_index, pair in enumerate(pairs):
                                    if pairs_cut[pair_index]:
                                        geometric_time_delay = (d[pair[0]] + _cable_delays[pair[0]]) - (d[pair[1]] + _cable_delays[pair[1]])

                                        vals = ((geometric_time_delay - time_delay_dict[key][pair_index])**2)/time_delay_measurement_uncertainty_ns**2 #Assumes time delays are accurate
                                        chi_2 += numpy.sum(vals)
                    else:
                        #Assuming you DON'T know the source, and using the precalculated time delays with the existing geometry to determine if the array points ANYWHERE for that source.
                        for key in use_sources:
                            td_dict = td_dicts[key]

                            mesh_azimuth_deg, mesh_elevation_deg, overlap_map = chi_2_cor.generateTimeDelayOverlapMap(mode, td_dict, overlap_window_ns, value_mode='distance', plot_map=False, mollweide=False,center_dir='E',window_title=None, include_baselines=include_baselines)
                            linear_max_index, theta_best, phi_best, t_best_0subtract1, t_best_0subtract2, t_best_0subtract3, t_best_1subtract2, t_best_1subtract3, t_best_2subtract3 = chi_2_cor.mapMax(overlap_map, max_method=0, verbose=False, zenith_cut_ENU=[90,115], zenith_cut_array_plane=[0,88], pol=mode) #This is used so that max value must be in reasonable window.

                            max_value = overlap_map.flatten()[linear_max_index]
                            
                            chi_2 += ((max_value - overlap_goal)**2)/(overlap_error**2) 

                    if include_pulsers:
                        for key in included_pulsers:
                            d0 = (numpy.sqrt((pulser_locations_ENU[key][0] - ant0_x)**2 + (pulser_locations_ENU[key][1] - ant0_y)**2 + (pulser_locations_ENU[key][2] - ant0_z)**2 )/c)*1.0e9 #ns
                            d1 = (numpy.sqrt((pulser_locations_ENU[key][0] - ant1_x)**2 + (pulser_locations_ENU[key][1] - ant1_y)**2 + (pulser_locations_ENU[key][2] - ant1_z)**2 )/c)*1.0e9 #ns
                            d2 = (numpy.sqrt((pulser_locations_ENU[key][0] - ant2_x)**2 + (pulser_locations_ENU[key][1] - ant2_y)**2 + (pulser_locations_ENU[key][2] - ant2_z)**2 )/c)*1.0e9 #ns
                            d3 = (numpy.sqrt((pulser_locations_ENU[key][0] - ant3_x)**2 + (pulser_locations_ENU[key][1] - ant3_y)**2 + (pulser_locations_ENU[key][2] - ant3_z)**2 )/c)*1.0e9 #ns

                            d = [d0,d1,d2,d3]

                            for pair_index, pair in enumerate(pairs):
                                if pairs_cut[pair_index]:
                                    # pulser_time_delay_dict
                                    # pulser_time_delay_error_dict
                                    if pulser_time_delay_dict[key][pair_index][0][0] == pair[0] and pulser_time_delay_dict[key][pair_index][0][1] == pair[1]:
                                        geometric_time_delay = (d[pair[0]] + _cable_delays[pair[0]]) - (d[pair[1]] + _cable_delays[pair[1]])
                                        vals = ((geometric_time_delay - pulser_time_delay_dict[key][pair_index][1])**2) #Assumes time delays are accurate
                                        chi_2 += pulser_weight*numpy.sum(vals)
                                    else:
                                        print('PAIR INDICES DONT MATCH, SOMETHING IS WRONG')

                    if include_baseline_measurements:
                        #This will weight against differences that result in longer baselines than measured.   I.e. smaller number if current baseline > measured.  Large for current < measured. 
                        current_baselines = {   '01':numpy.sqrt((ant0_x - ant1_x)**2 + (ant0_y - ant1_y)**2 + (ant0_z - ant1_z)**2),\
                                                '02':numpy.sqrt((ant0_x - ant2_x)**2 + (ant0_y - ant2_y)**2 + (ant0_z - ant2_z)**2),\
                                                '03':numpy.sqrt((ant0_x - ant3_x)**2 + (ant0_y - ant3_y)**2 + (ant0_z - ant3_z)**2),\
                                                '12':numpy.sqrt((ant1_x - ant2_x)**2 + (ant1_y - ant2_y)**2 + (ant1_z - ant2_z)**2),\
                                                '13':numpy.sqrt((ant1_x - ant3_x)**2 + (ant1_y - ant3_y)**2 + (ant1_z - ant3_z)**2),\
                                                '23':numpy.sqrt((ant2_x - ant3_x)**2 + (ant2_y - ant3_y)**2 + (ant2_z - ant3_z)**2)}
                        
                        #Adds chi^2 as soon as any baseline exceeds that measured with a tape measure.
                        #chi_2 += 100*numpy.any([(current_baselines['01'] - measured_baselines['01']) > baseline_upper_bound_tolerance, (current_baselines['02'] - measured_baselines['02']) > baseline_upper_bound_tolerance, (current_baselines['03'] - measured_baselines['03']) > baseline_upper_bound_tolerance, (current_baselines['12'] - measured_baselines['12']) > baseline_upper_bound_tolerance, (current_baselines['13'] - measured_baselines['13']) > baseline_upper_bound_tolerance, (current_baselines['23'] - measured_baselines['23']) > baseline_upper_bound_tolerance])
                        for pair_index, pair in enumerate(pairs):
                            if pairs_cut[pair_index]:
                                key = str(int(pair[0])) + str(int(pair[1]))
                                chi_2 += ((current_baselines[key] - measured_baselines[key])**2)/(baseline_measurement_uncertainty_m**2)

                    if limit_array_plane_azimuth_range == True:
                        plane_normal_vector = n_func(numpy.array([ant0_x, ant0_y, ant0_z]),     numpy.array([ant2_x, ant2_y, ant2_z]),     (numpy.array([ant0_x, ant0_y, ant0_z])     + numpy.array([ant3_x, ant3_y, ant3_z]))/2.0)
                        array_plane_az = numpy.rad2deg(numpy.arctan2(plane_normal_vector[1],plane_normal_vector[0]))
                        chi_2 += numpy.max(numpy.abs(array_plane_az) - allowed_array_plane_azimuth_range ,0) #Within range 0 added, outside is adds the difference in degrees outside of the range.  
                    if include_sanity:
                        for pair_index, pair in enumerate(pairs):
                            if pairs_cut[pair_index]:
                                for key in use_sources:
                                    if mode == 'hpol':
                                        if pair_index == 0: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_0subtract1)) - numpy.abs(time_delay_dict[key][0])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 1: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_0subtract2)) - numpy.abs(time_delay_dict[key][1])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 2: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_0subtract3)) - numpy.abs(time_delay_dict[key][2])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 3: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_1subtract2)) - numpy.abs(time_delay_dict[key][3])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 4: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_1subtract3)) - numpy.abs(time_delay_dict[key][4])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 5: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_2subtract3)) - numpy.abs(time_delay_dict[key][5])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        
                                        if include_pulsers:
                                            for key in included_pulsers:
                                                if pair_index == 0: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_0subtract1)) - numpy.abs(pulser_time_delay_dict[key][0][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 1: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_0subtract2)) - numpy.abs(pulser_time_delay_dict[key][1][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 2: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_0subtract3)) - numpy.abs(pulser_time_delay_dict[key][2][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 3: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_1subtract2)) - numpy.abs(pulser_time_delay_dict[key][3][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 4: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_1subtract3)) - numpy.abs(pulser_time_delay_dict[key][4][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 5: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_hpol_2subtract3)) - numpy.abs(pulser_time_delay_dict[key][5][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                    else:
                                        if pair_index == 0: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_0subtract1)) - numpy.abs(time_delay_dict[key][0])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 1: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_0subtract2)) - numpy.abs(time_delay_dict[key][1])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 2: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_0subtract3)) - numpy.abs(time_delay_dict[key][2])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 3: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_1subtract2)) - numpy.abs(time_delay_dict[key][3])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 4: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_1subtract3)) - numpy.abs(time_delay_dict[key][4])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        if pair_index == 5: 
                                            #This can indicate that the baseline is not possible to create. 
                                            difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_2subtract3)) - numpy.abs(time_delay_dict[key][5])
                                            if difference < 0:
                                                chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                        
                                        if include_pulsers:
                                            for key in included_pulsers:
                                                if pair_index == 0: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_0subtract1)) - numpy.abs(pulser_time_delay_dict[key][0][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 1: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_0subtract2)) - numpy.abs(pulser_time_delay_dict[key][1][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 2: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_0subtract3)) - numpy.abs(pulser_time_delay_dict[key][2][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 3: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_1subtract2)) - numpy.abs(pulser_time_delay_dict[key][3][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 4: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_1subtract3)) - numpy.abs(pulser_time_delay_dict[key][4][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.
                                                if pair_index == 5: 
                                                    #This can indicate that the baseline is not possible to create. 
                                                    difference = numpy.max(numpy.abs(chi_2_cor.t_vpol_2subtract3)) - numpy.abs(pulser_time_delay_dict[key][5][1])
                                                    if difference < 0:
                                                        chi_2 -= 100*difference #difference is negative, so subtracting negative is adding to chi^2 weighting against solution that results in maximum achievable baselines being too large.


                        #chi_2 += max(ant2_z - ant1_z,0) #If antenna 2 is lower than antenna 1 then it will add to chi^2 linearly as that happens.  
                    print(chi_2)
                    return chi_2
                except Exception as e:
                    print('Error in rawChi2')
                    print(e)
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(exc_type, fname, exc_tb.tb_lineno)


            
            #-12 ft on pulser locations relative to antennas to account for additional mast elevation.
            
            
            #rawChi2(ant0_x=antennas_phase_start[0][0], ant0_y=antennas_phase_start[0][1], ant0_z=antennas_phase_start[0][2], ant1_x=antennas_phase_start[1][0], ant1_y=antennas_phase_start[1][1], ant1_z=antennas_phase_start[1][2], ant2_x=antennas_phase_start[2][0], ant2_y=antennas_phase_start[2][1], ant2_z=antennas_phase_start[2][2], ant3_x=antennas_phase_start[3][0], ant3_y=antennas_phase_start[3][1], ant3_z=antennas_phase_start[3][2])

            m = Minuit(     rawChi2,\
                            ant0_x=initial_ant0_x,\
                            ant0_y=initial_ant0_y,\
                            ant0_z=initial_ant0_z,\
                            ant1_x=initial_ant1_x,\
                            ant1_y=initial_ant1_y,\
                            ant1_z=initial_ant1_z,\
                            ant2_x=initial_ant2_x,\
                            ant2_y=initial_ant2_y,\
                            ant2_z=initial_ant2_z,\
                            ant3_x=initial_ant3_x,\
                            ant3_y=initial_ant3_y,\
                            ant3_z=initial_ant3_z,\
                            cable_delay0=cable_delays[0],\
                            cable_delay1=cable_delays[1],\
                            cable_delay2=cable_delays[2],\
                            cable_delay3=cable_delays[3],\
                            error_ant0_x=initial_step_x,\
                            error_ant0_y=initial_step_y,\
                            error_ant0_z=initial_step_z,\
                            error_ant1_x=initial_step_x,\
                            error_ant1_y=initial_step_y,\
                            error_ant1_z=initial_step_z,\
                            error_ant2_x=initial_step_x,\
                            error_ant2_y=initial_step_y,\
                            error_ant2_z=initial_step_z,\
                            error_ant3_x=initial_step_x,\
                            error_ant3_y=initial_step_y,\
                            error_ant3_z=initial_step_z,\
                            error_cable_delay0=initial_step_cable_delay,\
                            error_cable_delay1=initial_step_cable_delay,\
                            error_cable_delay2=initial_step_cable_delay,\
                            error_cable_delay3=initial_step_cable_delay,\
                            errordef = 1.0,\
                            limit_ant0_x=ant0_physical_limits_x,\
                            limit_ant0_y=ant0_physical_limits_y,\
                            limit_ant0_z=ant0_physical_limits_z,\
                            limit_ant1_x=ant1_physical_limits_x,\
                            limit_ant1_y=ant1_physical_limits_y,\
                            limit_ant1_z=ant1_physical_limits_z,\
                            limit_ant2_x=ant2_physical_limits_x,\
                            limit_ant2_y=ant2_physical_limits_y,\
                            limit_ant2_z=ant2_physical_limits_z,\
                            limit_ant3_x=ant3_physical_limits_x,\
                            limit_ant3_y=ant3_physical_limits_y,\
                            limit_ant3_z=ant3_physical_limits_z,\
                            limit_cable_delay0=limit_cable_delay0,\
                            limit_cable_delay1=limit_cable_delay1,\
                            limit_cable_delay2=limit_cable_delay2,\
                            limit_cable_delay3=limit_cable_delay3,\
                            fix_ant0_x=fix_ant0_x,\
                            fix_ant0_y=fix_ant0_y,\
                            fix_ant0_z=fix_ant0_z,\
                            fix_ant1_x=fix_ant1_x,\
                            fix_ant1_y=fix_ant1_y,\
                            fix_ant1_z=fix_ant1_z,\
                            fix_ant2_x=fix_ant2_x,\
                            fix_ant2_y=fix_ant2_y,\
                            fix_ant2_z=fix_ant2_z,\
                            fix_ant3_x=fix_ant3_x,\
                            fix_ant3_y=fix_ant3_y,\
                            fix_ant3_z=fix_ant3_z,\
                            fix_cable_delay0=fix_cable_delay0,\
                            fix_cable_delay1=fix_cable_delay1,\
                            fix_cable_delay2=fix_cable_delay2,\
                            fix_cable_delay3=fix_cable_delay3)


            result = m.migrad(resume=False)

            print(result)
            m.hesse()
            if False:
                m.minos()
                pprint(m.get_fmin())
            else:
                try:
                    m.minos()
                    pprint(m.get_fmin())
                except:
                    print('MINOS FAILED, NOT VALID SOLUTION.')
            print('\a')

            # m.draw_mncontour('ant1_x','ant1_y')

            for antenna in range(4):
                fig = plt.figure()
                fig.canvas.set_window_title('Ant %i chi^2'%antenna)
                for index, key in enumerate(['ant%i_x'%antenna,'ant%i_y'%antenna,'ant%i_z'%antenna]):
                    plt.subplot(1,3,index + 1)
                    m.draw_profile(key)

            if cable_delay_guess_range is not None:
                fig = plt.figure()
                fig.canvas.set_window_title('Cable Delays')
                for antenna in range(4):
                    plt.subplot(2,2,antenna + 1)
                    m.draw_profile('cable_delay%i'%antenna)



            #12 variables
            ant0_phase_x = m.values['ant0_x']
            ant0_phase_y = m.values['ant0_y']
            ant0_phase_z = m.values['ant0_z']

            ant1_phase_x = m.values['ant1_x']
            ant1_phase_y = m.values['ant1_y']
            ant1_phase_z = m.values['ant1_z']

            ant2_phase_x = m.values['ant2_x']
            ant2_phase_y = m.values['ant2_y']
            ant2_phase_z = m.values['ant2_z']

            ant3_phase_x = m.values['ant3_x']
            ant3_phase_y = m.values['ant3_y']
            ant3_phase_z = m.values['ant3_z']

            ant0_ENU = numpy.array([ant0_phase_x, ant0_phase_y, ant0_phase_z])
            ant1_ENU = numpy.array([ant1_phase_x, ant1_phase_y, ant1_phase_z])
            ant2_ENU = numpy.array([ant2_phase_x, ant2_phase_y, ant2_phase_z])
            ant3_ENU = numpy.array([ant3_phase_x, ant3_phase_y, ant3_phase_z])

            chi2_ax.plot([initial_ant0_x , ant0_phase_x], [initial_ant0_y , ant0_phase_y], [initial_ant0_z , ant0_phase_z],c='r',alpha=0.5,linestyle='--')
            chi2_ax.plot([initial_ant1_x , ant1_phase_x], [initial_ant1_y , ant1_phase_y], [initial_ant1_z , ant1_phase_z],c='g',alpha=0.5,linestyle='--')
            chi2_ax.plot([initial_ant2_x , ant2_phase_x], [initial_ant2_y , ant2_phase_y], [initial_ant2_z , ant2_phase_z],c='b',alpha=0.5,linestyle='--')
            chi2_ax.plot([initial_ant3_x , ant3_phase_x], [initial_ant3_y , ant3_phase_y], [initial_ant3_z , ant3_phase_z],c='m',alpha=0.5,linestyle='--')

            chi2_ax.scatter(ant0_phase_x, ant0_phase_y, ant0_phase_z,marker='*',c='r',alpha=0.5,label='Final Ant0')
            chi2_ax.scatter(ant1_phase_x, ant1_phase_y, ant1_phase_z,marker='*',c='g',alpha=0.5,label='Final Ant1')
            chi2_ax.scatter(ant2_phase_x, ant2_phase_y, ant2_phase_z,marker='*',c='b',alpha=0.5,label='Final Ant2')
            chi2_ax.scatter(ant3_phase_x, ant3_phase_y, ant3_phase_z,marker='*',c='m',alpha=0.5,label='Final Ant3')
            
            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            chi2_ax.dist = 10
            plt.legend()



            chi2_fig = plt.figure()
            chi2_fig.canvas.set_window_title('Final Positions')
            chi2_ax = chi2_fig.add_subplot(111, projection='3d')
            chi2_ax.scatter(ant0_phase_x, ant0_phase_y, ant0_phase_z,marker='*',c='r',alpha=0.5,label='Final Ant0')
            chi2_ax.scatter(ant1_phase_x, ant1_phase_y, ant1_phase_z,marker='*',c='g',alpha=0.5,label='Final Ant1')
            chi2_ax.scatter(ant2_phase_x, ant2_phase_y, ant2_phase_z,marker='*',c='b',alpha=0.5,label='Final Ant2')
            chi2_ax.scatter(ant3_phase_x, ant3_phase_y, ant3_phase_z,marker='*',c='m',alpha=0.5,label='Final Ant3')

            chi2_ax.set_xlabel('East (m)',linespacing=10)
            chi2_ax.set_ylabel('North (m)',linespacing=10)
            chi2_ax.set_zlabel('Up (m)',linespacing=10)
            chi2_ax.dist = 10
            plt.legend()


            
            

            sources_ENU, data_slicer_cut_dict = info.loadValleySourcesENU() #Plot all potential sources
            
            for source_key, cut_dict in data_slicer_cut_dict.items():
                if source_key in only_plot:
                    #Accounting for potentially shifted antenna 0.  Centering ENU to that for calculations.
                    original_sources_ENU = numpy.array([sources_ENU[source_key][0] , sources_ENU[source_key][1] , sources_ENU[source_key][2]])
                    original_distance_m = numpy.linalg.norm(original_sources_ENU)
                    original_zenith_deg = numpy.rad2deg(numpy.arccos(original_sources_ENU[2]/original_distance_m))
                    original_elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(original_sources_ENU[2]/original_distance_m))
                    original_azimuth_deg = numpy.rad2deg(numpy.arctan2(original_sources_ENU[1],original_sources_ENU[0]))

                    sources_ENU_new = numpy.array([sources_ENU[source_key][0] - ant0_ENU[0] , sources_ENU[source_key][1] - ant0_ENU[1] , sources_ENU[source_key][2] - ant0_ENU[2]])
                    distance_m = numpy.linalg.norm(sources_ENU_new)
                    zenith_deg = numpy.rad2deg(numpy.arccos(sources_ENU_new[2]/distance_m))
                    elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(sources_ENU_new[2]/distance_m))
                    azimuth_deg = numpy.rad2deg(numpy.arctan2(sources_ENU_new[1],sources_ENU_new[0]))

                    #Only used for map.  Hists use higher resolution on tighter area.
                    map_resolution = 0.5 #degrees
                    range_phi_deg=(-180, 180)
                    range_theta_deg=(0,180)
                    n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                    n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                    
                    cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    #Want cor to be the initial correlator, which requires moving antennas to the positions they actually started at.
                    if mode == 'hpol':
                        cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,cor.A0_vpol,cor.A1_vpol,cor.A2_vpol,cor.A3_vpol,verbose=False)
                    else:
                        cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cor.A0_hpol,cor.A1_hpol,cor.A2_hpol,cor.A3_hpol,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,verbose=False)


                    adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                    adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)
                    
                    if mode == 'hpol':
                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
                        adjusted_cor.overwriteCableDelays(m.values['cable_delay0'], adjusted_cor.cable_delays[1], m.values['cable_delay1'], adjusted_cor.cable_delays[3], m.values['cable_delay2'], adjusted_cor.cable_delays[5], m.values['cable_delay3'], adjusted_cor.cable_delays[7])
                    else:
                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)
                        adjusted_cor.overwriteCableDelays(adjusted_cor.cable_delays[0], m.values['cable_delay0'], adjusted_cor.cable_delays[2], m.values['cable_delay1'], adjusted_cor.cable_delays[4], m.values['cable_delay2'], adjusted_cor.cable_delays[6], m.values['cable_delay3'])
                    
                    if source_key == 'Solar Plant':
                        measured_baselines = {'01':129*0.3048,
                                              '02':163*0.3048,
                                              '03':181*0.3048,
                                              '12':151*0.3048,
                                              '13':102*0.3048,
                                              '23':85 *0.3048}
                        time_delay_dict_solarplant = [-125.95932801, -126.69556678, -180.32099852,   -0.81123762,  -54.37417032,  -53.60793201]
                        print('min = %0.3f, max = %0.3f, expected = %0.3f, physical = %0.3f'%(numpy.min(adjusted_cor.t_hpol_0subtract1),numpy.max(adjusted_cor.t_hpol_0subtract1),time_delay_dict_solarplant[0], 1e9*measured_baselines['01']/c))
                        print('min = %0.3f, max = %0.3f, expected = %0.3f, physical = %0.3f'%(numpy.min(adjusted_cor.t_hpol_0subtract2),numpy.max(adjusted_cor.t_hpol_0subtract2),time_delay_dict_solarplant[1], 1e9*measured_baselines['02']/c))
                        print('min = %0.3f, max = %0.3f, expected = %0.3f, physical = %0.3f'%(numpy.min(adjusted_cor.t_hpol_0subtract3),numpy.max(adjusted_cor.t_hpol_0subtract3),time_delay_dict_solarplant[2], 1e9*measured_baselines['03']/c))
                        print('min = %0.3f, max = %0.3f, expected = %0.3f, physical = %0.3f'%(numpy.min(adjusted_cor.t_hpol_1subtract2),numpy.max(adjusted_cor.t_hpol_1subtract2),time_delay_dict_solarplant[3], 1e9*measured_baselines['12']/c))
                        print('min = %0.3f, max = %0.3f, expected = %0.3f, physical = %0.3f'%(numpy.min(adjusted_cor.t_hpol_1subtract3),numpy.max(adjusted_cor.t_hpol_1subtract3),time_delay_dict_solarplant[4], 1e9*measured_baselines['13']/c))
                        print('min = %0.3f, max = %0.3f, expected = %0.3f, physical = %0.3f'%(numpy.min(adjusted_cor.t_hpol_2subtract3),numpy.max(adjusted_cor.t_hpol_2subtract3),time_delay_dict_solarplant[5], 1e9*measured_baselines['23']/c))

                    if plot_expected_direction == False:
                        zenith_deg = None
                        azimuth_deg = None
                    if plot_time_delays_on_maps:
                        td_dict = {mode:{'[0, 1]' :  [time_delay_dict[source_key][0]], '[0, 2]' : [time_delay_dict[source_key][1]], '[0, 3]' : [time_delay_dict[source_key][2]], '[1, 2]' : [time_delay_dict[source_key][3]], '[1, 3]' : [time_delay_dict[source_key][4]], '[2, 3]' : [time_delay_dict[source_key][5]]}}
                    else:
                        td_dict = {}

                    if plot_overlap:
                        adjusted_cor.generateTimeDelayOverlapMap(mode, td_dict, overlap_window_ns, plot_map=True, mollweide=False,center_dir='E',window_title='Adjusted Map Overlap\n%s'%source_key, include_baselines=include_baselines)

                    ds.addROI(source_key,cut_dict)
                    roi_eventids = numpy.intersect1d(ds.getCutsFromROI(source_key),_eventids)
                    roi_impulsivity = ds.getDataFromParam(roi_eventids,'impulsivity_h')
                    roi_impulsivity_sort = numpy.argsort(roi_impulsivity) #NOT REVERSED
                    eventid = roi_eventids[roi_impulsivity_sort[-1]]
                    
                    #mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_array_plane=None, interactive=True,circle_zenith=original_zenith_deg, circle_az=original_azimuth_deg, time_delay_dict=td_dict)
                    adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,100], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,window_title=source_key, include_baselines=include_baselines)
                    #adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=True, zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict,window_title=source_key)
                    
                    if plot_histograms:
                        map_resolution = 0.25 #degrees
                        range_phi_deg=(azimuth_deg - 10, azimuth_deg + 10)
                        range_theta_deg=(zenith_deg - 10,zenith_deg + 10)
                        n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                        n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                        
                        cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                        cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        if mode == 'hpol':
                            cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,cor.A0_vpol,cor.A1_vpol,cor.A2_vpol,cor.A3_vpol,verbose=False)
                        else:
                            cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cor.A0_hpol,cor.A1_hpol,cor.A2_hpol,cor.A3_hpol,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,verbose=False)

                        adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                        adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        if mode == 'hpol':
                            adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
                            adjusted_cor.overwriteCableDelays(m.values['cable_delay0'], adjusted_cor.cable_delays[1], m.values['cable_delay1'], adjusted_cor.cable_delays[3], m.values['cable_delay2'], adjusted_cor.cable_delays[5], m.values['cable_delay3'], adjusted_cor.cable_delays[7])
                        else:
                            adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)
                            adjusted_cor.overwriteCableDelays(adjusted_cor.cable_delays[0], m.values['cable_delay0'], adjusted_cor.cable_delays[2], m.values['cable_delay1'], adjusted_cor.cable_delays[4], m.values['cable_delay2'], adjusted_cor.cable_delays[6], m.values['cable_delay3'])


                        hist = adjusted_cor.histMapPeak(numpy.sort(numpy.random.choice(roi_eventids,min(limit_events,len(roi_eventids)))), mode, plot_map=True, hilbert=False, max_method=0, use_weight=False, mollweide=False, center_dir='E', zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,100],circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='Hist ' + source_key, include_baselines=include_baselines,iterate_sub_baselines=iterate_sub_baselines)

            if include_pulsers:
                
                ant0_ENU = numpy.array([ant0_phase_x, ant0_phase_y, ant0_phase_z])
                ant1_ENU = numpy.array([ant1_phase_x, ant1_phase_y, ant1_phase_z])
                ant2_ENU = numpy.array([ant2_phase_x, ant2_phase_y, ant2_phase_z])
                ant3_ENU = numpy.array([ant3_phase_x, ant3_phase_y, ant3_phase_z])
                

                for key in included_pulsers:
                    #Calculate old and new geometries
                    #Distance needed when calling correlator, as it uses that distance.
                    original_pulser_ENU = numpy.array([pulser_locations_ENU[key][0] , pulser_locations_ENU[key][1] , pulser_locations_ENU[key][2]])
                    original_distance_m = numpy.linalg.norm(original_pulser_ENU)
                    original_zenith_deg = numpy.rad2deg(numpy.arccos(original_pulser_ENU[2]/original_distance_m))
                    original_elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(original_pulser_ENU[2]/original_distance_m))
                    original_azimuth_deg = numpy.rad2deg(numpy.arctan2(original_pulser_ENU[1],original_pulser_ENU[0]))

                    pulser_ENU_new = numpy.array([pulser_locations_ENU[key][0] - ant0_ENU[0] , pulser_locations_ENU[key][1] - ant0_ENU[1] , pulser_locations_ENU[key][2] - ant0_ENU[2]])
                    distance_m = numpy.linalg.norm(pulser_ENU_new)
                    zenith_deg = numpy.rad2deg(numpy.arccos(pulser_ENU_new[2]/distance_m))
                    elevation_deg = 90.0 - numpy.rad2deg(numpy.arccos(pulser_ENU_new[2]/distance_m))
                    azimuth_deg = numpy.rad2deg(numpy.arctan2(pulser_ENU_new[1],pulser_ENU_new[0]))

                    map_resolution = 0.25 #degrees
                    range_phi_deg = (-90, 90)
                    range_theta_deg = (0,180)
                    n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                    n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                    
                    #Load in correct reader for this pulser
                    reader = Reader(datapath,int(key.replace('run','')))
                    # cor_upsample =  len(reader.t())
                    if key == 'run1511':
                        cor = Correlator(reader,  upsample=cor_upsample, n_phi=100,range_phi_deg=[-180,180], n_theta=50,range_theta_deg=[0,180], waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                        cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        if mode == 'hpol':
                            cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,cor.A0_vpol,cor.A1_vpol,cor.A2_vpol,cor.A3_vpol,verbose=False)
                        else:
                            cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cor.A0_hpol,cor.A1_hpol,cor.A2_hpol,cor.A3_hpol,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,verbose=False)
                        cor.generateTimeIndices(debug=True)
                    
                    cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                    cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    if mode == 'hpol':
                        cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,cor.A0_vpol,cor.A1_vpol,cor.A2_vpol,cor.A3_vpol,verbose=False)
                    else:
                        cor.overwriteAntennaLocations(cor.A0_physical,cor.A1_physical,cor.A2_physical,cor.A3_physical,cor.A0_hpol,cor.A1_hpol,cor.A2_hpol,cor.A3_hpol,initial_ant0_ENU,initial_ant1_ENU,initial_ant2_ENU,initial_ant3_ENU,verbose=False)


                    adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                    adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                    if mode == 'hpol':
                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
                        adjusted_cor.overwriteCableDelays(m.values['cable_delay0'], adjusted_cor.cable_delays[1], m.values['cable_delay1'], adjusted_cor.cable_delays[3], m.values['cable_delay2'], adjusted_cor.cable_delays[5], m.values['cable_delay3'], adjusted_cor.cable_delays[7])
                    else:
                        adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)
                        adjusted_cor.overwriteCableDelays(adjusted_cor.cable_delays[0], m.values['cable_delay0'], adjusted_cor.cable_delays[2], m.values['cable_delay1'], adjusted_cor.cable_delays[4], m.values['cable_delay2'], adjusted_cor.cable_delays[6], m.values['cable_delay3'])                    

                    if plot_expected_direction == False:
                        zenith_deg = None
                        azimuth_deg = None
                    if plot_time_delays_on_maps:
                        # if key == 'run1509':
                        #     td_dict = {mode:{'[0, 1]' :  [pulser_time_delay_dict[key][0][1]], '[0, 2]' : [pulser_time_delay_dict[key][1][1]], '[0, 3]' : [pulser_time_delay_dict[key][2][1]], '[1, 2]' : [pulser_time_delay_dict[key][3][1]], '[1, 3]' : [pulser_time_delay_dict[key][4][1]], '[2, 3]' : [-99.04]}}
                        #     print(td_dict)
                        # else:
                        td_dict = {mode:{'[0, 1]' :  [pulser_time_delay_dict[key][0][1]], '[0, 2]' : [pulser_time_delay_dict[key][1][1]], '[0, 3]' : [pulser_time_delay_dict[key][2][1]], '[1, 2]' : [pulser_time_delay_dict[key][3][1]], '[1, 3]' : [pulser_time_delay_dict[key][4][1]], '[2, 3]' : [pulser_time_delay_dict[key][5][1]]}}
                    else:
                        td_dict = {}

                    eventid = numpy.random.choice(known_pulser_ids[key][mode])
                    
                    #mean_corr_values, fig, ax = cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_array_plane=None, interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict)
                    adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(eventid, mode, include_baselines=include_baselines, plot_map=True, plot_corr=False, hilbert=False, zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict)
                    #adjusted_mean_corr_values, adjusted_fig, adjusted_ax = adjusted_cor.map(eventid, mode, plot_map=True, plot_corr=False, hilbert=True, zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90], interactive=True,circle_zenith=zenith_deg, circle_az=azimuth_deg, time_delay_dict=td_dict)
                    if plot_histograms:
                        map_resolution = 0.1 #degrees
                        range_phi_deg=(azimuth_deg - 10, azimuth_deg + 10)
                        range_theta_deg=(zenith_deg - 10,zenith_deg + 10)
                        n_phi = numpy.ceil((max(range_phi_deg) - min(range_phi_deg))/map_resolution).astype(int)
                        n_theta = numpy.ceil((max(range_theta_deg) - min(range_theta_deg))/map_resolution).astype(int)
                        
                        #Load in correct reader for this pulser
                        reader = Reader(datapath,int(key.replace('run','')))
                        # cor_upsample =  len(reader.t())
                        
                        cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=original_distance_m)
                        cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        adjusted_cor = Correlator(reader,  upsample=cor_upsample, n_phi=n_phi,range_phi_deg=range_phi_deg, n_theta=n_theta,range_theta_deg=range_theta_deg, waveform_index_range=(None,None),crit_freq_low_pass_MHz=crit_freq_low_pass_MHz, crit_freq_high_pass_MHz=crit_freq_high_pass_MHz, low_pass_filter_order=low_pass_filter_order, high_pass_filter_order=high_pass_filter_order, plot_filter=False,apply_phase_response=apply_phase_response, tukey=False, sine_subtract=True,map_source_distance_m=distance_m)
                        adjusted_cor.prep.addSineSubtract(sine_subtract_min_freq_GHz, sine_subtract_max_freq_GHz, sine_subtract_percent, max_failed_iterations=3, verbose=False, plot=False)

                        if mode == 'hpol':
                            adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,adjusted_cor.A0_vpol,adjusted_cor.A1_vpol,adjusted_cor.A2_vpol,adjusted_cor.A3_vpol,verbose=False)
                            adjusted_cor.overwriteCableDelays(m.values['cable_delay0'], adjusted_cor.cable_delays[1], m.values['cable_delay1'], adjusted_cor.cable_delays[3], m.values['cable_delay2'], adjusted_cor.cable_delays[5], m.values['cable_delay3'], adjusted_cor.cable_delays[7])
                        else:
                            adjusted_cor.overwriteAntennaLocations(adjusted_cor.A0_physical,adjusted_cor.A1_physical,adjusted_cor.A2_physical,adjusted_cor.A3_physical,adjusted_cor.A0_hpol,adjusted_cor.A1_hpol,adjusted_cor.A2_hpol,adjusted_cor.A3_hpol,ant0_ENU,ant1_ENU,ant2_ENU,ant3_ENU,verbose=False)
                            adjusted_cor.overwriteCableDelays(adjusted_cor.cable_delays[0], m.values['cable_delay0'], adjusted_cor.cable_delays[2], m.values['cable_delay1'], adjusted_cor.cable_delays[4], m.values['cable_delay2'], adjusted_cor.cable_delays[6], m.values['cable_delay3'])                    

                        hist = adjusted_cor.histMapPeak(numpy.sort(numpy.random.choice(known_pulser_ids[key][mode],min(limit_events,len(known_pulser_ids[key][mode])))), mode, plot_map=True, hilbert=False, max_method=0, use_weight=False, mollweide=False, center_dir='E', zenith_cut_ENU=[90,180],zenith_cut_array_plane=[0,90],circle_zenith=zenith_deg, circle_az=azimuth_deg, window_title='Hist ' + key, include_baselines=include_baselines,iterate_sub_baselines=iterate_sub_baselines)




            print('\n')
            print(result)
            print('\n')
            print('Copy-Paste Prints:\n------------')
            print('')
            print('antennas_phase_%s = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, m.values['ant0_x'],m.values['ant0_y'],m.values['ant0_z'] ,  m.values['ant1_x'],m.values['ant1_y'],m.values['ant1_z'],  m.values['ant2_x'],m.values['ant2_y'],m.values['ant2_z'],  m.values['ant3_x'],m.values['ant3_y'],m.values['ant3_z']))
            print('antennas_phase_%s_hesse = {0 : [%f, %f, %f], 1 : [%f, %f, %f], 2 : [%f, %f, %f], 3 : [%f, %f, %f]}'%(mode, m.errors['ant0_x'],m.errors['ant0_y'],m.errors['ant0_z'] ,  m.errors['ant1_x'],m.errors['ant1_y'],m.errors['ant1_z'],  m.errors['ant2_x'],m.errors['ant2_y'],m.errors['ant2_z'],  m.errors['ant3_x'],m.errors['ant3_y'],m.errors['ant3_z']))
            print('')
            print('cable_delays_%s = numpy.array([%f,%f,%f,%f])'%(mode,m.values['cable_delay0'],m.values['cable_delay1'],m.values['cable_delay2'],m.values['cable_delay3']))
            print('cable_delays_%s_hesse = numpy.array([%f,%f,%f,%f])'%(mode,m.errors['cable_delay0'],m.errors['cable_delay1'],m.errors['cable_delay2'],m.errors['cable_delay3']))

            print('Code completed.')
            print('\a')


            
    except Exception as e:
        print('Error in main loop.')
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)






