'''
The purpose of this script is to strip meta data from phone photographs, determine the GPS coordinates, and plot 
them/tabulate them.

Following this guide for stripping data https://www.thepythoncode.com/article/extracting-image-metadata-in-python 
'''

import os
import sys
import gc
import pymap3d as pm
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader #Must be imported before matplotlib or else plots don't load.

sys.path.append(os.environ['BEACON_ANALYSIS_DIR'])
import tools.info as info
import matplotlib
import matplotlib.patches as mpatches
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import numpy
from scipy.linalg import lstsq
import math
import datetime as dt
plt.ion()
import inspect
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from PIL import Image
from PIL.ExifTags import TAGS
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation

import json




def jsonReader(filename, starting_utc_timestamp, ending_utc_timestamp,filetype='google'):
    '''
    This is intended to be a quick interface to the location history for the
    parameters I care about.
    
    # Google keys
    # locations: All location records.
    # timestampMs(int64): Timestamp (UTC) in milliseconds for the recorded location.
    # latitudeE7(int32): The latitude value of the location in E7 format (degrees multiplied by 10**7 and rounded to the nearest integer).
    # longitudeE7(int32): The longitude value of the location in E7 format (degrees multiplied by 10**7 and rounded to the nearest integer).
    # accuracy(int32): Approximate location accuracy radius in meters.
    # velocity(int32): Speed in meters per second.
    # heading(int32): Degrees east of true north.
    # altitude(int32): Meters above the WGS84 reference ellipsoid.
    # verticalAccuracy(int32): Vertical accuracy calculated in meters.
    # activity: Information about the activity at the location.
    # timestampMs(int64): Timestamp (UTC) in milliseconds for the recorded activity.
    # type: Description of the activity type.
    # confidence(int32): Confidence associated with the specified activity type
    '''
    if filetype == 'google':
        with open(filename, 'rb') as file:
            data = json.load(file)
            #import pdb; pdb.set_trace()
            output = {}
            output['timestamp_s'] = [] # timestampMs(int64): Timestamp (UTC) in milliseconds for the recorded location, but converted to seconds
            output['lat'] = [] # latitudeE7(int32): The latitude value of the location in E7 format (degrees multiplied by 10**7 and rounded to the nearest integer).
            output['lon'] = [] # longitudeE7(int32): The longitude value of the location in E7 format (degrees multiplied by 10**7 and rounded to the nearest integer).
            output['alt_m_wgs84'] = []  # altitude(int32): Meters above the WGS84 reference ellipsoid.
            output['acc_loc_m'] = [] # accuracy(int32): Approximate location accuracy radius in meters.
            output['acc_alt_m_wgs84'] = [] # verticalAccuracy(int32): Vertical accuracy calculated in meters.

            desired_keys = numpy.array(['timestampMs','latitudeE7','longitudeE7','altitude','accuracy','verticalAccuracy'])

            for location in sorted(data['locations'], key=lambda loc: int(loc['timestampMs'])):
                #Location data sorted by timestamp.
                time_s = int(location['timestampMs'])/1e3
                if time_s >= starting_utc_timestamp and time_s < ending_utc_timestamp:
                    #import pdb; pdb.set_trace()
                    if numpy.all(numpy.isin(desired_keys,[*location.keys()])):
                        #Such that all data is present
                        output['timestamp_s'].append(time_s)
                        output['lat'].append(location['latitudeE7'])
                        output['lon'].append(location['longitudeE7'])
                        output['alt_m_wgs84'].append(location['altitude'])
                        output['acc_loc_m'].append(location['accuracy'])
                        output['acc_alt_m_wgs84'].append(location['verticalAccuracy'])
            for key in list(output.keys()):
                output[key] = numpy.array(output[key])
            file.close()
    elif filetype == 'facebook':
        print('The facebook data is useless for altitude.')
        with open(filename, 'rb') as file:
                data = json.load(file)#['location_history']
                output = data
                print(data)
                file.close()

    return output





class Annotation3D(Annotation):
    '''Annotate the point xyz with text s'''

    def __init__(self, s, xyz, *args, **kwargs):
        Annotation.__init__(self,s, xy=(0,0), *args, **kwargs)
        self._verts3d = xyz        

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.xy=(xs,ys)
        Annotation.draw(self, renderer)

def annotate3D(ax, s, *args, **kwargs):
    '''add anotation text s to to Axes3d ax'''
    tag = Annotation3D(s, *args, **kwargs)
    ax.add_artist(tag)

# I explored the data taken from google etc. on my position but they were not useful. 
# starting_utc_timestamp_s = 1569913200 #Sep 1st
# ending_utc_timestamp_s = 1570518000 #Sep 8th
# location_history_file_google = os.environ['BEACON_ANALYSIS_DIR'] + r'data/google/Takeout/Location History/Location History.json' 
# # location_history_file_facebook = os.environ['BEACON_ANALYSIS_DIR'] + r'data/facebook/location/primary_location.json' 
# # facebook_location_dir = jsonReader(location_history_file_facebook, starting_utc_timestamp_s, ending_utc_timestamp_s, filetype='facebook')

# google_location_dir = jsonReader(location_history_file_google, starting_utc_timestamp_s, ending_utc_timestamp_s, filetype='google')
# plt.figure()
# cut = google_location_dir['acc_loc_m'] < 100
# plt.plot(google_location_dir['lon'][cut],google_location_dir['lat'][cut])

if __name__ == '__main__':
    plt.close('all')

    if True:
        exclude_images = ['IMG_20191004_115505.jpg']

        datapath = os.environ['BEACON_ANALYSIS_DIR'] + 'data/pictures/2019_hillside/' #'data/pictures/2019/'
        files = numpy.array(os.listdir(datapath))
        files = files[~numpy.isin(files,exclude_images)]

        verbose = False

        gps_raw_info = {}
        gps_info = {}
        for image_name in files:
            if image_name.split('.')[-1].lower() != 'jpg':
                continue
            image = Image.open(datapath + image_name)
            exifdata = image.getexif()
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                # decode bytes 
                if isinstance(data, bytes):
                    data = data.decode()
                if tag == 'GPSInfo':
                    gps_raw_info[image_name] = data.copy()
                    try:
                        gps_info[image_name] = {'lat' : [-1,1][data[1] == 'N'] * (data[2][0] + data[2][1]/60.0 + data[2][2]/3600.0),\
                                                'lon' : [-1,1][data[3] == 'E'] * (data[4][0] + data[4][1]/60.0 + data[4][2]/3600.0),\
                                                'alt' : data[6]}
                    except:
                        print('Failed to load data for '+image_name)
                if verbose:
                    print(f"{tag:25}: {data}")



        colors = ['tab:blue','tab:orange','tab:green','tab:red']#['b','g','r','c']
        pulser_colors = ['tab:purple','tab:brown','tab:pink']#['m','y','o']
        deploy_index = 1
        antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=deploy_index)#)
        origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
        pulser_locations = info.loadPulserPhaseLocationsENU(deploy_index=deploy_index)
        plot_phase = False
        plot_distance_threshold_1 = 300 #m
        plot_surface = True
        plot_pictures = False
        plot_best_fit_plane = True

        fig = plt.figure()
        fig.canvas.set_window_title('Antenna Locations')
        if plot_pictures:
            ax = fig.add_subplot(121, projection='3d')
        else:
            ax = fig.add_subplot(111, projection='3d')

        for i, a in antennas_physical.items():
            distance = numpy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
            if distance < plot_distance_threshold_1:
                ax.scatter(a[0], a[1], a[2], marker='o',color=colors[i],label='Physical %i'%i,alpha=0.8)

        if plot_phase == False:
            for i, a in antennas_phase_hpol.items():
                distance = numpy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                if distance < plot_distance_threshold_1:
                    ax.plot([antennas_physical[i][0],antennas_phase_hpol[i][0]],[antennas_physical[i][1],antennas_phase_hpol[i][1]],[antennas_physical[i][2],antennas_phase_hpol[i][2]],color=colors[i],linestyle='--',alpha=0.5)
                    ax.scatter(a[0], a[1], a[2], marker='*',color=colors[i],label='%s Phase Center %i'%('Hpol', i),alpha=0.8)
            for i, a in antennas_phase_vpol.items():
                distance = numpy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                if distance < plot_distance_threshold_1:
                    ax.plot([antennas_physical[i][0],antennas_phase_vpol[i][0]],[antennas_physical[i][1],antennas_phase_vpol[i][1]],[antennas_physical[i][2],antennas_phase_vpol[i][2]],color=colors[i],linestyle='--',alpha=0.5)
                    ax.scatter(a[0], a[1], a[2], marker='^',color=colors[i],label='%s Phase Center %i'%('Vpol', i),alpha=0.8)

        for site, key in enumerate(['run1507','run1509','run1511']):
            site += 1
            distance = numpy.sqrt(pulser_locations['physical'][key][0]**2 + pulser_locations['physical'][key][1]**2 + pulser_locations['physical'][key][2]**2)
            if distance < plot_distance_threshold_1:
                ax.scatter(pulser_locations['physical'][key][0], pulser_locations['physical'][key][1], pulser_locations['physical'][key][2], color=pulser_colors[site-1], marker='o',label='Physical Pulser Site %i'%site,alpha=0.8)

        enu_info = {}
        x = []
        y = []
        z = []
        for key, location in gps_info.items():
            enu_info[key] = pm.geodetic2enu(location['lat'],location['lon'],location['alt'],origin[0],origin[1],origin[2])
            distance = numpy.sqrt(enu_info[key][0]**2 + enu_info[key][1]**2 + enu_info[key][2]**2)
            if distance < plot_distance_threshold_1: 
                #Don't plot photos from barcroft, only site
                ax.scatter(enu_info[key][0], enu_info[key][1], enu_info[key][2], marker='.',color='k',alpha=0.8)                
                x.append(enu_info[key][0])
                y.append(enu_info[key][1])
                z.append(enu_info[key][2])
        if plot_surface:
            surf = ax.plot_trisurf(x, y, z, cmap=plt.cm.coolwarm,linewidth=0, antialiased=True,alpha=0.5)
        if plot_pictures:
            for index, image_name in enumerate(list(gps_info.keys())):
                annotate3D(ax, s=str(index), xyz=(x[index],y[index],z[index]), fontsize=10, xytext=(-3,3), textcoords='offset points', ha='right',va='bottom')
        if plot_best_fit_plane:
            #Solving for plane Ax = b
            A = numpy.matrix(numpy.vstack((x,y,numpy.ones_like(x))).T)
            b = numpy.matrix(z).T
            fit, residual, rnk, s = lstsq(A, b)
            print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            wire_frame_spacing_m = 5#m
            X,Y = numpy.meshgrid(numpy.arange(xlim[0], xlim[1],wire_frame_spacing_m),
                              numpy.arange(ylim[0], ylim[1],wire_frame_spacing_m))
            Z = numpy.zeros(X.shape)
            for r in range(X.shape[0]):
                for c in range(X.shape[1]):
                    Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]
            ax.plot_wireframe(X,Y,Z, color='k',alpha=0.1,label='Best Fit Plane\nEW Slope: %0.1f deg\nNS Slope: %0.1f deg'%(numpy.rad2deg(numpy.arctan(fit[0])),numpy.rad2deg(numpy.arctan(fit[1]))))

            altitude_offset_ft = 8
            print('\nAdding additional offset of %i ft to altitude to account for offset between my phone height and mast height'%altitude_offset_ft)
            plane = lambda E, N: fit[0] * E + fit[1] * N +  fit[2] + altitude_offset_ft*0.3048
            
            print('\nUsing input GPS positions of antennas, the new coordinates (with modified altitude) are:')
            for i, a in antennas_physical.items():
                lat, lon, alt = pm.enu2geodetic(a[0],a[1],plane(a[0],a[1]),origin[0],origin[1],origin[2])
                print('Antenna %i\n%s : (%f,%f,%f)\n%s : (%f,%f,%f)'%(i,'{:15}'.format('(E, N, U)'),a[0],a[1],plane(a[0],a[1]),'{:15}'.format('(lat, lon, alt)'),lat,lon,alt))
                ax.scatter(a[0],a[1],plane(a[0],a[1]), marker='+',s=50,color=colors[i],label='%i New Initial Position'%(i),alpha=1.0)





        ax.set_xlabel('E (m)')
        ax.set_ylabel('N (m)')
        ax.set_zlabel('Relative Elevation (m)')
        plt.legend()

        columns = 3
        rows = int(numpy.ceil(len(x)/columns))

        if plot_pictures:
            #Want twice as many columns, and only plot on the right half
            right_index_array = numpy.reshape(numpy.arange(rows*columns),(rows,columns))
            left_index_array = numpy.ones_like(right_index_array)*-999

            image_index_array_2d = numpy.hstack((left_index_array,right_index_array))
            image_index_array_flat = image_index_array_2d.flatten()

            for index, image_name in enumerate(list(gps_info.keys())):
                plt.subplot(rows,2*columns,numpy.where(image_index_array_flat == index)[0][0] + 1)
                plt.axis('off')
                plt.imshow(plt.imread(datapath + image_name)) #Just testing plotting last image
                #plt.title(str(index))
                #plt.title(image_name)
                patch = mpatches.Patch(color='white', alpha=0.0, label=str(index) + ' : ' + image_name.split('_')[-1].replace('.jpg',''))
                plt.legend(handles=[patch],fontsize=8)



        if False:
            datapath = os.environ['BEACON_ANALYSIS_DIR'] + 'data/pictures/2019/' #'data/pictures/2019/'
            hillside_files = numpy.array(os.listdir(os.environ['BEACON_ANALYSIS_DIR'] + 'data/pictures/2019_hillside/'))
            hillside_files = hillside_files[~numpy.isin(hillside_files,exclude_images)]
            files = numpy.array(os.listdir(datapath))
            files = files[~numpy.isin(files,exclude_images)]
            verbose = False

            gps_raw_info = {}
            gps_info = {}
            for image_name in files:
                if image_name.split('.')[-1].lower() != 'jpg':
                    continue
                image = Image.open(datapath + image_name)
                exifdata = image.getexif()
                for tag_id in exifdata:
                    # get the tag name, instead of human unreadable tag id
                    tag = TAGS.get(tag_id, tag_id)
                    data = exifdata.get(tag_id)
                    # decode bytes 
                    if isinstance(data, bytes):
                        data = data.decode()
                    if tag == 'GPSInfo':
                        gps_raw_info[image_name] = data.copy()
                        try:
                            gps_info[image_name] = {'lat' : [-1,1][data[1] == 'N'] * (data[2][0] + data[2][1]/60.0 + data[2][2]/3600.0),\
                                                    'lon' : [-1,1][data[3] == 'E'] * (data[4][0] + data[4][1]/60.0 + data[4][2]/3600.0),\
                                                    'alt' : data[6]}
                        except:
                            print('Failed to load data for '+image_name)
                    if verbose:
                        print(f"{tag:25}: {data}")
            #Duplicate of above (Maybe not update as much) with different display settings
            plot_distance_threshold_2 = 1300 #m
            
            fig = plt.figure()
            fig.canvas.set_window_title('Antenna Locations Full')
            ax = fig.add_subplot(111, projection='3d')

            for i, a in antennas_physical.items():
                distance = numpy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                if distance < plot_distance_threshold_2:
                    ax.scatter(a[0], a[1], a[2], marker='o',color=colors[i],label='Physical %i'%i,alpha=0.8)

            if plot_phase == True:
                for i, a in antennas_phase_hpol.items():
                    distance = numpy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                    if distance < plot_distance_threshold_2:
                        ax.plot([antennas_physical[i][0],antennas_phase_hpol[i][0]],[antennas_physical[i][1],antennas_phase_hpol[i][1]],[antennas_physical[i][2],antennas_phase_hpol[i][2]],color=colors[i],linestyle='--',alpha=0.5)
                        ax.scatter(a[0], a[1], a[2], marker='*',color=colors[i],label='%s Phase Center %i'%('Hpol', i),alpha=0.8)
                for i, a in antennas_phase_vpol.items():
                    distance = numpy.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                    if distance < plot_distance_threshold_2:
                        ax.plot([antennas_physical[i][0],antennas_phase_vpol[i][0]],[antennas_physical[i][1],antennas_phase_vpol[i][1]],[antennas_physical[i][2],antennas_phase_vpol[i][2]],color=colors[i],linestyle='--',alpha=0.5)
                        ax.scatter(a[0], a[1], a[2], marker='^',color=colors[i],label='%s Phase Center %i'%('Vpol', i),alpha=0.8)

            for site, key in enumerate(['run1507','run1509','run1511']):
                site += 1
                distance = numpy.sqrt(pulser_locations['physical'][key][0]**2 + pulser_locations['physical'][key][1]**2 + pulser_locations['physical'][key][2]**2)
                if distance < plot_distance_threshold_2:
                    ax.scatter(pulser_locations['physical'][key][0], pulser_locations['physical'][key][1], pulser_locations['physical'][key][2], color=pulser_colors[site-1], marker='o',label='Physical Pulser Site %i'%site,alpha=0.8)

            enu_info = {}
            x = numpy.array([])
            y = numpy.array([])
            z = numpy.array([])
            r = numpy.array([])
            hillside_cut = numpy.array([],dtype=bool)
            for key, location in gps_info.items():
                enu_info[key] = pm.geodetic2enu(location['lat'],location['lon'],location['alt'],origin[0],origin[1],origin[2])
                distance = numpy.sqrt(enu_info[key][0]**2 + enu_info[key][1]**2 + enu_info[key][2]**2)
                if distance < plot_distance_threshold_2: 
                    #Don't plot photos from barcroft, only site
                    ax.scatter(enu_info[key][0], enu_info[key][1], enu_info[key][2], marker='.',color='k',alpha=0.8)                
                    x = numpy.append(x,enu_info[key][0])
                    y = numpy.append(y,enu_info[key][1])
                    z = numpy.append(z,enu_info[key][2])
                    r = numpy.append(r,distance)
                    hillside_cut = numpy.append(hillside_cut,key in hillside_files)


            if plot_surface:
                surf = ax.plot_trisurf(x[hillside_cut], y[hillside_cut], z[hillside_cut], cmap=plt.cm.coolwarm,linewidth=0, antialiased=True,alpha=0.5)

            ax.set_xlabel('E (m)')
            ax.set_ylabel('N (m)')
            ax.set_zlabel('Relative Elevation (m)')
            plt.legend()