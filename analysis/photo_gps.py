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

if __name__ == '__main__':
    plt.close('all')
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
    deploy_index = 12
    antennas_physical, antennas_phase_hpol, antennas_phase_vpol = info.loadAntennaLocationsENU(deploy_index=deploy_index)#)
    origin = info.loadAntennaZeroLocation(deploy_index=deploy_index)
    pulser_locations = info.loadPulserPhaseLocationsENU(deploy_index=deploy_index)
    plot_phase = False
    plot_distance_threshold_1 = 300 #m
    plot_surface = True
    plot_pictures = True
    plot_best_fit_plane = False

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
        ax.plot_wireframe(X,Y,Z, color='k',alpha=0.1,label='Best Fit Plane')

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