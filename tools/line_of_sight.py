'''
This tool is designed to use the code written by Cosmin to retreive ground locations in Lat Lon that likely correspond
to signals coming from an observed azimuth and zenith direction.  It achieves this via line of sight calculations
and a model of the topology near BEACON.  
'''

import matplotlib.pyplot as plt
plt.ion()
import ROOT
import os
import sys
import numpy
import pymap3d as pm
import simplekml
site_map_datapath = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','map_data','beacon-site.root')

f = ROOT.TFile.Open(site_map_datapath)
dmap = f.Get("invelevmap")
latmap = f.Get("invelevmaplat")
lonmap = f.Get("invelevmaplon")

def interpolateLatLon(azimuth_deg, elevation_deg):
    '''
    Given a azimuth_deg and elevation angle it will do the required angle conversion to interface with the ROOT code. 
    This expects input of azimuth_deg/el from an ENU perspective (0 az = East, 90 az = North) and converts them.  
    '''
    #convert to north based az
    az_interal = 90-azimuth_deg  # I think ? check sign
    if az_interal < -180:
        az_interal += 360
    if az_interal > 180:
        az_interal -= 360
    return ( latmap.Interpolate(az_interal,elevation_deg), lonmap.Interpolate(az_interal,elevation_deg))

def circleSource(azimuth_deg, elevation_deg, radius_deg, n_points=1000, spread_limit=5, save_kml=False, save_name=None, save_description=None, color_rgba=(255,0,255,255)):
    '''
    This will create a circle centered on azimuth_deg, elevation_deg, and return latitude and longitude
    projected intersection points.  
    '''
    if save_kml == True:
        print('Attempting to save the circle as a kml file.')
        kml_color_hex = simplekml.Color.rgb(int(color_rgba[0]),int(color_rgba[1]),int(color_rgba[2]),a=int(color_rgba[3]))
        print(kml_color_hex)
        kml = simplekml.Kml()

        if save_name is None:
            if numpy.size(radius_deg) == 1:
                if azimuth_deg < 0:
                    save_name = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','map_data','output',str('az_E%0.3fS_elevation_%0.3f+-%s'%(abs(azimuth_deg), elevation_deg, str(radius_deg))).replace('.','p') + '.kml') 
                else:
                    save_name = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','map_data','output',str('az_E%0.3fN_elevation_%0.3f+-%s'%(azimuth_deg, elevation_deg, str(radius_deg))).replace('.','p') + '.kml') 
            else:
                if azimuth_deg < 0:
                    save_name = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','map_data','output',str('az_E%0.3fS_elevation_%0.3f'%(abs(azimuth_deg), elevation_deg)).replace('.','p') + '.kml') 
                else:
                    save_name = os.path.join(os.environ['BEACON_ANALYSIS_DIR'],'tools','map_data','output',str('az_E%0.3fN_elevation_%0.3f'%(azimuth_deg, elevation_deg)).replace('.','p') + '.kml') 



    if numpy.size(radius_deg) == 1:
        radius_deg = numpy.array([float(radius_deg)])
    else:
        radius_deg = numpy.asarray(radius_deg)

    all_lat = []
    all_lon = []
    for rad_index, rad in enumerate(radius_deg):
        if azimuth_deg < 0:
            label = 'Az: E %0.3f S, El: %0.3f, Radius: %0.3f'%(abs(azimuth_deg), elevation_deg, rad)
        else:
            label = 'Az: E %0.3f N, El: %0.3f, Radius: %0.3f'%(abs(azimuth_deg), elevation_deg, rad)

        t = numpy.linspace(0,2*numpy.pi, n_points)
        x = azimuth_deg + rad * numpy.cos(t)#rad * (1 - t**2) / (1 + t**2)
        y = elevation_deg + rad * numpy.sin(t)#rad * (2 * t) / (1 + t**2)

        #import pdb; pdb.set_trace()

        lat = numpy.zeros_like(x)
        lon = numpy.zeros_like(x)
        central_lat, central_lon = interpolateLatLon(azimuth_deg, elevation_deg)


        for index, (az_deg, el_deg) in enumerate(zip(x,y)):
            lat[index], lon[index] = interpolateLatLon(az_deg, el_deg)

        cut = ~numpy.logical_or(numpy.abs(lat-central_lat) > spread_limit, numpy.abs(lon-central_lon) > spread_limit)
        lat = lat[cut]
        lon = lon[cut]

        all_lat.append(lat)
        all_lon.append(lon)

        if save_kml == True:
            if save_description is None:
                save_description = 'E %0.3f S +- %0.3f'%(abs(azimuth_deg), rad) + ': A curve generated with approximate resoltion to help identify potential sources in that source direction.'
            coords = list(zip(lon,lat))
            #import pdb; pdb.set_trace()
            line = kml.newlinestring(name=label, description=save_description, coords = coords)
            line.style.linestyle.color = kml_color_hex
            line.style.linestyle.width = 5

            if rad_index == 0:
                pnt = kml.newpoint(name=label.split(', Radius')[0], description="Central Point of Plotted Circle", coords=[(central_lon,central_lat)])
                pnt.style.iconstyle.scale = 2
                pnt.style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/paddle/wht-blank.png'
                pnt.style.iconstyle.color = kml_color_hex


    if save_kml:
        kml.save(save_name)
        return all_lat, all_lon, kml
    else:
        return all_lat, all_lon


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        azimuth_deg = float(sys.argv[1])
        elevation_deg = float(sys.argv[2])
        if len(sys.argv) > 3:
            radius_deg = float(sys.argv[3])
        else:
            radius_deg = numpy.array([1])


        lat_map, lon_map = interpolateLatLon(azimuth_deg, elevation_deg)
        print(lat_map, lon_map)

        save_kml = True

        if save_kml:
            lat_map, lon_map, out_kml = circleSource(azimuth_deg, elevation_deg, radius_deg, save_kml=True)
            print(lat_map, lon_map)

    else:
        print('Not yet setup, but this should plot the visible distances for BEACON.')
