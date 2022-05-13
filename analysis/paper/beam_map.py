#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy

import matplotlib as mpl
import matplotlib.patheffects as PathEffects



if __name__ == '__main__':
    # In[8]:


    azimuths = numpy.arange(-90, 90.1, 1)
    zeniths = numpy.arange(0, 100.1, 1)


    # In[17]:

    data_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/beam_maps'


    azimuth_grid = numpy.load(os.path.join(data_path,"azimuth_grid_all_beams.npy"))
    zenith_grid = numpy.load(os.path.join(data_path,"zenith_grid_all_beams.npy"))
    power_array = numpy.load(os.path.join(data_path,"power_grid_all_beams.npy"))

    major_fontsize = 18
    minor_fontsize = 14


    # In[20]:


    powerdB = 10*numpy.log10(power_array)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    cmap = plt.get_cmap('Greys')
    max_power = max(power_array.flatten())
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ax.pcolormesh(azimuth_grid, 90-zenith_grid, power_array/max_power, shading="gouraud", norm=norm, cmap='Greys')
    ax.set_facecolor("#440154")
    plt.ylim(-10,90)
    #plt.gca().invert_yaxis()
    cbar = plt.colorbar(sm)

    # plot circles around beams, with radius 3 D
    text_cmap = plt.get_cmap('hsv')
    text_colors = numpy.linspace(0,1,20)
    for i in range(20):
        
        theta = det.beams[i]["theta"]
        phi = det.beams[i]["phi"]
        theta_idx = numpy.where(zeniths == theta)[0][0]
        phi_idx = numpy.where(azimuths == phi)[0][0]
        beam_power = powerdB[theta_idx][phi_idx]
        j = 0
        beam_dB_less = 100
        while beam_dB_less >= beam_power-3:
            j += 1
            beam_dB_less = powerdB[theta_idx-j][phi_idx]
        txt = plt.text(det.beams[i]["phi"]+1, 90-det.beams[i]["theta"]+1, i, fontsize=minor_fontsize, c=text_cmap(text_colors[i]))
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        circle = plt.Circle((det.beams[i]["phi"], 90-det.beams[i]["theta"]), j, edgecolor=text_cmap(text_colors[i]), fill=False)
        ax.add_patch(circle)
        
    plt.xticks(numpy.arange(-90,90.1, 30))
    plt.xlim(-90, 90)
    cbar.set_label('Normalized Power', rotation=90, labelpad=15, fontsize=major_fontsize)
    plt.xlabel(r'Azimuth (deg)', fontsize=major_fontsize)
    plt.ylabel(r'Elevation (deg)', fontsize=major_fontsize)
    plt.savefig("/home/avz5228/Pictures/beam_map.pdf")

