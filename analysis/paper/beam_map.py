#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy

import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import os
from matplotlib import cm
from beacon.analysis.paper.new_beam_plot import makeBeamPlot







if __name__ == '__main__':
    # In[8]:


    azimuths = numpy.arange(-90, 90.1, 1)
    zeniths = numpy.arange(0, 100.1, 1)


    # In[17]:

    data_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/beam_maps'
    out_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/figures'


    azimuth_grid = numpy.load(os.path.join(data_path,"azimuth_grid_all_beams.npy"))
    zenith_grid = numpy.load(os.path.join(data_path,"zenith_grid_all_beams.npy"))
    power_array = numpy.load(os.path.join(data_path,"power_grid_all_beams.npy"))

    major_fontsize = 24
    minor_fontsize = 20


    beams = {
                0 : {"theta": 90, "phi": -50},
                1: {"theta": 90, "phi": -25},
                2: {"theta": 90, "phi": 0},
                3: {"theta": 90, "phi": 25},
                4: {"theta": 90, "phi": 50},
                5: {"theta": 30, "phi": -20},
                6: {"theta": 43, "phi": -20},
                7: {"theta": 57, "phi": -20},
                8: {"theta": 70, "phi": -20},
                9: {"theta": 30, "phi": 20},
                10: {"theta": 43, "phi": 20},
                11: {"theta": 57, "phi": 20},
                12: {"theta": 70, "phi": 20},
                13: {"theta": 80, "phi": 0},
                14: {"theta": 70, "phi": 0},
                15: {"theta": 60, "phi": 0},
                16: {"theta": 50, "phi": 0},
                17: {"theta": 40, "phi": 0},
                18: {"theta": 30, "phi": 0},
                19: {"theta": 20, "phi": 0}
                }


    thetas = numpy.array([])
    phis = numpy.array([])
    for i in range(20):
        if i < 5:
            continue
        thetas = numpy.append(thetas, beams[i]['theta'])
        phis = numpy.append(phis, beams[i]['phi'])
    sort_indices = numpy.lexsort((phis,90-thetas), axis=0)


    c1 = numpy.zeros((20,4))
    c1[0:5]               = numpy.asarray([cm.GnBu(x) for x in numpy.linspace(0.3, 0.9, 5)])
    c1[5+sort_indices]    = numpy.asarray([cm.YlOrRd(x) for x in numpy.linspace(0.2, 0.9, 15)])[::-1]

    c2 = numpy.zeros((20,4))
    c2[0:5]               = numpy.asarray([cm.YlOrRd(x) for x in numpy.linspace(0.3, 0.9, 5)])
    c2[5+sort_indices]    = numpy.asarray([cm.GnBu(x) for x in numpy.linspace(0.2, 0.9, 15)])[::-1]


    for cmap_index, colors in enumerate([c1, c2]):
        if cmap_index == 0:
            continue

        # In[5]:


        powerdB = 10*numpy.log10(power_array)

        fig = plt.figure(figsize=(9,10))
        ax = plt.subplot(2,1,1)


        
        cmap = plt.get_cmap('Greys')
        max_power = max(power_array.flatten())
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.pcolormesh(azimuth_grid, 90-zenith_grid, power_array/max_power, shading="gouraud", norm=norm, cmap='Greys', rasterized=True)
        ax.set_facecolor("#440154")
        plt.ylim(-10,90)
        #plt.gca().invert_yaxis()
        cbar = plt.colorbar(sm)

        # plot circles around beams, with radius 3 dB less than max of that beam
        # text_cmap = plt.get_cmap('hsv')
        # text_colors = numpy.linspace(0,1,20)


        for i in range(20):
            
            theta = beams[i]["theta"]
            phi = beams[i]["phi"]
            theta_idx = numpy.where(zeniths == theta)[0][0]
            phi_idx = numpy.where(azimuths == phi)[0][0]
            beam_power = powerdB[theta_idx][phi_idx]
            j = 0
            beam_dB_less = 100
            while beam_dB_less >= beam_power-3:
                j += 1
                beam_dB_less = powerdB[theta_idx-j][phi_idx]
            txt = plt.text(beams[i]["phi"]+3 + 2*(i<5), 90-beams[i]["theta"]+3, i, fontsize=minor_fontsize, c=colors[i])
            txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='black')])
            circle = plt.Circle((beams[i]["phi"], 90-beams[i]["theta"]), j, edgecolor=colors[i], fill=False)
            ax.add_patch(circle)
            
        plt.xticks(numpy.arange(-90,90.1, 30))
        plt.xlim(-90, 90)
        cbar.set_label('Normalized Power', rotation=90, labelpad=15, fontsize=major_fontsize)
        plt.xlabel(r'Azimuth (deg)', fontsize=major_fontsize)
        plt.ylabel(r'Elevation (deg)', fontsize=major_fontsize)
        
        ax2 = plt.subplot(2,1,2)
        makeBeamPlot(fig, ax2, major_fontsize=major_fontsize, minor_fontsize=minor_fontsize, mode='c', suppress_legend=True, _colors=colors)
        
        ax.xaxis.set_tick_params(labelsize=minor_fontsize)
        ax.yaxis.set_tick_params(labelsize=minor_fontsize)

        ax2.xaxis.set_tick_params(labelsize=minor_fontsize)
        ax2.yaxis.set_tick_params(labelsize=minor_fontsize)

        plt.tight_layout()

        plt.savefig(os.path.join(out_path, 'beam_map_opt%i.pdf'%(cmap_index+1)))

