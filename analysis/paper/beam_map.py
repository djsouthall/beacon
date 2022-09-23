#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
import numpy

import matplotlib as mpl
import matplotlib.patheffects as PathEffects
import os
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from beacon.analysis.paper.new_beam_plot import makeBeamPlot
import ROOT

def get_voltage_snr(power_snr):
    return 1.806*numpy.sqrt(power_snr)-0.383
def get_power_snr(voltage_snr):
    return 0.333*voltage_snr**2 -0.217*voltage_snr+1.33


if __name__ == '__main__':
    # In[8]:
    plt.close('all')

    azimuths = numpy.arange(-90, 90.1, 1)
    zeniths = numpy.arange(0, 100.1, 1)


    # In[17]:

    data_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/beam_maps'
    out_path = '/home/dsouthall/Projects/Beacon/beacon/analysis/paper/figures'


    azimuth_grid = numpy.load(os.path.join(data_path,"azimuth_grid_all_beams.npy"))
    zenith_grid = numpy.load(os.path.join(data_path,"zenith_grid_all_beams.npy"))
    power_array = numpy.load(os.path.join(data_path,"power_grid_all_beams.npy"))


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

    views = ['vstack']#['hstack', 'original']#'vstack', 

    include_hists = True

    for view in views:
        for primary in ['voltage']:
            for cmap_index, colors in enumerate([c1, c2]):
                if cmap_index == 0:
                    continue

                powerdB = 10*numpy.log10(power_array)


                if view == 'vstack':
                    major_fontsize = 26
                    minor_fontsize = 22
                    if include_hists:
                        cbar_hack = False
                        equal_aspect = False
                        if cbar_hack == False:
                            if equal_aspect:
                                fig, [ax, ax2, ax3] = plt.subplots(3,1,figsize=(18,24),gridspec_kw={'height_ratios':[4,2,2]},constrained_layout=True)
                                ax.set_aspect('equal')
                            else:
                                fig, [ax, ax2, ax3] = plt.subplots(3,1,figsize=(18,18),gridspec_kw={'height_ratios':[4,2,2]},constrained_layout=True)
                        else:
                            fig = plt.figure(figsize=(18,18))
                            gs = plt.GridSpec(3, 2, width_ratios=[20, 1], height_ratios=[3, 2, 3])
                            ax = fig.add_subplot(gs[0, 0])
                            cbar_ax = fig.add_subplot(gs[0, 1])
                            cbar_ax.axis('off')
                            ax2 = fig.add_subplot(gs[1, 0])
                            ax3 = fig.add_subplot(gs[2, 0])
                    else:
                        fig, [ax, ax2] = plt.subplots(3,1,figsize=(18,12),gridspec_kw={'height_ratios':[3,2]},constrained_layout=True)
                elif view == 'hstack':
                    major_fontsize = 24
                    minor_fontsize = 20
                    fig, [ax, ax2] = plt.subplots(1,2,figsize=(18,8),constrained_layout=True)
                else:
                    major_fontsize = 24
                    minor_fontsize = 20
                    fig, [ax, ax2] = plt.subplots(2,1,figsize=(9,10),constrained_layout=True)

                plt.sca(ax)

                cmap = plt.get_cmap('Greys')
                max_power = max(power_array.flatten())
                norm = mpl.colors.Normalize(vmin=0, vmax=1)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                ax.pcolormesh(azimuth_grid, 90-zenith_grid, power_array/max_power, shading="gouraud", norm=norm, cmap='Greys', rasterized=True)
                ax.set_facecolor("#440154")
                plt.ylim(-10,90)
                #plt.gca().invert_yaxis()
                if view == 'hstack':
                    cbar = plt.colorbar(sm, aspect=50)
                else:
                    if include_hists:
                        if cbar_hack == True:
                            cbar = plt.colorbar(sm, cax=cbar_ax, ax=ax)
                        else:
                            if equal_aspect:
                                cbar = plt.colorbar(sm, shrink=0.8)
                            else:
                                cbar = plt.colorbar(sm)
                    else:
                        
                        cbar = plt.colorbar(sm)

                for i in range(20):
                    beams[i]['hex_color'] = mpl.colors.to_hex(colors[i])
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
                if include_hists:
                    cbar.set_label('Normalized Power', rotation=90, fontsize=major_fontsize)
                else:
                    cbar.set_label('Normalized Power', rotation=90, labelpad=15, fontsize=major_fontsize)
                cbar.ax.tick_params(labelsize=minor_fontsize)
                plt.xlabel(r'Azimuth (deg)', fontsize=major_fontsize)
                plt.ylabel(r'Elevation (deg)', fontsize=major_fontsize)
                
                # ax2 = plt.subplot(2,1,2)
                plt.sca(ax2)
                makeBeamPlot(fig, ax2, major_fontsize=major_fontsize, minor_fontsize=minor_fontsize, mode='c', suppress_legend=True, _colors=colors, primary=primary)
                
                ax.xaxis.set_tick_params(labelsize=minor_fontsize)
                ax.yaxis.set_tick_params(labelsize=minor_fontsize)

                ax2.xaxis.set_tick_params(labelsize=minor_fontsize)
                ax2.yaxis.set_tick_params(labelsize=minor_fontsize)

                # plt.tight_layout() Doesn't work with constrained_layout


                if include_hists:
                    plt.sca(ax3)
                    # ax2.get_shared_y_axes().join(ax2, ax3)
                    f = ROOT.TFile.Open(os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/', "htresh_5733_6640.root"))
                    h = f.Get("hthresh")
                    n_beams = h.GetNbinsX() # number of x bins
                    n_bins = h.GetNbinsY()

                    # plt.figure()

                    vals = {}
                    bin_centers = {}
                    max_val = 0
                    for b in range(n_beams+1):
                        if b == 0:
                            continue

                        vals['beam%i'%(b-1)] = numpy.zeros(n_bins-1)
                        bin_centers['beam%i'%(b-1)] = numpy.zeros(n_bins-1)
                        for i in range(n_bins):
                            if i == 0:
                                continue

                            bin_centers['beam%i'%(b-1)][i-1] = h.GetYaxis().GetBinCenter(i) #loaded in in power units
                            vals['beam%i'%(b-1)][i-1] = h.GetBinContent(b,i)

                        if max(vals['beam%i'%(b-1)]) > max_val:
                            max_val = max(vals['beam%i'%(b-1)])


                    for beam in range(20):
                        plt.axvline(beam, lw=1, c='k')

                        w = bin_centers['beam%i'%beam][1] - bin_centers['beam%i'%beam][0]
                        plt.barh(bin_centers['beam%i'%beam], vals['beam%i'%beam]/max_val, height=w, fc=colors[beam], left=beam)
                        lines = plt.step(bin_centers['beam%i'%beam], vals['beam%i'%beam]/max_val, where='post', drawstyle='steps', lw=1, c='k')[0]
                        x = lines.get_xdata()
                        y = lines.get_ydata()
                        lines.set_xdata(y + beam)
                        lines.set_ydata(x + w/2)

                    plt.xticks(numpy.arange(20).astype(int))

                    plt.xlim(0,20)

                    if primary == 'voltage':
                        ax3.yaxis.tick_right()
                        ax3.yaxis.set_label_position('right')
                        ax3.set_ylabel("Beam Power Threshold", fontsize=major_fontsize)#"Beam Power Threshold (SNR)"
                        plt.ylim(10,40)
                        secax3_y = ax3.secondary_yaxis('left', functions=(get_voltage_snr, get_power_snr))
                        secax3_y.set_ylabel("Beam Voltage Threshold", fontsize=major_fontsize)
                        # plt.ylim(5.5,12)
                    else:
                        plt.ylim(10,40)
                        ax3.set_ylabel("Beam Power Threshold", fontsize=major_fontsize)#"Beam Power Threshold (SNR)"
                        secax3_y = ax3.secondary_yaxis('right', functions=(get_voltage_snr, get_power_snr))
                        secax3_y.set_ylabel("Beam Voltage Threshold", fontsize=major_fontsize)
                                    
                    ax3.yaxis.set_major_formatter(FormatStrFormatter(r'%0.1f $\sigma$'))#_\mathrm{P}
                    secax3_y.yaxis.set_major_formatter(FormatStrFormatter(r'%0.1f $\sigma$'))#_\mathrm{V}

                    plt.xlabel(r'Relative Frequency (Offset By Beam Number)', fontsize=major_fontsize)




                import time
                # if cbar_hack == False:
                #     plt.tight_layout()
                plt.savefig(os.path.join(out_path, 'beam_map_opt%i_primary_axis_%s_%s_%i.pdf'%(cmap_index+1, primary, view, time.time())))
                plt.tight_layout() #must be after save for save figure to look normal
