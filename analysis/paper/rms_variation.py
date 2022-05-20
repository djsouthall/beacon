#!/usr/bin/env python3
# coding: utf-8

from datetime import datetime
from datetime import timedelta
import calendar
from scipy.fft import rfft, rfftfreq
import numpy
import numpy.ma as ma
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz, SkyCoord
import os
plt.ion()




if __name__ == '__main__':
    plt.close('all')
    major_fontsize = 24
    minor_fontsize = 18

    std_vs_time_list = numpy.load(os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/rms_galaxy' , 'runs_5375_to_5450.npy'), allow_pickle=True)

    unix_times = []
    for std_vs_time in std_vs_time_list:
        for i in range(len(std_vs_time)):
            if i%60 == 0:
                unix_times.append(std_vs_time[i][0])



    ast_time = Time(unix_times, format='unix') 
        
    sun = get_sun(ast_time)
    galaxy = SkyCoord.from_name('Sag A*')
    beacon = EarthLocation(lat=37.589339*u.deg, lon=-118.23761867*u.deg, height=3.8505272*u.km)
    sun_altaz = sun.transform_to(AltAz(obstime=ast_time, location=beacon))
    sag_altaz = galaxy.transform_to(AltAz(obstime=ast_time, location=beacon))
    alt_sun = sun_altaz.alt.value
    alt_galaxy = sag_altaz.alt.value


    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(2, 1, 1)
    #plt.figure(figsize=(10,4))
    for std_vs_time in std_vs_time_list:
        ax1.plot(std_vs_time[:,0], std_vs_time[:,1], '.', ms=0.5, alpha=0.1, c="black", lw=4)
    plt.ylim(1,1.8)
    ax1.get_xaxis().set_ticklabels([])
    ax1.set_ylabel("RMS (adu)", fontsize=major_fontsize)

    ax2 = fig.add_subplot(2, 1, 2)
    #plt.figure(figsize=(10,4))
    ax2.plot(unix_times, alt_sun, label="Sun", c='r', lw=4)
    ax2.plot(unix_times, alt_galaxy, label="Galaxy", c="dodgerblue", lw=4)
    ax2.set_ylabel("Elevation (deg)", fontsize=major_fontsize)
    ax2.legend(loc=1, fontsize=minor_fontsize)
    ax2.set_xlabel("Unix Time (s)", fontsize=major_fontsize)

    plt.tight_layout()


    rms = []
    t = []
    for run in std_vs_time_list[:23]:
        for std in run[:,1]:
            rms.append(std)
        for time in run[:,0]:
            t.append(time)

    masked_rms = ma.masked_greater(numpy.array(rms), 1.6)
    masked_rms2 = ma.masked_less(masked_rms, 1.2)
    split_rms = numpy.array_split(masked_rms2, 2000)
    split_time = numpy.array_split(numpy.array(t), 2000)

    mean_rms = []
    median_time = []
    for array in split_rms:
        mean_rms.append(numpy.mean(array))
    for array in split_time:
        median_time.append(numpy.median(array))

    ast_time = Time(median_time, format='unix') 
        
    sun = get_sun(ast_time)
    galaxy = SkyCoord.from_name('Sag A*')
    beacon = EarthLocation(lat=37.589339*u.deg, lon=-118.23761867*u.deg, height=3.8505272*u.km)
    sun_altaz = sun.transform_to(AltAz(obstime=ast_time, location=beacon))
    sag_altaz = galaxy.transform_to(AltAz(obstime=ast_time, location=beacon))
    alt_sun = sun_altaz.alt.value
    alt_galaxy = sag_altaz.alt.value


    fig = plt.figure(figsize=(8,5))
    ax1 = fig.add_subplot(2, 1, 1)
    #plt.figure(figsize=(10,4))
    ax1.plot(median_time, mean_rms, '.', ms=5, alpha=1, c="black", lw=4)
    plt.ylim(1.2,1.5)
    ax1.get_xaxis().set_ticklabels([])
    ax1.set_ylabel("RMS (adu)", fontsize=major_fontsize)

    ax2 = fig.add_subplot(2, 1, 2)
    #plt.figure(figsize=(10,4))
    ax2.plot(median_time, alt_sun, label="Sun", c='r', lw=4)
    ax2.plot(median_time, alt_galaxy, label="Galaxy", c='dodgerblue', lw=4)
    ax2.set_ylabel("Elevation (deg)", fontsize=major_fontsize)
    ax2.legend(loc=1, fontsize=minor_fontsize)
    ax2.set_xlabel("Unix Time (s)", fontsize=major_fontsize)

    ax1.xaxis.set_tick_params(labelsize=minor_fontsize)
    ax1.yaxis.set_tick_params(labelsize=minor_fontsize)
    ax2.xaxis.set_tick_params(labelsize=minor_fontsize)
    ax2.yaxis.set_tick_params(labelsize=minor_fontsize)

    ax2.minorticks_on()
    ax2.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    ax2.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.25)

    ax1.minorticks_on()
    ax1.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    ax1.grid(b=True, which='minor', color='tab:gray', linestyle='--',alpha=0.25)



    plt.tight_layout()







    fig = plt.figure(figsize=(10,6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    #plt.figure(figsize=(10,4))

    times = (median_time - min(median_time))/3600.0
    ax1.plot(times, mean_rms, '.',label='Force Triggers RMS', ms=5, alpha=1, c="black", lw=4)

    rms_amp = (1.45 - 1.29)/2
    rms_mid = (1.45 + 1.29)/2
    galaxy_amp = (max(alt_galaxy) - min(alt_galaxy))/2
    galaxy_mid = (max(alt_galaxy) + min(alt_galaxy))/2
    sun_amp = (max(alt_sun) - min(alt_sun))/2
    sun_mid = (max(alt_sun) + min(alt_sun))/2

    # ax2.plot(times, (rms_amp/sun_amp)*alt_sun + rms_mid, label="Scaled Galaxy", c='r', lw=4)
    # ax2.plot(times, (rms_amp/galaxy_amp)*alt_galaxy + rms_mid, label="Scaled Galaxy", c='dodgerblue', lw=4)

    ax1.set_ylabel("RMS (adu)", fontsize=major_fontsize)

    ax2.plot(times, alt_sun, label="Sun", c='r', lw=4)
    ax2.plot(times, alt_galaxy, label="Galaxy", c='dodgerblue', lw=4)
    ax2.set_ylabel("Elevation (deg)", fontsize=major_fontsize)
    ax1.legend(loc='lower left', fontsize=minor_fontsize, framealpha=1)
    ax2.legend(loc='lower right', fontsize=minor_fontsize, framealpha=1)
    ax1.set_xlabel("Time (h, arb offset)", fontsize=major_fontsize)
    ax1.set_ylim(1.25,1.5)
    ax2.set_ylim(-130,120)
    ax2.set_yticks([-90,-60,-30,0,30,60,90])

    ax1.xaxis.set_tick_params(labelsize=minor_fontsize)
    ax1.yaxis.set_tick_params(labelsize=minor_fontsize)
    ax2.xaxis.set_tick_params(labelsize=minor_fontsize)
    ax2.yaxis.set_tick_params(labelsize=minor_fontsize)

    ax2.grid(b=True, which='major', color='k', linestyle='-', alpha=0.5)
    ax1.grid(b=True, which='major', color='tab:gray', linestyle='--',alpha=0.25)

    ax2.set_xticks(numpy.arange(0, 24*3.5, 12))
    ax2.set_xlim(0,69)

    plt.tight_layout()
    fig.savefig('./figures/rms_galaxy.pdf')









