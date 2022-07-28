#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# coding: utf-8

from datetime import datetime
from datetime import timedelta
import calendar
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq
import numpy
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation, AltAz, SkyCoord
import inspect
import os
import sys
plt.ion()



def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    from math import factorial
    
    try:
        window_size = numpy.abs(numpy.int(window_size))
        order = numpy.abs(numpy.int(order))
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        import pdb; pdb.set_trace()
    # except ValueError, msg:
    #     raise ValueError("window_size and order have to be of type int")
    # if window_size % 2 != 1 or window_size < 1:
    #     raise TypeError("window_size size must be a positive odd number")
    # if window_size < order + 2:
    #     raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = numpy.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = numpy.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - numpy.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + numpy.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = numpy.concatenate((firstvals, y, lastvals))
    return numpy.convolve( m[::-1], y, mode='valid')


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also: 

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y


def scatterToFill(x,y,x_step_min=5/(24*60)):
    '''
    For every x_step values this will calculate the maximum and minimum y value,
    then use the set of those values to get a fill_between.
    '''
    try:
        indices = numpy.argsort(x)
        x = numpy.asarray(x)
        y = numpy.asarray(y)

        x = numpy.concatenate(numpy.vstack((x[indices] - 1, x[indices], x[indices] + 1)))
        y = numpy.concatenate(numpy.vstack((y[indices], y[indices], y[indices])))
        indices = numpy.argsort(x)

        x = numpy.copy(numpy.asarray(x)[indices])
        unique_x = numpy.unique(x)
        unique_x = unique_x[::3]
        y = numpy.copy(numpy.asarray(y)[indices])

        _y_min = numpy.zeros_like(unique_x)
        _y_max = numpy.zeros_like(unique_x)
        for i, x_val in enumerate(unique_x):
            cut = numpy.logical_and(x >= x_val - x_step_min/2, x < x_val + x_step_min/2)
            
            sorted_y = numpy.sort(y[cut])
            # numpy.mean(sorted_y[0:10])
            # numpy.mean(sorted_y[::-1][0:10])
            _y_min[i] = numpy.mean(sorted_y[0:2])#numpy.min(y[cut])
            _y_max[i] = numpy.mean(sorted_y[::-1][0:2])#numpy.max(y[cut])
            # cut = x == x_val
            # _x[i] = numpy.mean(x[cut])
            # _y_min[i] = numpy.min(y[cut])
            # _y_max[i] = numpy.min(y[cut])

        if False:
            _y_min = smooth(_y_min,window_len=11,window='hanning')
            _y_max = smooth(_y_max,window_len=11,window='hanning')
        else:
            _y_min = savitzky_golay(_y_min, 100, 4, deriv=0, rate=1)
            _y_max = savitzky_golay(_y_max, 100, 4, deriv=0, rate=1)
        sampled_x = numpy.linspace(0, 1, 1000)

        sampled_y_min = interp1d(unique_x, _y_min,kind='cubic', bounds_error=False, fill_value='extrapolate')(sampled_x)
        sampled_y_max = interp1d(unique_x, _y_max,kind='cubic', bounds_error=False, fill_value='extrapolate')(sampled_x)

        return sampled_x, sampled_y_min, sampled_y_max
    except Exception as e:
        print('\nError in %s'%inspect.stack()[0][3])
        print(e)
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        import pdb; pdb.set_trace()



def produceCurvesFromTimes(median_times):
    '''
    Given median times as defined by Andrew, this will produce the required
    upsampled curves.
    '''
    upsampled_times = numpy.linspace(min(median_time), max(median_time), 50000)
    ast_time = Time(upsampled_times, format='unix') 
        
    sun = get_sun(ast_time)
    galaxy = SkyCoord.from_name('Sag A*')
    beacon = EarthLocation(lat=37.589339*u.deg, lon=-118.23761867*u.deg, height=3.8505272*u.km)
    sun_altaz = sun.transform_to(AltAz(obstime=ast_time, location=beacon))
    sag_altaz = galaxy.transform_to(AltAz(obstime=ast_time, location=beacon))
    alt_sun = sun_altaz.alt.value
    alt_galaxy = sag_altaz.alt.value
    az_galaxy = sag_altaz.az.value

    sidereal_time = ast_time.sidereal_time('apparent', beacon)

    hour = sidereal_time.hour

    sid_day_frac = sidereal_time.hour/24

    solar_day_frac = []
    for time in upsampled_times:
        utc = datetime.utcfromtimestamp(time)
        day_frac = (utc.hour*3600 + utc.minute*60 + utc.second)/(3600*24)
        solar_day_frac.append(day_frac)

    # # import pdb; pdb.set_trace()
    # plt.figure()
    # plt.scatter(solar_day_frac,alt_sun)
    # plt.scatter(solar_day_frac,alt_galaxy)

    # window = 5/(24*60)

    # for t in linespace

    out = {}
    out['sun'] = scatterToFill(solar_day_frac,alt_sun,x_step_min=5/(24*60))
    out['galaxy'] = scatterToFill(solar_day_frac,alt_galaxy,x_step_min=5/(24*60))


    return out

curves = {}
# curves['test'] = produceCurvesFromTimes(median_time)



#std_vs_time_list = numpy.load(os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/rms_galaxy' , 'runs_5375_to_5450.npy'), allow_pickle=True)

runs = numpy.arange(5800,5900)
std_vs_time_list = []
for run in runs:
    data = numpy.load(os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/rms_galaxy/runs/' , f"{run}.npy"))#, allow_pickle=True
    std_vs_time_list.append(data.astype(list))

rms = []
t = []
for run in std_vs_time_list:
    for std in run[:,1]:
        rms.append(std)
    for time in run[:,0]:
        t.append(time)



masked_rms = ma.masked_greater(numpy.array(rms), 1.7)
masked_rms2 = ma.masked_less(masked_rms, 1.1)
split_rms = numpy.array_split(masked_rms2, 5000)
split_time = numpy.array_split(numpy.array(t), 5000)

mean_rms = []
median_time = []
for array in split_rms:
    mean_rms.append(numpy.ma.mean(array))
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
az_galaxy = sag_altaz.az.value



sidereal_time = ast_time.sidereal_time('apparent', beacon)

hour = sidereal_time.hour

sid_day_frac = sidereal_time.hour/24



solar_day_frac = []
for time in median_time:
    utc = datetime.utcfromtimestamp(time)
    day_frac = (utc.hour*3600 + utc.minute*60 + utc.second)/(3600*24)
    solar_day_frac.append(day_frac)

dB = 20*numpy.ma.log10(mean_rms)
dB = dB - numpy.ma.mean(dB)

sep_solar_day_frac = solar_day_frac
sep_dB = dB
sep_alt_sun = alt_sun
sep_alt_galaxy = alt_galaxy
curves['sep'] = produceCurvesFromTimes(median_time)



runs = numpy.arange(7000,7100)
std_vs_time_list = []
for run in runs:
    data = numpy.load(os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/rms_galaxy/runs/' , f"{run}.npy"))
    std_vs_time_list.append(data.astype(list))



rms = []
t = []
for run in std_vs_time_list:
    for std in run[:,1]:
        rms.append(std)
    for time in run[:,0]:
        t.append(time)



masked_rms = ma.masked_greater(numpy.array(rms), 1.7)
masked_rms2 = ma.masked_less(masked_rms, 1.1)
split_rms = numpy.array_split(masked_rms2, 5000)
split_time = numpy.array_split(numpy.array(t), 5000)

mean_rms = []
median_time = []
for array in split_rms:
    mean_rms.append(numpy.ma.mean(array))
for array in split_time:
    median_time.append(numpy.median(array))

dB = 20*numpy.ma.log10(mean_rms)
dB = dB - numpy.ma.mean(dB)


ast_time = Time(median_time, format='unix') 


sun = get_sun(ast_time)
galaxy = SkyCoord.from_name('Sag A*')
beacon = EarthLocation(lat=37.589339*u.deg, lon=-118.23761867*u.deg, height=3.8505272*u.km)
sun_altaz = sun.transform_to(AltAz(obstime=ast_time, location=beacon))
sag_altaz = galaxy.transform_to(AltAz(obstime=ast_time, location=beacon))
alt_sun = sun_altaz.alt.value
alt_galaxy = sag_altaz.alt.value
az_galaxy = sag_altaz.az.value



solar_day_frac = []
for time in median_time:
    utc = datetime.utcfromtimestamp(time)
    day_frac = (utc.hour*3600 + utc.minute*60 + utc.second)/(3600*24)
    solar_day_frac.append(day_frac)






feb_solar_day_frac = solar_day_frac
feb_dB = dB
feb_alt_sun = alt_sun
feb_alt_galaxy= alt_galaxy
curves['feb'] = produceCurvesFromTimes(median_time)



if __name__ == '__main__':
    plt.close('all')

    include_cbar = True
    normalize_counts = True
    log_counts = False

    major_fontsize = 36
    minor_fontsize = 20
    # fig = plt.figure(figsize=(10,6))
    Z_sep, xedges_sep, yedges_sep = numpy.histogram2d(numpy.array(sep_solar_day_frac), numpy.array(sep_dB), range=[[0,1],[-0.6,0.6]], bins=50)
    Z_feb, xedges_feb, yedges_feb = numpy.histogram2d(numpy.array(feb_solar_day_frac), numpy.array(feb_dB), range=[[0,1],[-0.6,0.6]], bins=50)

    min_Z = min(numpy.min(Z_feb[Z_feb>0]),numpy.min(Z_sep[Z_sep>0]))
    max_Z = max(numpy.max(Z_feb),numpy.max(Z_sep))
    cmap = plt.get_cmap("Greys")#plt.get_cmap("coolwarm")
    if normalize_counts:
        Z_feb = Z_feb/max_Z
        Z_sep = Z_sep/max_Z
        max_Z = 1.0

    if log_counts:
        norm = matplotlib.colors.LogNorm(vmin=min_Z, vmax=max_Z)
    else:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_Z)


    fig = plt.figure(figsize=(12,8))

    ax = fig.add_subplot(2, 2, 1)
    plt.xticks(fontsize=minor_fontsize+2)
    plt.yticks(fontsize=minor_fontsize+2)

    im = ax.pcolormesh(xedges_sep, yedges_sep, Z_sep.T, norm=norm, cmap=cmap)

    if include_cbar == True:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.55, 0.02, 0.43])
        cbar = fig.colorbar(im, cax=cbar_ax)
        if normalize_counts:
            cbar_ax.set_ylabel('Normalized Counts', fontsize=major_fontsize)
        else:
            cbar_ax.set_ylabel('Counts', fontsize=major_fontsize)
        # ax1.get_xaxis().set_ticklabels([])
        ax.set_ylabel("Noise (dB)", fontsize=major_fontsize)

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)

    # ax.get_xaxis().set_ticklabels([])
    # ax.set_ylabel("Noise (dB)", fontsize=major_fontsize)
    ax.set_title("September 2021", fontsize=major_fontsize)

    ax = fig.add_subplot(2, 2, 3)
    plt.xticks(fontsize=minor_fontsize+2)
    plt.yticks(fontsize=minor_fontsize+2)
    if True:
        plt.fill_between(curves['sep']['sun'][0], curves['sep']['sun'][1], curves['sep']['sun'][2], alpha=0.5, step='mid', color="dodgerblue", label="Sun")
        plt.plot(curves['sep']['sun'][0], curves['sep']['sun'][1], alpha=0.7, lw=3, color="dodgerblue")
        plt.plot(curves['sep']['sun'][0], curves['sep']['sun'][2], alpha=0.7, lw=3, color="dodgerblue")

        plt.fill_between(curves['sep']['galaxy'][0], curves['sep']['galaxy'][1], curves['sep']['galaxy'][2], alpha=0.5, step='mid', color="tab:red", label="Galaxy")
        plt.plot(curves['sep']['galaxy'][0], curves['sep']['galaxy'][1], alpha=1.0, lw=2, color="tab:red")
        plt.plot(curves['sep']['galaxy'][0], curves['sep']['galaxy'][2], alpha=1.0, lw=2, color="tab:red")
    else:
        plt.plot(sep_solar_day_frac, sep_alt_sun, '.', alpha=0.05, label="Sun")
        plt.plot(sep_solar_day_frac, sep_alt_galaxy, '.', alpha=0.05, label="Galaxy")


    plt.ylabel("Elevation (deg)", fontsize=major_fontsize)
    plt.xlabel("Fraction, Solar Day", fontsize=major_fontsize)
    ax.set_yticks(numpy.arange(-70,51,35))

    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([1, 0.50, 0.04, 0.5])
    # fig.colorbar(sm, cax=cbar_ax)
    # cbar_ax.set_ylabel('Counts')

    ax = fig.add_subplot(2, 2, 2)
    plt.xticks(fontsize=minor_fontsize+2)
    plt.yticks(fontsize=minor_fontsize+2)
    ax.pcolormesh(xedges_feb, yedges_feb, Z_feb.T, norm=norm, cmap=cmap)
    ax.get_xaxis().set_ticklabels([])
    ax.set_ylabel("Noise (dB)", fontsize=major_fontsize)
    ax.set_title("February 2022", fontsize=major_fontsize)

    ax = fig.add_subplot(2, 2, 4)
    plt.xticks(fontsize=minor_fontsize+2)
    plt.yticks(fontsize=minor_fontsize+2)
    if True:
        plt.fill_between(curves['feb']['sun'][0], curves['feb']['sun'][1], curves['feb']['sun'][2], alpha=0.5, step='mid', color="dodgerblue", label="Sun")
        plt.plot(curves['feb']['sun'][0], curves['feb']['sun'][1], alpha=1.0, lw=2, color="dodgerblue")
        plt.plot(curves['feb']['sun'][0], curves['feb']['sun'][2], alpha=1.0, lw=2, color="dodgerblue")

        plt.fill_between(curves['feb']['galaxy'][0], curves['feb']['galaxy'][1], curves['feb']['galaxy'][2], alpha=0.5, step='mid', color="tab:red", label="Galaxy")
        plt.plot(curves['feb']['galaxy'][0], curves['feb']['galaxy'][1], alpha=1.0, lw=2, color="tab:red")
        plt.plot(curves['feb']['galaxy'][0], curves['feb']['galaxy'][2], alpha=1.0, lw=2, color="tab:red")
    else:
        plt.plot(feb_solar_day_frac, feb_alt_sun, '.', alpha=0.05, label="Sun")
        plt.plot(feb_solar_day_frac, feb_alt_galaxy, '.', alpha=0.05, label="Galaxy")

    plt.legend()
    plt.ylabel("Elevation (deg)", fontsize=major_fontsize)
    plt.xlabel("Fraction, Solar Day", fontsize=major_fontsize)
    ax.set_yticks(numpy.arange(-70,51,35))

        # # fig.subplots_adjust(right=0.8)
        # cbar_ax = fig.add_axes([1, 0.50, 0.04, 0.5])
        # fig.colorbar(sm, cax=cbar_ax)
        # cbar_ax.set_ylabel('Counts')

    plt.tight_layout()
    if include_cbar == True:
        fig.subplots_adjust(right=0.85)

    fig.savefig('./figures/diurnal_seperated.pdf')

    wide_mode = True

    for mode in [0, 1]:

        if wide_mode == True:
            fig = plt.figure(figsize=(18,6))
            major_fontsize = 28
            minor_fontsize = 22
        else:
            fig = plt.figure(figsize=(14,7))
            major_fontsize = 36
            minor_fontsize = 20

        ax = fig.add_subplot(1, 2, 1)
        plt.xticks(fontsize=minor_fontsize+2)
        plt.yticks(fontsize=minor_fontsize+2)

        ax.set_ylabel("Noise RMS (dB)", fontsize=major_fontsize)
        ax.set_title("September 2021", fontsize=major_fontsize)

        ax.set_xlabel("Fraction, Solar Day", fontsize=major_fontsize)
        ax.set_xticks(numpy.arange(0,1.1,0.2))
        #ax.set_yticks(numpy.arange(-0.4,0.61,0.15))
        ax.set_yticks([-0.6,-0.3,0.0,0.3,0.6])
        ax.set_ylim(-0.62,0.62)

        if mode == 0:
            ax.plot(sep_solar_day_frac, sep_dB, '.', color="black", alpha=0.2)
        else:
            im = ax.pcolormesh(xedges_sep, yedges_sep, Z_sep.T, norm=norm, cmap=cmap)

        ax2 = ax.twinx()

        if True:
            ax2.fill_between(curves['sep']['sun'][0], curves['sep']['sun'][1], curves['sep']['sun'][2], alpha=0.5, step='mid', color="dodgerblue", label="Sun")
            ax2.plot(curves['sep']['sun'][0], curves['sep']['sun'][1], alpha=1.0, lw=2, color="dodgerblue")
            ax2.plot(curves['sep']['sun'][0], curves['sep']['sun'][2], alpha=1.0, lw=2, color="dodgerblue")

            ax2.fill_between(curves['sep']['galaxy'][0], curves['sep']['galaxy'][1], curves['sep']['galaxy'][2], alpha=0.5, step='mid', color="tab:red", label="Galaxy")
            ax2.plot(curves['sep']['galaxy'][0], curves['sep']['galaxy'][1], alpha=1.0, lw=2, color="tab:red")
            ax2.plot(curves['sep']['galaxy'][0], curves['sep']['galaxy'][2], alpha=1.0, lw=2, color="tab:red")
        else:
            ax2.plot(sep_solar_day_frac, sep_alt_sun, '.', alpha=0.2, label="Sun")
            ax2.plot(sep_solar_day_frac, sep_alt_galaxy, '.', alpha=0.2, label="Galaxy")

        #plt.ylabel("Elevation (deg)", fontsize=major_fontsize)
        plt.gca().get_yaxis().set_ticks([-90, -60, -30, 0, 30, 60])
        plt.ylim(-115, 75)
        ax2.set_yticks([])
        plt.xlim(0,1)

        ax3 = fig.add_subplot(1, 2, 2)
        plt.xticks(fontsize=minor_fontsize+2)
        plt.yticks(fontsize=minor_fontsize+2)

        if mode == 0:
            ax3.plot(feb_solar_day_frac, feb_dB, '.', color="black", alpha=0.2, label = 'Noise Values')
        else:
            ax3.pcolormesh(xedges_feb, yedges_feb, Z_feb.T, norm=norm, cmap=cmap)

        #ax3.set_ylabel("Noise (dB)", fontsize=major_fontsize)
        ax3.set_title("February 2022", fontsize=major_fontsize)

        ax3.set_xlabel("Fraction, Solar Day", fontsize=major_fontsize)
        ax3.set_xticks(numpy.arange(0,1.1,0.2))
        ax3.set_ylim(-0.6,0.6)

        ax3.set_yticks([])

        ax4 = ax3.twinx()
        ax2.get_shared_y_axes().join(ax2, ax4)
        plt.ylim(-115, 75)
        plt.xticks(fontsize=minor_fontsize+2)
        plt.yticks(fontsize=minor_fontsize+2)

        if True:
            ax4.fill_between(curves['feb']['sun'][0], curves['feb']['sun'][1], curves['feb']['sun'][2], alpha=0.5, step='mid', color="dodgerblue", label="Sun")
            ax4.plot(curves['feb']['sun'][0], curves['feb']['sun'][1], alpha=1.0, lw=2, color="dodgerblue")
            ax4.plot(curves['feb']['sun'][0], curves['feb']['sun'][2], alpha=1.0, lw=2, color="dodgerblue")

            ax4.fill_between(curves['feb']['galaxy'][0], curves['feb']['galaxy'][1], curves['feb']['galaxy'][2], alpha=0.5, step='mid', color="tab:red", label="Galaxy")
            ax4.plot(curves['feb']['galaxy'][0], curves['feb']['galaxy'][1], alpha=1.0, lw=2, color="tab:red")
            ax4.plot(curves['feb']['galaxy'][0], curves['feb']['galaxy'][2], alpha=1.0, lw=2, color="tab:red")
        else:
            ax4.plot(feb_solar_day_frac, feb_alt_sun, '.', alpha=0.2, label="Sun")
            ax4.plot(feb_solar_day_frac, feb_alt_galaxy, '.', alpha=0.2, label="Galaxy")


        plt.ylabel("Elevation (deg)", fontsize=major_fontsize)
        plt.xlabel("Fraction, Solar Day", fontsize=major_fontsize)
        plt.gca().get_yaxis().set_ticks([-90, -60, -30, 0, 30, 60])
        plt.ylim(-115, 75)
        plt.xlim(0,1)
        # ax4.set_yticks(numpy.arange(-70,51,35))

        handles1, labels1 = ax3.get_legend_handles_labels()
        handles2, labels2 = ax4.get_legend_handles_labels()

        handles = list(numpy.append(handles1, handles2))
        labels = list(numpy.append(labels1, labels2))
        leg = ax4.legend(handles=handles, fontsize=minor_fontsize-2, loc='lower right')

        # leg = ax3.legend(fontsize=minor_fontsize, loc='lower left')
        # leg = ax4.legend(fontsize=minor_fontsize, loc='lower right')

        # for lh in leg.legendHandles:
        #     lh.set_alpha(1)
        #     lh.set_ms(20)

        plt.tight_layout()

        if wide_mode == True:
            if mode == 0:
                fig.savefig('./figures/diurnal_combined_scatter_wide.pdf')
            elif mode == 1:
                fig.savefig('./figures/diurnal_combined_hist_wide.pdf')
        else:
            if mode == 0:
                fig.savefig('./figures/diurnal_combined_scatter.pdf')
            elif mode == 1:
                fig.savefig('./figures/diurnal_combined_hist.pdf')


