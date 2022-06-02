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



#std_vs_time_list = numpy.load(os.path.join('/home/dsouthall/Projects/Beacon/beacon/analysis/paper/data/rms_galaxy' , 'runs_5375_to_5450.npy'), allow_pickle=True)

runs = numpy.arange(5800,7000)
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



masked_rms = ma.masked_greater(numpy.array(rms), 1.6)
masked_rms2 = ma.masked_less(masked_rms, 1.2)
split_rms = numpy.array_split(masked_rms2, 50000)
split_time = numpy.array_split(numpy.array(t), 50000)

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



dB = 10*numpy.ma.log10(mean_rms)
dB = dB - numpy.ma.mean(dB)

r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = numpy.linspace(-4, 4, 500)
    y = numpy.exp( -t**2 ) + numpy.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, numpy.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """


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



if __name__ == '__main__':
    plt.close('all')

    include_cbar = True
    normalize_counts = True
    log_counts = False

    major_fontsize = 24
    minor_fontsize = 18
    # fig = plt.figure(figsize=(10,6))
    Z_solar, xedges_solor, yedges_solor = numpy.histogram2d(numpy.array(solar_day_frac), numpy.array(dB), range=[[0,1],[-0.5,0.5]], bins=75)
    Z_sid, xedges_sid, yedges_sid = numpy.histogram2d(numpy.array(sid_day_frac), numpy.array(dB), range=[[0,1],[-0.5,0.5]], bins=75)

    min_Z = min(numpy.min(Z_sid[Z_sid>0]),numpy.min(Z_solar[Z_solar>0]))
    max_Z = max(numpy.max(Z_sid),numpy.max(Z_solar))
    cmap = plt.get_cmap("coolwarm")
    if normalize_counts:
        Z_sid = Z_sid/max_Z
        Z_solar = Z_solar/max_Z
        max_Z = 1.0

    if log_counts:
        norm = matplotlib.colors.LogNorm(vmin=min_Z, vmax=max_Z)
    else:
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_Z)


    fig, axes = plt.subplots(2,2, figsize=(12,8))

    ax1 = plt.subplot(2, 2, 1)# fig.add_subplot(2, 2, 1)
    ax1.get_xaxis().set_ticklabels([])
    plt.xticks(fontsize=minor_fontsize)
    plt.yticks(fontsize=minor_fontsize)
    plt.xlim(0,1)
    #plt.figure(figsize=(10,4))

    im = ax1.pcolormesh(xedges_sid, yedges_sid, Z_sid.T, norm=norm, cmap=cmap, rasterized=True)

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
        ax1.set_ylabel("Noise (dB)", fontsize=major_fontsize)

        plt.tight_layout()
        fig.subplots_adjust(right=0.85)


    plt.subplot(2, 2, 3)# fig.add_subplot(2, 2, 3)
    plt.xticks(fontsize=minor_fontsize)
    plt.yticks(fontsize=minor_fontsize)
    plt.xlim(0,1)
    plt.ylim(-95,65)
    plt.gca().get_yaxis().set_ticks([-90, -60, -30, 0, 30, 60])


    sampled_x, sampled_y_min, sampled_y_max = scatterToFill(sid_day_frac, alt_sun)
    plt.fill_between(sampled_x, sampled_y_min, sampled_y_max, alpha=0.5, step='mid', color="dodgerblue", label="Sun")
    plt.plot(sampled_x[::5], sampled_y_min[::5], alpha=0.7, lw=3, color="dodgerblue")
    plt.plot(sampled_x[::5], sampled_y_max[::5], alpha=0.7, lw=3, color="dodgerblue")

    sampled_x, sampled_y_min, sampled_y_max = scatterToFill(sid_day_frac, alt_galaxy)
    plt.fill_between(sampled_x, sampled_y_min, sampled_y_max, alpha=0.5, step='mid', color="tab:red", label="Galaxy")
    plt.plot(sampled_x[::5], sampled_y_min[::5], alpha=0.7, lw=3, color="tab:red")
    plt.plot(sampled_x[::5], sampled_y_max[::5], alpha=0.7, lw=3, color="tab:red")


    # plt.plot(sid_day_frac, alt_sun, '.', alpha=0.1, label="sun", c='k')
    # plt.plot(sid_day_frac, alt_galaxy, '.', alpha=0.1, label="galaxy", c='k')
    #plt.plot(day_frac, alt_sun, '.', label="Sun")
    plt.ylabel("Elevation (deg)", fontsize=major_fontsize)
    plt.xlabel("Fraction, Sidereal Day", fontsize=major_fontsize)

    ax1.set_xlim(0,1)
    ax2 = plt.subplot(2, 2, 2)# fig.add_subplot(2, 2, 2)
    ax2.get_xaxis().set_ticklabels([])
    plt.xticks(fontsize=minor_fontsize)
    plt.yticks(fontsize=minor_fontsize)
    plt.xlim(0,1)
    im = ax2.pcolormesh(xedges_solor, yedges_solor, Z_solar.T, norm=norm, cmap=cmap, rasterized=True)

    # ax2.set_ylabel("Noise (dB)", fontsize=major_fontsize)

    ax3 = plt.subplot(2, 2, 4)# fig.add_subplot(2, 2, 4)
    plt.xticks(fontsize=minor_fontsize)
    plt.yticks(fontsize=minor_fontsize)
    plt.xlim(0,1)
    plt.ylim(-95,65)
    plt.gca().get_yaxis().set_ticks([-90, -60, -30, 0, 30, 60])

    sampled_x, sampled_y_min, sampled_y_max = scatterToFill(solar_day_frac, alt_sun)
    plt.fill_between(sampled_x, sampled_y_min, sampled_y_max, alpha=0.5, step='mid', color="dodgerblue", label="Sun")
    plt.plot(sampled_x[::5], sampled_y_min[::5], alpha=0.7, lw=3, color="dodgerblue")
    plt.plot(sampled_x[::5], sampled_y_max[::5], alpha=0.7, lw=3, color="dodgerblue")

    sampled_x, sampled_y_min, sampled_y_max = scatterToFill(solar_day_frac, alt_galaxy)
    plt.fill_between(sampled_x, sampled_y_min, sampled_y_max, alpha=0.5, step='mid', color="tab:red", label="Galaxy")
    plt.plot(sampled_x[::5], sampled_y_min[::5], alpha=0.7, lw=3, color="tab:red")
    plt.plot(sampled_x[::5], sampled_y_max[::5], alpha=0.7, lw=3, color="tab:red")
    plt.legend(loc='lower right', fontsize=minor_fontsize)

    # plt.plot(solar_day_frac, alt_sun, '.', alpha=0.05, label="sun", c='k')
    # plt.plot(solar_day_frac, alt_galaxy, '.', alpha=0.05, label="galaxy", c='k')

    # plt.legend(fontsize=minor_fontsize)
    # #plt.plot(day_frac, alt_sun, '.', label="Sun")
    # plt.legend(fontsize=minor_fontsize)
    # plt.ylabel("Elevation (deg)", fontsize=major_fontsize)
    plt.xlabel("Fraction, Solar Day", fontsize=major_fontsize)
    ax2.set_xlim(0,1)

    # plt.colorbar(im, ax=plt.gcf().get_axes().ravel().tolist())

    plt.tight_layout()
    if include_cbar == True:
        fig.subplots_adjust(right=0.85)
    fig.savefig('./figures/diurnal.pdf')

