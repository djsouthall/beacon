'''
Physical constants and unit conversions

Time (ns)
Mass (kg)
Distance (m)
Frequency (GHz)
Charge (C)
Time (s) # CHECK
'''

# Physical constants
speed_light = 0.299792 # m ns^-1
mass_proton = 1.6726e-27 # kg
mu_0 = 1.256637e-6 # m kg s^-2 A^-2
boltzmann = 1.3806488e-23 # J K^-1 = kg m^2 s^-2 K^-1 

# Earth
depth_ice = 3000. # m
radius_earth = 6.371e6 + depth_ice # m, from PREM, plus depth of ice
density_water = 1.e3 / mass_proton # nucleons m^-3

# Unit conversions
cm_to_m = 1.e-2
km_to_m = 1.e3
MHz_to_GHz = 1.e-3
GHz_to_Hz = 1.e9
yr_to_s = 365.25 * 24. * 60. * 60.
