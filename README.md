# BEACON

## People 

### Current Contributor:

 - Dan Southall 
dsouthall@uchicago.edu


# OVERVIEW

This contains the analysis tools I've created/used for working with BEACON data.

Currently this is only contains working files that I (Dan Southall) have been working with, and is not intended as a comprehensive BEACON toolkit.

# Table of Contents

0.0.0 [Prep Work](#prep-work)

0.1.0 [Dependencies](#dependencies)

0.2.0 [Paths](#paths)


---

## 0.0.0 Prep Work

## 0.1.0 Dependencies

The code was developed using compatible builds of Python 3.7.1 and ROOT 6.16.00.  A module has been built on Midway2 specifically for this purpose and can be loaded using the command:

    . beacon/loadmodules

This will unload the current ROOT and Python modules and load the recommended versions.

Python packages:

numpy - http://www.numpy.org/

scipy - http://www.scipy.org/

matplotlib / pylab - http://matplotlib.org/

Certain portions of the code require large amounts of memory.  If the code is breaking this may be something to check.

This code also uses:
beaconroot - https://github.com/beaconTau/beaconroot 

## 0.2.0 Paths

Many of the scripts in this beacon analysis git will expect two system variables to be set:
  - *BEACON_INSTALL_DIR* : This is the location of where beaconroot is installed (see the beaconroot guide)/
  - *BEACON_DATA* : This is the location of the BEACON data.  This will be the folder than contins myriad of run folders you will want to examine with this code. 
