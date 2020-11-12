# BEACON

## People 

### Current Contributor:

 - Dan Southall 
dsouthall@uchicago.edu


# OVERVIEW

This contains the analysis tools I've created/used for working with BEACON data.

Currently this is only contains working files that I (Dan Southall) have been working with, and is not intended as a comprehensive BEACON toolkit.

# Table of Contents

0.0.0 [Prep Work](#000-prep-work)

0.1.0 [Dependencies](#010-dependencies)

0.2.0 [Paths](#020-paths)

1.0.0 [Calibration](#100-calibration)

2.0.0 [Usage](#200-usage)

2.1.0 [Farming File Creation](#210-farming-file-creation)

2.2.0 [Existing Analysis Scripts](#220-existing-analysis-scripts)

2.3.0 [Adding Analysis Scripts](#230-adding-analysis-scripts)

3.0.0 [Analysis](#300-analysis)

3.1.0 [Plotting](#300-plotting)


---

## 0.0.0 Prep Work

## 0.1.0 Dependencies

The code was developed using compatible builds of Python 3.7.1 (or newer) and ROOT 6.16.00 (or newer).  A module has been built on Midway2 specifically for this purpose, however due to recent changes in how UChicago/RCC handle python modules this is no longer available.  An updated process has been developed to handle the new guidlines.  This will bring the user into a conda environment with ROOT installed.  The user will need to recompile beaconroot with the new version of ROOT if they want this code to work.  Because this module is no longer a custom built module, it does not have all of the python packages we use pre-loaded.  See [Section 0.1.1](https://github.com/djsouthall/beacon/blob/master/README.md#011-known-missing-python-modules) for a list of known missing modules, and how to install them to run this BEACON analysis code.  Python and ROOT can still be loaded using the command:

    . beacon/loadmodules

This will unload the current ROOT and Python modules and load the recommended versions.

Python packages:

numpy - http://www.numpy.org/

scipy - http://www.scipy.org/

matplotlib / pylab - http://matplotlib.org/

Certain portions of the code require large amounts of memory.  If the code is breaking this may be something to check.

This code also uses:
beaconroot - https://github.com/beaconTau/beaconroot 

## 0.1.1 Known Missing Python Modules

For any of the following modules that you find are missing (by running code and getting errors), simply run the line of code:

    pip3 install PACKAGE
    
With the appropriate version of python loaded.

**Known Packages**

- astropy
- pymap3d

## 0.2.0 Paths

Many of the scripts in this beacon analysis git will expect some system variables to be set:
  - *BEACON_INSTALL_DIR* : This is the location of where beaconroot is installed (see the beaconroot guide).  Note that the examples folder has typically been copied to this directory such that the script defining the reader is easily found.
  - *BEACON_DATA* : This is the location of the BEACON data.  This will be the folder than contins myriad of run folders you will want to examine with this code. 
  - *BEACON_ANALYSIS_DIR* : This is the location of this package (the folder that contains the .git file).

I am also moving towards using package-like import, which will require you to add the relevant paths to your PYTHONPATH.  An example of the lines I have in my bashrc for BEACON can be found below:

BEACON_ANALYSIS_DIR="/home/dsouthall/Projects/Beacon/beacon/"
export BEACON_ANALYSIS_DIR
export PYTHONPATH=$PYTHONPATH:/home/dsouthall/Projects/Beacon/

## 1.0.0 Calibration

Position calibration for BEACON antennas is done using some of the analysis scripts, with the resulting calibrated position values being stored in the tools/info.py script.  By setting the default_deploy_index you can ensure that the correct calibration is called when working in all other analysis scripts. 


## 2.0.0 Usage

The general use of this code depends on first creating hdf5 analysis files.  These files contain the calculated numbers produced by many of the analsysis scripts.  By precomputing these values and storing them in hdf5 files, the actual plotting and intrepretation side of analysis can occur much faster.  Analysis files are created and managed by the tools/data_handler.py scipt/class.  This class internally supports certain calculated quantities as "musts" that are defined by the class.  If the class is called to open a file and those parameters aren't present then it will calculate them.  Other parameters are added adhoc by various analysis scripts.  These analysis scripts are mentioned in more detail below.

## 2.1.0 Farming File Creation

Analysis files are created using the tools/data_handler.py script, with additional important analysis datasets being added by running other analysis files.  This is all typically done initially by preparing the tools/farm.py script to run the analysis/all_analysis.sh script for all run numbers you want analysis files for.  With the appropriate python version loaded, running the tools/farm.py will send the analysis/all_analysis.sh jobs to the cluster.  Each calling of analysis/all_analysis.sh will call the listed (in analysis/all_analysis.sh) analysis scripts, and add the resulting datasets to the analysis hdf5 file for later use.

## 2.2.0 Existing Analysis Scripts

Below is a list of the current set of analysis scripts that are in analysis/all_analysis.sh.  There are not currently descriptions of each file here. 

tools/data_handler.py
analysis/save_time_delays.py
analysis/rf_bg_search.py
analysis/similarity.py
analysis/impulsivity.py
analysis/correlate_with_background_templates.py
analysis/cr_search/simple_cr_template_search.py
analysis/time_averaged_spectrum.py

## 2.3.0 Adding Analysis Scripts

To add an additional analysis script it is recommended to look at the general structure of an existing analysis file and ensure it operates similarly.  Typically this means allowing the run number to be a script input parameter, and then adding it to the list of scripts in the analysis/all_analysis.sh script.

## 3.0.0 Analysis

Here analysis specifically means work done "after batch", i.e. working with the precomputed data, doing things like making maps, histograms, etc.

## 3.1.0 Plotting

The tools/data_slicer.py and internally defined class is currently the main way to do post analysis.  It provides many tools for interpreting the pre-computed data. 



