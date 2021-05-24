# BEACON

## People 

### Programmer:

Dan Southall - dsouthall@uchicago.edu


# OVERVIEW

This contains the analysis tools I've created/used for working with BEACON data.  The general structure of this analysis packages consists of a series of developed classes of [tools](https://github.com/djsouthall/beacon/tree/master/tools), that provide helpful features for accessing and analyzing BEACON data.  These tools are then used as needed in [analysis scripts](), most of which are intended to investigate singular questions or serve particular purposes.  

Most of these analysis scripts tend to focus less on generality, and use the tools to generate plots and analyze data as was appropriate for the goal of that script.  Much of the work done in these analysis scripts depends on the accessing measured event-by-event information that has been pre-computed and stored in hdf5 files (referred hereafter as analysis files).  More information on generating these analysis files can be found [Section 2.1.0](https://github.com/djsouthall/beacon/blob/master/README.md#210-farming-file-creation)

Many of the tools will depend on stored meta data such as antenna positions.  This information is generally stored in plain text in the [tools/info.py](https://github.com/djsouthall/beacon/tree/master/tools/info.py) script.  Some data is stored in .csv files or calculated (such as known pulser or airplane events, coordinates of antennas and sources in ENU) but these can also be easily loaded using the functions in [tools/info.py](https://github.com/djsouthall/beacon/tree/master/tools/info.py).  Antenna positions, pulser positions, and cable delays are all examples of information found in the [tools/info.py](https://github.com/djsouthall/beacon/tree/master/tools/info.py) script, but they must be handled with particular caution as they are *calibration dependant*.  When this information is accessed be sure that you are using the appropriate *deploy_index* (or hard coding the *default_deploy* at the top of the [tools/info.py](https://github.com/djsouthall/beacon/tree/master/tools/info.py) script), as this determines which calibration is being used when returning coordinates and information.


# Table of Contents

0.0.0 [Prep Work](#000-prep-work)

0.1.0 [Dependencies](#010-dependencies)

0.2.0 [Installing on Midway](#020-installing-on-midway)

0.2.1 [Loading the Python and ROOT Environment on Midway](#021-loading-the-python-and-root-environment-on-midway)

0.2.2 [Installing Beaconroot](#022-installing-beaconroot)

0.2.3 [Installing libRootFftwWrapper](#023-installing-librootfftwwrapper)

0.2.4 [Known Missing Python Modules](#024-known-missing-python-modules)

0.2.5 [Paths](#025-paths)

0.3.0 [Installing on Windows](#030-installing-on-windows)

0.4.0 [Installing on Ubuntu](#040-installing-on-ubuntu)

0.4.1 [Preparing Ubuntu](#041-preparing-ubuntu)

0.4.1 [Installing FFTW3](#041-installing-fftw3)

0.4.2 [Optional Git Setup](#042-optional-git-setup)

0.4.3 [Getting Setup to Code](#043-getting-setup-to-code)

0.4.4 [Optional iPython Setup](#044-optional-ipython-setup)

0.5.0 [Testing Setup](#050-testing-setup)

1.0.0 [Calibration](#100-calibration)

2.0.0 [Usage](#200-usage)

2.1.0 [Farming File Creation](#210-farming-file-creation)

2.2.0 [Existing Analysis Scripts](#220-existing-analysis-scripts)

2.3.0 [Adding Analysis Scripts](#230-adding-analysis-scripts)

3.0.0 [Analysis](#300-analysis)

3.1.0 [Sample Scripts](#310-sample-scripts)

3.2.0 [Plotting](#320-plotting)


---

## 0.0.0 Prep Work

This sections describes how to get setup running the code.  Much of these instructions assume that you are *starting from scratch*, so feel free to skip steps if you are confident you know what you are doing.

* If starting on UChicago's Midway supercomputing cluster I recommend starting with [Section 0.2.0](https://github.com/djsouthall/beacon/blob/master/README.md#020-installing-on-midway).
* If you are starting on a Windows computer I recommend starting with [Section 0.3.0](https://github.com/djsouthall/beacon/blob/master/README.md#020-installing-on-windows), which will tell you how to get Ubuntu runnin on a modern windows pc, before moving on to [Section 0.4.0](https://github.com/djsouthall/beacon/blob/master/README.md#040-installing-on-ubuntu).
* If you are starting from a linux capable PC (linux, Mac, or Windows with WSL), then I recommend you start at [Section 0.4.0](https://github.com/djsouthall/beacon/blob/master/README.md#040-installing-on-ubuntu).


## 0.1.0 Dependencies

The code was developed using compatible builds of Python 3.7.1 (or newer) and ROOT 6.22.02 (or newer).  It may work with older versions, but no guarentees are made.  For maximum ease-of-startup it is recommended that the user utilize the Conda environment on [Midway (UChicago's high-performance computing system)])(https://rcc.uchicago.edu/docs/) that has been created to ensure compatable versions of python and ROOT are available and loaded.  Setup for Midway is outline in [Section 0.2.1](https://github.com/djsouthall/beacon/blob/master/README.md#021-loading-the-python-and-root-environment-on-midway).  Details are also provided for getting started in general on Ubuntu in [Section 0.4.0](https://github.com/djsouthall/beacon/blob/master/README.md#040-installing-on-ubuntu), as well as getting started on Windows via installing Ubuntu using the Windows Subsytem for Linux Version 2 (WSL2) in [Section 0.3.0](https://github.com/djsouthall/beacon/blob/master/README.md#030-installing-on-windows).

This analysis framework is built using the BEACON event reader developed by [Cosmin Deaconu](https://github.com/cozzyd).  The [beaconroot](https://github.com/beaconTau/beaconroot) repository must be installed as described in that packages [README](https://github.com/beaconTau/beaconroot/blob/master/README.md).   This reader utilizes compiled C++/ROOT code to do the underlying event handling, and provides a simple python class which is what is directly referenced in this analysis code.  See [Section 0.2.2](https://github.com/djsouthall/beacon/blob/master/README.md#022-installing-beaconroot).

The majority of the scripts in this analysis package utilize the [FFTPrepper](https://github.com/djsouthall/beacon/blob/2d2233c13ca2d1d659f543ce8e78c44b760a49ba/tools/fftmath.py#L43) class, which acts as a wrapper class on the beaconroot Reader class - providing additional tools for streamlining the process of upsampling, and filtering (additionally there are some daughter classes defined to aid in cross correlations / time delay calculations as well as comparing events to a provided template event).  As part of the optional filtering available in these classes, the so-called Sine Subtraction method of CW removal is available for use when loading signals.  A FFTPrepper object can have these [SineSubtract](https://github.com/djsouthall/beacon/blob/master/tools/sine_subtract.py) objects added to it for use when loading signals.  These utilize code from [libRootFftwWrapper](https://github.com/nichol77/libRootFftwWrapper).  Instructions for installing libRootFftWrapper are included in multiple locations below, however you can also follow the instructions in the [README](https://github.com/nichol77/libRootFftwWrapper/blob/master/README.md) of that package.  Ensure to install with the correct version of python and ROOT.  See [Section 0.2.3](https://github.com/djsouthall/beacon/blob/master/README.md#013-installing-libRootFftwWrapper).

Finally, there are required python packages that are required for most scripts in this analysis package.  See [Section 0.2.4](https://github.com/djsouthall/beacon/blob/master/README.md#014-known-missing-python-modules) for a list of known missing modules from midway, and how to install them to run this BEACON analysis code.  In general you can see a list at [requirments](https://github.com/djsouthall/beacon/blob/master/requirements.txt).

Note that the details for getting started on Midway and through Ubuntu should largely overlap, but may use different methods (for instance the Ubuntu instructions involve a conda environment being created).  If you have the know-how you can pick and choose.

## 0.2.0 Installing on Midway

## 0.2.1 Loading the Python and ROOT Environment on Midway

Python and ROOT can still be loaded using the command:

    . beacon/loadmodules

This executes a simple bash script that will unload the current ROOT and Python modules and load the recommended versions.  Issues have occured in the past when this is called multiple times in the same instance of a terminal, so if there is a problem it is recommended you close the current terminal, run this line of code once, and then proceed. 

Python packages:

numpy - http://www.numpy.org/

scipy - http://www.scipy.org/

matplotlib / pylab - http://matplotlib.org/

A additional required modules can be found in [Section 0.2.4](https://github.com/djsouthall/beacon/blob/master/README.md#024-known-missing-python-modules).


## 0.2.2 Installing Beaconroot

Follow the instructiuons specified in the packages [README](https://github.com/beaconTau/beaconroot/blob/master/README.md).  The main causes of problems when installing this package tend to be not properly defining and exporting the environment variables.  ROOT, Beaconroot, and libFFtwWrapper, must all be installed using the same compiler, and discprency is a common cause of issues when attempting this install.   It is recommended you open a clean new terminal, load in the correct versions of python.  For additional troubleshooting steps you can check the recommendations in [Section 0.2.3](https://github.com/djsouthall/beacon/blob/master/README.md#023-installing-libRootFftwWrapper).


## 0.2.3 Installing libRootFftwWrapper

Follow the instructiuons specified in the packages [README](https://github.com/nichol77/libRootFftwWrapper/blob/master/README.md).  See below for instructions specific to installing on Midway with the suggested Conda environment.

ROOT, Beaconroot, and libFFtwWrapper, must all be installed using the same compiler, and discprency is a common cause of issues when attempting this install.   It is recommended you open a clean new terminal, load in the correct versions of python.

The README assumes that your compiler is already loaded, however on Midway it may be necessary to execute:

    module load cmake

You can check the version of the compiler by typing:

    make configure

and then hitting t. This can be compared with the ROOT compiler listed next to the *std=* flag when executing::

    root-config --cflags

If there is an issue when running the make command as specified in the README, try checking the above.  If errors are present you can execute:

    make distclean

followed by:

    make

This may reset the make configuration file generated when make was run initially, hopefully now listing the correct compilers. 

## 0.2.4 Known Missing Python Modules

For any of the following modules that you find are missing (by running code and getting errors), simply run the line of code:

    pip3 install PACKAGE
    
With the appropriate version of python loaded.

**Known Packages**

- astropy
- pymap3d

## 0.2.5 Paths

Many of the scripts in this beacon analysis git will expect some system variables to be set:
  - *BEACON_INSTALL_DIR* : This is the location of where beaconroot is installed (see the beaconroot guide).  Note that the examples folder has typically been copied to this directory such that the script defining the reader is easily found.
  - *BEACON_DATA* : This is the location of the BEACON data.  Each run will have a folder containing the header, event, and status files.
  - *BEACON_PROCESSED_DATA* : This is the location of generated hdf5 analysis files for BEACON.  Any computations like time delays or reconstruction directions are stored here.
  - *BEACON_ANALYSIS_DIR* : This is the location of this package (the folder that contains the .git file).

I am also moving towards using package-like import, which will require you to add the relevant paths to your PYTHONPATH.  An example of the lines I have in my bashrc for BEACON can be found below:

BEACON_ANALYSIS_DIR="/home/dsouthall/Projects/Beacon/beacon/"
export BEACON_ANALYSIS_DIR
export PYTHONPATH=$PYTHONPATH:/home/dsouthall/Projects/Beacon/

## 0.3.0 Installing on Windows

This section actually gets you to the point where you can have installed and setup WSL2, and then can follow [Section 0.4.0](https://github.com/djsouthall/beacon/blob/master/README.md#040-installing-on-ubuntu)

Getting setup with Windows Subsystem for Linux - Most of this section can be ignored if you are on a Mac or linux machine.

1. Follow this video to get Ubuntu setup in WSL2: https://www.youtube.com/watch?v=_fntjriRe48&ab_channel=DavidBombal 
    * You will want to update to a recent version of windows.  This will take a long time and should be done in advance
2. To get GUI interface for WSL2 install xming.
3. Allow xming through your firewall on both private and public networks.
4. Run xlaunch -> Multiple windows -> Display number 0 -> Next -> Start no client -> Next -> Clipboard enabled -> No Access Control enabled -> Next -> Finish
5. Run Ubuntu and perform any required initial setup if not already complete.
    * `cd ~`
    * `vim .bashrc`
    * `i` - This will let you edit the file in vim
    * Add a new line to the bashrc that says:
    * `export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0`
6. You can now test that the graphical elements are working by running something like xeyes
    * `sudo apt-get install x11-apps`
    * `xeyes` - A pair of eyes should pop up that track mouse movement.  You can close this with ctrl-c
    * If this did not work then try double checking your firewall settings.  Additionally hover over the xming app in the tray of our windows toolbar.  Ensure that it reads the same port number that you have set in your bashrc.  Above you were instructed to use Display number 0, but it may autoset to 0.0.  In which case it may be easier to update your bash script to match what xming wants to do: `export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0.0`
7. You can see where your WSL files are stored by executing `explorer.exe .`, which will open Windows Explorer in the current directory.  You can then copy and past the path from above.  This is where you should direct any code editor, if you want to run windows programs to edit the code on your linux subsystem.

## 0.4.0 Installing on Ubuntu

This assumes you have a functional Ubuntu kernal.  Other linux systems can be used but you will need to convert some functions and syntax, the details of which are not outlined here.  If you need to get setup with Ubuntu on Windows see [Section 0.3.0](https://github.com/djsouthall/beacon/blob/master/README.md#030-installing-on-windows)

## 0.4.1 Preparing Ubuntu

Before installing the code, you need to make sure you can install packages like make and cmake.  When errors occurred for me the following link helped: https://linuxize.com/post/how-to-install-pip-on-ubuntu-18.04/ .  Ideally you can ignore that and just type the following:
    * `sudo apt update`
    * `sudo apt-get update`
    * `sudo apt install python3-pip`
    * `sudo apt-get install make`
    * `sudo apt-get install cmake`
    * If no errors occur then you should be good to go.

## 0.4.1 Installing FFTW3

Install FFTW3: `sudo apt-get install -y libfftw3-dev`

## 0.4.2 Optional Git Setup
Do this if you plan on using git commits and pushes.

1. Update git username: `git config --global user.name "Your Name"`
2. Update git email: `git config --global user.email "your@email"`
3. Optionally set vim as your editor: `git config --global core.editor "vim"`
4. If you have 2FA setup then pushing will take more then a simple password.  You will need to "Generate an Access Token" for your password from this specific pull.  I would google this.  You likely don't need strong access, I enable `repo`, and `write:discussion`.
5. If you don't want to enter these every time you wish to push you can setup a credential cache by following this guide: 
    * https://docs.github.com/en/github/getting-started-with-github/getting-started-with-git/caching-your-github-credentials-in-git 
    * You will still be prompted for the generated personal access token the first time you push, but it will cache this for the time window you give it.

## 0.4.3 Getting Setup to Code
1. Get the Conda installation file: `wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
2. Install Conda: `bash ./Anaconda3-2020.11-Linux-x86_64.sh`, Follow the prompts.
3. Create the root environment: `conda create -n my_root_env root -c conda-forge`
4. Activate the environment: `conda activate my_root_env` - You can consider adding this to bashrc if you think you will always want to be in this environment.
5. Create a BEACON directory:
    * `cd ~`
    * `mkdir beacon` - this can alternatively be done in any folder other than your home/~ folder.  I typically have a `projects` folder.  In which case I perform this after `cd ~/projects` .  You will need to account for this in step 12 and onward.
    * cd beacon
6. Clone this repository: `git clone https://github.com/djsouthall/beacon`
7. Clone the beaconroot repository: `git clone https://github.com/beaconTau/beaconroot`
8. Clone the libRootFftwWrapper repository: `git clone https://github.com/nichol77/libRootFftwWrapper`
9. Make a directory to store data: `mkdir data`
10. Type:
    * `cd $CONDA_PREFIX`
    * `mkdir -p ./etc/conda/activate.d`
    * `mkdir -p ./etc/conda/deactivate.d`
    * `touch ./etc/conda/activate.d/env_vars.sh`
    * `touch ./etc/conda/deactivate.d/env_vars.sh`
11. Now we must define environment variables (variables accessible by the linux kernal itself, as well as within the python shell.  Typically these will be defined in bashrc, but we are setting this all up in a conda environment, so we will define them such that they are only active when in that environment.   Type: `vim $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh` This will open a text editor called vim.
12. Starting from the top of the text editor write (replace "username" with your own):
    * `#!/bin/sh` <- Can add this at the top of the script if it does not already exist.  Not necessary likely.

            export BEACON_INSTALL_DIR=/home/username/beacon/beaconroot/
            export BEACON_ANALYSIS_DIR=/home/username/beacon/beacon/
            export BEACON_DATA=/home/username/beacon/data/
            export BEACON_PROCESSED_DATA=/home/username/beacon/data/processed/
            export LIB_ROOT_FFTW_WRAPPER_DIR=/home/username/beacon/libRootFftwWrapper/
            export BEACON_ANALYSIS_DIR=/home/username/beacon/beacon/
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BEACON_INSTALL_DIR/lib:$ANITA_INSTALL_DIR/lib
            export PYTHONPATH=$PYTHONPATH:/home/username/beacon/:/home/username/beacon/beaconroot/
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$BEACON_INSTALL_DIR/lib
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$LIB_ROOT_FFTW_WRAPPER_DIR
            export PYTHONPATH=$PYTHONPATH:/home/username/beacon/
13. Save and exit by typing esc followed by: `:x` , then enter
14. Reactivate the environment: `conda activate my_root_env`
15. Type:
    * `cd ~/beacon/libRootFftwWrapper`
    * `make`
    * `make install` - if there are troubles try `sudo make install`
    * `cd ~/beacon/beaconroot`
    * `make`
    * `make install` - if there are troubles try `sudo make install`
    * if there are problems during or after running make you may need to remove the build directory in that respect folder.  Ensure you are in the correct package directory and type `rm -rf build` before attempting make again.
16. Test if all of this worked by doing:
    * `cd ~/beacon`
    * `python beacon/analysis/sample_script_B.py`
17. The test will likely end in an error due to a missing dependency (such as pandas, matplotlib, etc.). For each error you get, install the missing dependency: `conda install -c conda-forge <missing package>`  .  A list of expected dependencies can be seen at https://github.com/djsouthall/beacon/blob/master/requirements.txt .  You can attempt to install all requirements using the command `conda install --file requirements.txt` while in the folder containing requirements.txt.  This will install what it can, of the remaining it is likely they are custom or default packages.  Likely you can ignore these unless an obvious problem occurs at some later time. 
18. Keep repeating steps 16-17 until you get an output that displays the number of events.
19. At this point you may start coming across errors related to not having data.  This is great!  Copy data to the data directory we made earlier and hopefully all is good moving forward. If you do not know where to access the data contact Dan Southall or someone else in the know.
20. To deactivate your environment type: `conda deactivate`

## 0.4.4 Optional iPython Setup
If it is not already installed I would recommend getting and using it.  It is a better command line interface (CLI) for python.
1. `sudo apt-get install ipython3`
2. `ipython`
3. `%run ~/beacon/beacon/analysis/sample_script_B.py` - This or any other script can be used to test running.  By using the %run shortcut in the CLI, you will stay in the namespace of the executed script, which lets you play around with plots, variables, and also helps in debugging.
4. Create an ipython profile and enable the autoreload function following this post: https://stackoverflow.com/a/43020072

## 0.5.0 Testing Setup

Attempt running both test analysis scripts to check if the code is functioning correctly:
* [analysis/sample_script_A.py](https://github.com/djsouthall/beacon/blob/master/analysis/sample_script_A.py) 
* [analysis/sample_script_B.py](https://github.com/djsouthall/beacon/blob/master/analysis/sample_script_B.py) 

## 1.0.0 Calibration

Position calibration for BEACON antennas is done using some of the analysis scripts, with the resulting calibrated position values being stored in the [tools/info.py](https://github.com/djsouthall/beacon/tree/master/tools/info.py) script.  By setting the default_deploy_index you can ensure that the correct calibration is called when working in all other analysis scripts. 


## 2.0.0 Usage

The general use of this code depends on first creating hdf5 analysis files.  These files contain the calculated numbers produced by many of the analsysis scripts.  By precomputing these values and storing them in hdf5 files, the actual plotting and intrepretation side of analysis can occur much faster.  Analysis files are created and managed by the [tools/data_handler.py](https://github.com/djsouthall/beacon/tree/master/tools/data_handler.py) scipt/class.  This class internally supports certain calculated quantities as "musts" that are defined by the class.  If the class is called to open a file and those parameters aren't present then it will calculate them.  Other parameters are added adhoc by various analysis scripts.  These analysis scripts are mentioned in more detail below.

## 2.1.0 Farming File Creation

Analysis files are created using the [tools/data_handler.py](https://github.com/djsouthall/beacon/tree/master/tools/data_handler.py) script, with additional important analysis datasets being added by running other analysis files.  This is all typically done initially by preparing the [tools/farm.py](https://github.com/djsouthall/beacon/tree/master/tools/farm.py) script to run the analysis/all_analysis.sh script for all run numbers you want analysis files for.  With the appropriate python version loaded, running the [tools/farm.py](https://github.com/djsouthall/beacon/tree/master/tools/farm.py) will send the analysis/all_analysis.sh jobs to the cluster.  Each calling of analysis/all_analysis.sh will call the listed (in analysis/all_analysis.sh) analysis scripts, and add the resulting datasets to the analysis hdf5 file for later use.

## 2.2.0 Existing Analysis Scripts

Below is a list of the current set of analysis scripts that are in analysis/all_analysis.sh.  The user should be aware that each of the following scripts has a series of parameters that can be adjusted at the top of the __main__() function.  These should be explored before farming the analysis scripts to ensure they are doing what is intended. 

[tools/data_handler.py](https://github.com/djsouthall/beacon/tree/master/tools/data_handler.py)
    Checks if an alysis file exists for the specified run.  If a file does not exist then one will be generated and stored in the BEACON_DATA folder (as described in [Section 0.2.5](https://github.com/djsouthall/beacon/blob/master/README.md#025-paths) or [Section 0.4.3](https://github.com/djsouthall/beacon/blob/master/README.md#043-getting-setup-to-code))).  If a file exists it will ensure the file has the necessary required datasets.  If the file does not Creates the analysis file (if it does not exist).  
[analysis/save_time_delays.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/save_time_delays.py)
    Calculates and stores the time delays for each of the baselines/polarizations.  These are stored with the meta-data describing the filters/other parameters used for calculating the time delays.  Multiple time delay datasets can exist in a single analysis file, but they currently must have slightly different parameter selection to not be overwritten.  For instance you can run the [analysis/save_time_delays.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/save_time_delays.py) script to measure time delays with and without Hilbert envelopes applied.
[analysis/rf_bg_search.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/rf_bg_search.py)
    Operates similar to [analysis/save_time_delays.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/save_time_delays.py) but uses the Correlator class to determine the current best guess at pointing/source direction for each event.  
[analysis/flag_cw.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/flag_cw.py)
    Loops over events with a the Sine Subtraction method, and stores information about located CW. 
[analysis/similarity.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/similarity.py)
    Calculates a metric for "similarity" among different events, by comparing the calculated time delays from [analysis/save_time_delays.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/save_time_delays.py).  Gives a rough count of the number of events in that run with a comperable set of time delays.
[analysis/impulsivity.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/impulsivity.py)
    Calculates and stores the impulsivity for each event.  This loops over all stored time delay datasets, and utilizes those time delays to shift and sum signals to calculate the metric of impulsivity. 
[analysis/correlate_with_background_templates.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/correlate_with_background_templates.py)
    Calculates the correlation value of every event with any user-specified event templates.
[analysis/cr_search/simple_cr_template_search.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/cr_search/simple_cr_template_search.py)
    Calculates the correlation value of every event with simple model of a cosmic ray signal (generate by convolving the system response with a bi-polar delta function).
[analysis/time_averaged_spectrum.py](https://github.com/djsouthall/beacon/tree/master/tools/analysis/time_averaged_spectrum.py)
    This performs calculations that can be used to explore the noise level of subsets of the spectrum as a function of time.  Intended for investing sky noise dominence.

## 2.3.0 Adding Analysis Scripts

To add an additional analysis script it is recommended to look at the general structure of an existing analysis file and ensure it operates similarly.  Typically this means allowing the run number to be a script input parameter, and then adding it to the list of scripts in the [analysis/all_analysis.sh script](https://github.com/djsouthall/beacon/tree/master/tools/analysis/all_analysis.sh).

## 3.0.0 Analysis

Here analysis specifically means work done "after batch", i.e. working with the precomputed data, doing things like making maps, histograms, etc.  This section also includes any discussion of analysis that does not require precomputing values.  

## 3.1.0 Sample Scripts

Two sample scripts have been created for onboarding purposes.  The first ([analysis/sample_script_A.py](https://github.com/djsouthall/beacon/tree/master/analysis/sample_script_A.py)) is intended to demonstrate how one can load in events, antenna positions, etc.  This mostly aims to dodge the use of most of the tools developed for handling this data, and just provide an example of the raw event handling and how one can access information like pulser eventids.

[analysis/sample_script_B.py](https://github.com/djsouthall/beacon/tree/master/analysis/sample_script_B.py) is meant to showcase some of the tools, by loading in an event, plotting the waveforms and correlation maps, with various filters applied.  This still does not attempt to present the full capabilities, as it is intended to be useful before any analysis files have been generated (and thus does not use the [tools/data_slicer.py](https://github.com/djsouthall/beacon/tree/master/tools/data_slicer.py) tool).

## 3.2.0 Plotting

The [tools/data_slicer.py](https://github.com/djsouthall/beacon/tree/master/tools/data_slicer.py) and internally defined class is currently the main way to do post analysis.  It provides many tools for interpreting the pre-computed data.
