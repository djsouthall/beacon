import numpy
import os
import sys
sys.path.append(os.environ['BEACON_INSTALL_DIR'])
from examples.beacon_data_reader import Reader
import matplotlib.pyplot as plt
plt.ion()

if __name__ == '__main__':
    plt.close('all')
    # If your data is elsewhere, pass it as an argument
    datapath = sys.argv[1] if len(sys.argv) > 1 else os.environ['BEACON_DATA']
    run = 367 #Selects which run to examine
    reader = Reader(datapath,run)

    