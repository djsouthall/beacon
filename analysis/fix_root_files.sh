#!/bin/bash
# This will regenerate the status and header trees for the range of runs given.   
echo 'Fixing Status Trees'
for i in {1600..1750}; do ${BEACON_INSTALL_DIR}bin/beaconroot-convert status /project2/avieregg/beacon/telem/raw/run$i/status/ /project2/avieregg/beacon/telem/root/run$i/status.root ; done
echo 'Fixing Header Trees'
for i in {1600..1750}; do ${BEACON_INSTALL_DIR}bin/beaconroot-convert header /project2/avieregg/beacon/telem/raw/run$i/header/ /project2/avieregg/beacon/telem/root/run$i/header.root ; done