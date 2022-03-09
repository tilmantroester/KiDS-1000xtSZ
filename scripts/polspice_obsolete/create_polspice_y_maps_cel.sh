#!/bin/bash

export HEALPIX=/home/ttroester/Codes/Healpix_3.40
export POLSPICE=/home/ttroester/Codes/PolSpice_v03-05-01/bin/spice
python KiDS1000_measurements.py \
--y-raw-map ../data/y_maps/ACT/BN.fits \
--y-masks ../data/y_maps/ACT/BN_mask_sharp.fits \
--y-map-path ../data/y_maps/polspice/ACT_BN \
#--dry-run \
