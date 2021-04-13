#!/bin/bash

export HEALPIX=/home/ttroester/Codes/Healpix_3.40
export POLSPICE=/home/ttroester/Codes/PolSpice_v03-05-01/bin/spice
python polspice_xcorr.py \
--machine=cuillin \
--n-slot=5 \
--n-thread=12 \
--partition=all \
--walltime=24:00:00 \
--shear-paths ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3 ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5 ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7 ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9 ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2 \
--probe-paths ../data/y_maps/polspice/ACT_BN/ \
--output-path=../results/measurements/shear_KiDS1000_y_ACT_BN_thetamax120/ \
--thetamax 120.0

# --shear-paths ../data/shear_maps_KiDS1000/z0.1-0.3 ../data/shear_maps_KiDS1000/z0.3-0.5 ../data/shear_maps_KiDS1000/z0.5-0.7 ../data/shear_maps_KiDS1000/z0.7-0.9 ../data/shear_maps_KiDS1000/z0.9-1.2 \
