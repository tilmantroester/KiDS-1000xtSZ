#!/bin/bash

export HEALPIX=/home/ttroester/Codes/Healpix_3.40
export POLSPICE=/home/ttroester/Codes/PolSpice_v03-05-01/bin/spice
python polspice_xcorr.py \
--machine=cuillin \
--n-slot=12 \
--n-thread=12 \
--partition=WL \
--walltime=24:00:00 \
--shear-paths ../data/shear_maps_KiDS1000/z0.1-0.3 ../data/shear_maps_KiDS1000/z0.3-0.5 ../data/shear_maps_KiDS1000/z0.5-0.7 ../data/shear_maps_KiDS1000/z0.7-0.9 ../data/shear_maps_KiDS1000/z0.9-1.2 \
--probe-paths /disk09/ttroester/Planck/CIB/polspice/2048/galactic/Planck-353 /disk09/ttroester/Planck/CIB/polspice/2048/galactic/Planck-545 /disk09/ttroester/Planck/CIB/polspice/2048/galactic/Planck-857 \
--output-path=../results/measurements/shear_KiDS1000_CIB \
--jackknife-block-file=../data/KiDS1000_jk_blocks.npz \
--jackknife-block-key=64 \
--only-jackknife