#!/bin/bash

export OMP_NUM_THREADS=40
python namaster_measurement.py \
--bin-operator ../data/xcorr/bin_operator_log_n_bin_13_ell_51-2952_namaster.txt \
--shear-maps ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/triplet.fits \
--shear-masks ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/doublet_weight.fits \
--foreground-map ../data/y_maps/ACT/BN.fits \
--foreground-mask ../data/y_maps/ACT/BN_mask.fits \
--foreground-mask-already-applied \
--output-path=../results/measurements/shear_KiDS1000_y_ACT_BN_namaster/ \
# --pymaster-workspace-path ../results/measurements/shear_KiDS1000_y_ACT_BN_namaster/ \
# --shear-maps ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/triplet.fits \
# --shear-masks ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/doublet_weight.fits \