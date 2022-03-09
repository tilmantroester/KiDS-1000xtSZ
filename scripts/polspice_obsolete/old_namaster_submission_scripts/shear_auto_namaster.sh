#!/bin/bash

export OMP_NUM_THREADS=40
python namaster_measurement.py \
--bin-operator ../data/xcorr/bin_operator_log_n_bin_13_ell_51-2952_namaster.txt \
--shear-maps ../data/shear_maps_KiDS1000/z0.1-0.3/triplet.fits ../data/shear_maps_KiDS1000/z0.9-1.2/triplet.fits \
--shear-masks ../data/shear_maps_KiDS1000/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000/z0.9-1.2/doublet_weight.fits \
--shear-auto \
--output-path=../results/measurements/shear_KiDS1000_auto_namaster/ \
--pymaster-workspace-input-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_auto_namaster/ \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_auto_namaster/ \
--compute-covariance \

# --pymaster-workspace-input-path /disk09/ttroester/project_triad/namaster_workspaces/y_y_ACT_BN_namaster/ \

# --compute-covariance \
# --shear-maps ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/triplet.fits \
# --shear-masks ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/doublet_weight.fits \