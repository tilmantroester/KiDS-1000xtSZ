#!/bin/bash

export OMP_NUM_THREADS=40
python namaster_measurement.py \
--bin-operator delta_ell_10 \
--foreground-map ../data/y_maps/ACT/BN.fits \
--foreground-mask ../data/y_maps/ACT/BN_mask.fits \
--foreground-mask-already-applied \
--foreground-auto \
--output-path=../results/measurements/y_y_ACT_BN_namaster/ \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/y_y_ACT_BN_namaster_delta_ell_10/ \

# --pymaster-workspace-input-path /disk09/ttroester/project_triad/namaster_workspaces/y_y_ACT_BN_namaster/ \

# --compute-covariance \
# --shear-maps ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/triplet.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/triplet.fits \
# --shear-masks ../data/shear_maps_KiDS1000_cel_N/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.3-0.5/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.5-0.7/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.7-0.9/doublet_weight.fits ../data/shear_maps_KiDS1000_cel_N/z0.9-1.2/doublet_weight.fits \