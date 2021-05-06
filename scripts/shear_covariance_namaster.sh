#!/bin/bash

export OMP_NUM_THREADS=40
python namaster_covariance.py \
--shear-w-maps ../data/shear_maps_KiDS1000/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000/z0.9-1.2/doublet_weight.fits \
--shear-w2e2-maps ../data/shear_maps_KiDS1000/z0.1-0.3/w_sq_e_sq.fits ../data/shear_maps_KiDS1000/z0.9-1.2/w_sq_e_sq.fits \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_auto_namaster/ \
--n-iter 3
