#!/bin/bash

export OMP_NUM_THREADS=40
python namaster_shear_randoms_measurements.py \
--bin-operator ../data/xcorr/bin_operator_log_n_bin_13_ell_51-2952_namaster.txt \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
--pymaster-workspace /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_auto_namaster/pymaster_workspace_shear_1_shear_1.fits \
--n-iter 1 \
--n-randoms 100 \
--Cl-decoupled-filename ../results/measurements/shear_KiDS1000_true_randoms_namaster/run_0/Cl_decoupled_z0.9-1.2_gal \
--Cl-coupled-filename ../results/measurements/shear_KiDS1000_true_randoms_namaster/run_0/Cl_coupled_z0.9-1.2_gal \
