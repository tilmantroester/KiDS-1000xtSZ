#!/bin/bash

export OMP_NUM_THREADS=20
python namaster_shear_randoms_measurements.py \
--bin-operator ../data/xcorr/bin_operator_log_n_bin_13_ell_51-2952_namaster.txt \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
--pymaster-workspace /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_shear_KiDS1000_binary_mask/pymaster_workspace_shear_4_shear_4.fits \
--n-iter 0 \
--n-randoms 100 \
--Cl-decoupled-filename ../results/measurements/shear_KiDS1000_true_randoms_with_signal_binary_mask/run_0/Cl_decoupled_z0.9-1.2_gal \
--Cl-coupled-filename ../results/measurements/shear_KiDS1000_true_randoms_with_signal_binary_mask/run_0/Cl_coupled_z0.9-1.2_gal \
--Cl-decoupled-no-noise-bias-filename ../results/measurements/shear_KiDS1000_true_randoms_with_signal_binary_mask/run_0/Cl_decoupled_no_noise_bias_z0.9-1.2_gal \
--nofz-files ../runs/base_setup/data/load_source_nz/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.asc \
--bandpower-windows-filename ../results/measurements/shear_KiDS1000_true_randoms_with_signal_binary_mask/pymaster_bandpower_windows_shear_4_shear_4.npy \
--compute-coupling-matrix \
# --shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
# --pymaster-workspace /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_auto_namaster/pymaster_workspace_shear_1_shear_0.fits \
# --n-iter 1 \
# --n-randoms 100 \
# --Cl-decoupled-filename ../results/measurements/shear_KiDS1000_true_randoms_namaster/run_0/Cl_decoupled_z0.1-0.3_z0.9-1.2_gal \
# --Cl-coupled-filename ../results/measurements/shear_KiDS1000_true_randoms_namaster/run_0/Cl_coupled_z0.1-0.3_z0.9-1.2_gal \
