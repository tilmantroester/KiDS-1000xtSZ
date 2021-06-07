#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
##SBATCH --mem-per-cpu 2000                         # In MB
#SBATCH --time 14-00:00                      # Time in days-hours:min
#SBATCH --job-name=nmt_cov_EETE               # this will be displayed if you write squeue in terminal and will be in the title of all emails slurm sends
#SBATCH --requeue                           # Allow requeing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ttr@roe.ac.uk
#SBATCH -o ../logs/slurm/log_nmt_cov_EETE.out
#SBATCH -e ../logs/slurm/log_nmt_cov_EETE.err
#SBATCH --partition=all

source ${HOME}/Codes/miniconda/bin/activate analysis

export OMP_NUM_THREADS=32
python namaster_covariance.py \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2.npz \
--foreground-mask ../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_cel_y_ACT_BN/ \
--n-iter 3 \
--nside 2048 \
--probes shear shear shear foreground \
# --dry-run \
#--exact-noise-noise \
#--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
