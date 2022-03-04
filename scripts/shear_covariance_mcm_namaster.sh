#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 32
##SBATCH --mem-per-cpu 2000                         # In MB
#SBATCH --time 14-00:00                      # Time in days-hours:min
#SBATCH --job-name=nmt_cov               # this will be displayed if you write squeue in terminal and will be in the title of all emails slurm sends
#SBATCH --requeue                           # Allow requeing
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ttr@roe.ac.uk
#SBATCH -o ../logs/slurm/log_nmt_cov.out
#SBATCH -e ../logs/slurm/log_nmt_cov.err
#SBATCH --partition=all

source ${HOME}/Codes/miniconda/bin/activate analysis

export OMP_NUM_THREADS=32
python namaster_covariance.py \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_shear_KiDS1000/ \
--n-iter 3 \
--nside 2048 \
--signal-signal \
# --dry-run

python namaster_covariance.py \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_shear_KiDS1000/ \
--n-iter 3 \
--nside 2048 \
--exact-noise-noise \
# --dry-run

python namaster_covariance.py \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_shear_KiDS1000/ \
--n-iter 3 \
--nside 2048 \
--exact-noise-signal \
# --dry-run
