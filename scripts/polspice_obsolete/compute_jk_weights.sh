#!/usr/bin/bash

# python compute_jk_weights.py \
# --weight-paths ../data/shear_maps/z0.1-0.3/doublet_weight.fits ../data/y_maps/polspice/milca/singlet_mask.fits \
# --jk-def ../data/KiDS1000_jk_blocks.npz \
# --output ../results/measurements/shear_KiDS1000_y/z0.1-0.3-milca/jk_weights.npz

# python compute_jk_weights.py \
# --weight-paths ../data/shear_maps/z0.3-0.5/doublet_weight.fits ../data/y_maps/polspice/milca/singlet_mask.fits \
# --jk-def ../data/KiDS1000_jk_blocks.npz \
# --output ../results/measurements/shear_KiDS1000_y/z0.3-0.5-milca/jk_weights.npz

# python compute_jk_weights.py \
# --weight-paths ../data/shear_maps/z0.5-0.7/doublet_weight.fits ../data/y_maps/polspice/milca/singlet_mask.fits \
# --jk-def ../data/KiDS1000_jk_blocks.npz \
# --output ../results/measurements/shear_KiDS1000_y/z0.5-0.7-milca/jk_weights.npz

# python compute_jk_weights.py \
# --weight-paths ../data/shear_maps/z0.7-0.9/doublet_weight.fits ../data/y_maps/polspice/milca/singlet_mask.fits \
# --jk-def ../data/KiDS1000_jk_blocks.npz \
# --output ../results/measurements/shear_KiDS1000_y/z0.7-0.9-milca/jk_weights.npz

python compute_jk_weights.py \
--weight-paths ../data/shear_maps/z0.9-1.2/doublet_weight.fits ../data/y_maps/polspice/milca/singlet_mask.fits \
--jk-def ../data/KiDS1000_jk_blocks.npz \
--output ../results/measurements/shear_KiDS1000_y/z0.9-1.2-milca/jk_weights.npz

