#!/bin/bash

export OMP_NUM_THREADS=44
for i in $(seq 100 400);
do
    echo
    echo "Iteration ${i}"
    echo
    python namaster_measurement.py \
    --bin-operator ../data/xcorr/bin_operator_log_n_bin_13_ell_51-2952_namaster.txt \
    --shear-maps ../data/shear_maps_KiDS1000/z0.1-0.3/triplet.fits ../data/shear_maps_KiDS1000/z0.9-1.2/triplet.fits \
    --shear-masks ../data/shear_maps_KiDS1000/z0.1-0.3/doublet_weight.fits ../data/shear_maps_KiDS1000/z0.9-1.2/doublet_weight.fits \
    --shear-auto \
    --output-path=../results/measurements/shear_KiDS1000_auto_randoms_namaster/run_${i}/ \
    --pymaster-workspace-input-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_auto_namaster/ \
    --randomize-shear \
    --n-iter 1 \

done