#!/bin/bash

export HEALPIX=/home/ttroester/Codes/Healpix_3.40
export POLSPICE=/home/ttroester/Codes/PolSpice_v03-05-01/bin/spice
python KiDS1000_measurements.py \
--shear-catalogs /disk09/KIDS/KIDSCOLLAB_V1.0.0/WL_gold_cat_release_DR4.1/KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits \
--shear-catalog-hdu=1 \
--z-min=0.1 --z-max=0.3 \
--shear-map-path ../data/shear_maps_KiDS1000_cel/z0.1-0.3/ \
#--only-north \
#--dry-run \
