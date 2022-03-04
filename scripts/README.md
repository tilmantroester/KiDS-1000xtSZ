# Make cosmic shear measurements

This is a summary in how the NaMaster measurement and covariance estimates were computed. The scripts listed here are the main scripts used for the measurements in the paper. The remaining scripts in the directory are obsolete and should be cleaned up.

## 1. Create catalogs and catalog statistics
```
python compute_KiDS_stats.py
```
Creates separate catalogs for north, south, and the combined patches, in both celestial and galactic coordinates.
The celestial catalogs need e1 flipped when used as a map.

## 2. Create ell bins:
```
python create_binning_operator.py
```

## 3. Compute mode-mixing matrices and measure Cls
```
python make_all_shear_shear_measurements.py
```

## 4. Compute covariance mode-mixing matrices
The `namaster_covariance.py` script computes the covariance mode-mixing matrices. This takes a while. For the signal-signal term, the call is
```
python namaster_covariance.py \
--shear-catalogs ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz ../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz \
--pymaster-workspace-output-path /disk09/ttroester/project_triad/namaster_workspaces/shear_KiDS1000_shear_KiDS1000/ \
--n-iter 3 \
--signal-signal
```
For the noise-noise, and noise-signal terms, change the last option to `--exact-noise-noise` and `--exact-noise-signal`, respectively. An example slurm submission script can be found in
```
./shear_covariance_mcm_namaster.sh
```


## 5. Compute Gaussian covariance matrices
The covariance matrices are computed in `make_all_shear_gaussian_covariance_matrices.py`. Change the `mode` variable to the set of terms that should be computed.

## 6. Assemble data and covariances into single files for likelihood
`assemble_cosmic_shear_data_and_covariance_files.py` assembles the `txt` files of the covariance terms. Uncomment the key/tag pair at the top of the script for the term that should be assembled. The terms still need to be added up, which is done with `add_up_covariance_terms.py`. This is currently targeted at the joint shear-tSZ analysis and might not work out of the box for a cosmic shear-only analyses.

# Make cross-correlation and joint analysis measurements

## 1. Compute mode-mixing matrices and measure Cls
```
python make_all_shear_tSZ_measurements.py
```

## 2. Compute covariance mode-mixing matrices
Run for both TETE and EETE if joint analysis with cosmic shear is required.
```
./namaster_covariance_mcm.sh
```

## 3. Compute auto-spectrum of foreground map for covariance
```
python compute_foreground_cov_Cls.py
```

## 4. Compute Gaussian covariance matrices
```
python make_all_shear_tSZ_gaussian_covariance_matrices.py
```

## 5. Assemble data and covariances into single files for likelihood
```
python assemble_joint_data_and_covariance_files.py
```

## 6. Add up covariance contributions
```
python add_up_covariance_terms.py
```

## 7. Fit GP to CIB cross-correlation
```
python reduce_CIB_data.py
```
