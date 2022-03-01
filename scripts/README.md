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
```
./shear_covariance_mcm_namaster.sh
```

## 5. Compute Gaussian covariance matrices
```
python make_all_shear_gaussian_covariance_matrices.py
```

## 6. Assemble data and covariances into single files for likelihood
```
python assemble_data_and_covariance_files.py
```

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
