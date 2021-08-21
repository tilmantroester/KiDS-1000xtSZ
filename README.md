# Make cosmic shear measurements

## Create catalogs and catalog statistics
```
python compute_KiDS_stats.py
```
Creates separate catalogs for north, south, and the combined patches, in both celestial and galactic coordinates.
The celestial catalogs need e1 flipped when used as a map.

## Create ell bins:
```
python create_binning_operator.py
```

## Compute mode-mixing matrices and measure Cls
```
python make_all_shear_shear_measurements.py
```

## Compute covariance mode-mixing matrices
```
./shear_covariance_mcm_namaster.sh
```

## Compute Gaussian covariance matrices
```
python make_all_shear_gaussian_covariance_matrices.py
```

## Assemble data and covariances into single files for likelihood
```
python assemble_data_and_covariance_files.py
```

# Make cross-correlation and joint analysis measurements

## Compute mode-mixing matrices and measure Cls
```
python make_all_shear_tSZ_measurements.py
```

## Compute covariance mode-mixing matrices
Run for both TETE and EETE if joint analysis with cosmic shear is required.
```
./namaster_covariance_mcm.sh
```

## Compute auto-spectrum of foreground map for covariance
```
python compute_foreground_cov_Cls.py
```

## Compute Gaussian covariance matrices
```
python make_all_shear_tSZ_gaussian_covariance_matrices.py
```

## Assemble data and covariances into single files for likelihood
```
python assemble_joint_data_and_covariance_files.py
```

## Compute CIB-shear cross-correlation and covariance
```
python reduce_CIB_data.py
```

## Add up covariance contributions
```
python add_up_covariance_terms.py
```