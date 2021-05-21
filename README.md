# Make measurements
## Cosmic shear

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
python make_all_shear_gaussian_covariance_matrices
```

## Assemble data and covariances into single files for likelihood
```
python assemble_data_and_covariance_files.py
```