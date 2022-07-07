# Joint analysis of shear-tSZ cross-correlations and cosmic shear with KiDS-1000, Planck, and ACT

This repository contains the scripts, utilities, data vectors, and covariances used in the paper [Joint constraints on cosmology and the impact of baryon feedback: combining KiDS-1000 lensing with the thermal Sunyaev-Zeldovich effect from Planck and ACT](https://arxiv.org/abs/2109.04458). 

The main dependencies are
- [NaMaster](https://github.com/LSSTDESC/NaMaster), for the estimation of the pseudo-Cls and Gaussian covariances,
- [pyhmcode](https://github.com/tilmantroester/pyhmcode), for the non-linear power spectra models and halo profiles,
- [CCL](https://github.com/LSSTDESC/CCL), for the non-Gaussian covariance contributions,
- [kcap](https://github.com/KiDS-WL/kcap), for the inference pipeline.

## Repository structure
- `data/`: Contains catalogue statistics and data products with multiple applications (e.g., speficiation of the ell-binning).
- `notebooks/`: Collection of Jupyter notebooks used in data reduction and exploration.
- `results/`: Data vectors and covariances used in the analysis.
- `paper/`: Scripts to create the plots used in the paper. 
- `scripts/`: Collection of scripts used in the data reduction. The measurement "pipeline" is described in [`scripts/README.md`](scripts/README.md).
- `tools/`: Collection of tools used in the analysis. The distinction between `scripts` and `tools` is somewhat arbitrary.