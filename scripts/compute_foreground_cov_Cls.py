import os
import argparse

import pymaster as nmt
import healpy

import numpy as np
import scipy.interpolate

if __name__ == "__main__":
    nside = 2048

    make_plots = False

    binning_operator_file = "../data/xcorr/bin_operator_log_n_bin_12_ell_51-2952_namaster.txt"                     # noqa: E501

    beam = None

    # Planck milca
    # foreground_map_file = "../data/y_maps/polspice/milca/triplet.fits"
    # foreground_mask_file = "../data/y_maps/polspice/milca/singlet_mask.fits"
    # beam = "../data/xcorr/beams/beam_Planck.txt"
    # cov_Cls_file = "../results/measurements/y_milca_y_milca/cov_Cls/Cl_cov_beam_deconv_raw_0-0.npz"                # noqa: E501
    # cov_Cls_smoothed_file = "../results/measurements/y_milca_y_milca/cov_Cls/Cl_cov_beam_deconv_smoothed_0-0.npz"  # noqa: E501
    # Planck NILC
    # foreground_map_file = "../data/y_maps/Planck_processed/nilc_full.fits"
    # foreground_mask_file = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # cov_Cls_file = "../results/measurements/y_nilc_y_nilc/cov_Cls/Cl_cov_beam_deconv_raw_0-0.npz"                # noqa: E501
    # cov_Cls_smoothed_file = "../results/measurements/y_nilc_y_nilc/cov_Cls/Cl_cov_beam_deconv_smoothed_0-0.npz"  # noqa: E501
    # Ziang nocib
    # foreground_map_file = "../data/y_maps/Planck_processed/ziang/ymap_rawcov_needlet_galmasked_v1.02_bp.fits"      # noqa: E501
    # foreground_mask_file = "../data/y_maps/Planck_processed/ziang/ycibmask_G.fits"                                 # noqa: E501
    # ACT BN
    # foreground_map_file = "../data/y_maps/ACT/BN.fits"
    # foreground_mask_file = "../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits"
    # ACT BN nocib
    # foreground_map_file = "../data/y_maps/ACT/BN_deproject_cib.fits"
    # foreground_mask_file = "../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits"

    # Planck 100 GHz HFI
    # foreground_map_file = "/disk09/ttroester/Planck/frequency_maps/HFI_SkyMap_100_2048_R3.01_full.fits"
    # foreground_mask_file = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # cov_Cls_file = "../results/measurements/100GHz_HFI_100GHz_HFI/cov_Cls/Cl_cov_raw_0-0.npz"                # noqa: E501
    # cov_Cls_smoothed_file = "../results/measurements/100GHz_HFI_100GHz_HFI/cov_Cls/Cl_cov_smoothed_0-0.npz"  # noqa: E501
    # Planck 545 GHz CIB
    foreground_map_file = "../data/CIB_maps/CIB-GNILC-F545_beam10.fits"
    foreground_mask_file = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    cov_Cls_file = "../results/measurements/545GHz_CIB_545GHz_CIB/cov_Cls/Cl_cov_raw_0-0.npz"                # noqa: E501
    cov_Cls_smoothed_file = "../results/measurements/545GHz_CIB_545GHz_CIB/cov_Cls/Cl_cov_smoothed_0-0.npz"  # noqa: E501

    os.makedirs(os.path.split(cov_Cls_file)[0], exist_ok=True)

    binning_operator = np.loadtxt(binning_operator_file)
    ell = np.arange(binning_operator.size)

    if beam is not None:
        beam = np.loadtxt(beam)

    nmt_bins = nmt.NmtBin(nside=nside, bpws=binning_operator,
                          ells=ell, weights=2*ell+1)
    ell = np.arange(3*nside)

    y_map = healpy.read_map(foreground_map_file)
    y_mask = healpy.read_map(foreground_mask_file)

    y_map[y_map == healpy.UNSEEN] = 0
    y_mask[y_mask == healpy.UNSEEN] = 0

    print("Creating field")
    field_y = nmt.NmtField(y_mask, [y_map], beam=beam, spin=0)

    mean_w2 = (y_mask**2).mean()

    print("Computing coupling matrices")
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(field_y, field_y, nmt_bins)

    print("Computing Cls")
    Cl_y_y_coupled = nmt.compute_coupled_cell(field_y, field_y)
    Cl_y_y_decoupled = wsp.decouple_cell(Cl_y_y_coupled)

    intp = scipy.interpolate.UnivariateSpline(
                np.log(ell[1:]),
                np.log(Cl_y_y_coupled[0][1:]/mean_w2),
                s=10)

    ell_bin = nmt_bins.get_effective_ells()

    Cl_cov = Cl_y_y_coupled[0]/mean_w2
    # We include the noise in the signal here
    Cl_noise_cov = np.zeros_like(Cl_cov)

    np.savez(cov_Cls_file,
             ell=ell,
             Cl_cov=[Cl_cov], Cl_noise_cov=[Cl_noise_cov])

    Cl_cov[1:] = np.exp(intp(np.log(ell[1:])))
    np.savez(cov_Cls_smoothed_file,
             ell=ell,
             Cl_cov=[Cl_cov], Cl_noise_cov=[Cl_noise_cov])

    if make_plots:
        import matplotlib.pyplot as plt
        plt.loglog(ell_bin, ell_bin**2*Cl_y_y_decoupled[0])
        plt.loglog(ell, ell**2*Cl_y_y_coupled[0]/mean_w2, alpha=0.3)
        plt.loglog(ell[1:], ell[1:]**2*np.exp(intp(np.log(ell[1:]))),
                   alpha=1.0)
