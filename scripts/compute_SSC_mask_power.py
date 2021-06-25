import numpy as np

import healpy

import sys
sys.path.append("../tools/")
from misc_utils import file_header  # noqa: E402

from namaster_cosmic_shear_measurements import make_maps

if __name__ == "__main__":
    shear_mask_file = None
    shear_catalog_file = None

    # shear_name = "KiDS1000"
    # foreground_name = "Planck_gal40_ps"
    # shear_mask_file = "../data/shear_maps_KiDS1000/z0.1-1.2/n_gal.fits"
    # foreground_mask_file = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"     # noqa: E501

    # ACT
    shear_name = "KiDS1000_cel"
    foreground_name = "ACT_BN_Planck_gal40_ps"
    shear_catalog_file = "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-1.2.npz"  # noqa: E501
    foreground_mask_file = "../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits"

    if shear_catalog_file is not None:
        shear_data = np.load(shear_catalog_file)
        _, _, w_map = make_maps(2048,
                                shear_data["e1"],
                                shear_data["e2"],
                                shear_data["w"],
                                shear_data["pixel_idx"])
        shear_mask = w_map > 0
    else:
        shear_mask = healpy.read_map(shear_mask_file) > 0

    foreground_mask = healpy.read_map(foreground_mask_file) > 0

    ell = np.arange(3*2048)

    W_shear_shear_l = healpy.anafast(shear_mask, shear_mask)
    A_shear = shear_mask.sum()*healpy.nside2pixarea(2048)
    W_shear_shear_l = W_shear_shear_l * (2*ell + 1)/A_shear**2

    W_shear_foreground_l = healpy.anafast(shear_mask, foreground_mask)
    A_foreground = foreground_mask.sum()*healpy.nside2pixarea(2048)
    W_shear_foreground_l = (
        W_shear_foreground_l * (2*ell + 1)/(A_shear*A_foreground))

    W_shear_foreground_overlap_l = healpy.anafast(shear_mask*foreground_mask)
    A_shear_foreground_overlap = (
        (shear_mask*foreground_mask).sum()*healpy.nside2pixarea(2048))
    W_shear_foreground_overlap_l = (
        W_shear_foreground_overlap_l*(2*ell + 1)/A_shear_foreground_overlap**2)

    header = file_header(f"\\ell, W_l = \\sum_m W_lm W^*_lm\n"
                         f"A_{shear_name} = {A_shear} sr")
    np.savetxt(f"../data/xcorr/cov/W_l/shear_{shear_name}_binary_auto.txt",
               np.vstack((ell, W_shear_shear_l)).T,
               header=header)

    header = file_header(f"\\ell, W_l = \\sum_m W^A_lm W^B^*_lm\n"
                         f"A_{shear_name} = {A_shear} sr, "
                         f"A_{foreground_name} = {A_foreground} sr")
    np.savetxt(f"../data/xcorr/cov/W_l/"
               f"shear_{shear_name}_{foreground_name}_binary.txt",
               np.vstack((ell, W_shear_foreground_l)).T,
               header=header)

    header = file_header(f"\\ell, W_l = \\sum_m W_lm W^*_lm\n"
                         f"A_{shear_name}_{foreground_name}_overlap = "
                         f"{A_shear_foreground_overlap} sr")
    np.savetxt(f"../data/xcorr/cov/W_l/"
               f"shear_{shear_name}_{foreground_name}_overlap_binary_auto.txt",
               np.vstack((ell, W_shear_foreground_overlap_l)).T,
               header=header)
