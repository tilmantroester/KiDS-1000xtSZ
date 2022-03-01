import os

import numpy as np
import astropy.io.fits as fits

import healpy

import sys
sys.path.append("../tools/")
from misc_utils import file_header, read_partial_map


def transform_cel_to_gal(ra, dec, e1, e2):
    r = healpy.Rotator(coord=["C", "G"])
    l, b = r(ra, dec, lonlat=True)
    alpha = r.angle_ref(ra, dec, lonlat=True)
    e = np.sqrt(e1**2 + e2**2)
    # Need to switch sign of e1 here
    phi = np.arctan2(e2, -e1)/2
    e1_gal = e*np.cos(2*(phi+alpha))
    e2_gal = e*np.sin(2*(phi+alpha))

    return l, b, e1_gal, e2_gal


def prepare_catalog(catalog_filename,
                    column_names={"x": "x", "y": "y",
                                  "e1": "e1", "e2": "e2",
                                  "w": "w", "m": "m",
                                  "c1": "c1", "c2": "c2"},
                    c_correction="data", m_correction="catalog",
                    z_min=None, z_max=None,
                    selections=[("weight", "gt", 0.0)],
                    hdu_idx=1, verbose=False):
    """Prepare lensing catalogues.

    Arguments:
        catalog_filename (str): File name of shape catalog.
        column_names (dict, optional): Dictionary of column names. Required
            entries are "x", "y", "e1", "e2", "w". If c_correction="catalog",
            "c1", "c2" are required. If m_correction="catalog", "m" is
            required. (default ``{"x":"x", "y":"y", "e1":"e1", "e2":"e2",
            "w":"w", "m":"m", "c1":"c1", "c2":"c2"}``).
        c_correction (str, optional): Apply c correction. Options are "catalog"
            and "data". (default: "data").
        m_correction (str, optional): Apply m correction. Options are "catalog"
            or ``None``. (default: "catalog").
        z_min (float, optional): Lower bound for redshift cut (default: None).
        z_max (float, optional): Upper bound for redshift cut (default: None).
        selections (list, optional): List of catalog selection criteria. The
            list entries consist of tuples with three elements: column name,
            operator, and value. The supported operators are "eq", "neq", "gt",
            "ge", "lt", "le". (default ``[("weight", "gt", 0.0)]``).
        hdu_idx (int, optional): Index of the FITS file HDU to use.
            (default 1).
        verbose (bool, optional): Verbose output (default False).

    Returns:
        (tuple): Tuple containing:
            (numpy.array): RA
            (numpy.array): Dec
            (numpy.array): e1
            (numpy.array): e2
            (numpy.array): w
            (numpy.array): m
    """
    hdu = fits.open(catalog_filename)

    mask = np.ones(hdu[hdu_idx].data.size, dtype=bool)
    # Apply selections to catalog
    for col, op, val in selections:
        if verbose:
            print("Applying {} {} on {}.".format(op, val, col))
        if op == "eq":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] == val)
        elif op == "neq":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] != val)
        elif op == "gt":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] > val)
        elif op == "ge":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] >= val)
        elif op == "lt":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] < val)
        elif op == "le":
            mask = np.logical_and(mask, hdu[hdu_idx].data[col] <= val)
        else:
            raise ValueError("Operator {} not supported.".format(op))

    # For convenience, redshfit cuts can be applied through the arguments
    # z_min and z_max as well.
    if z_min is not None and z_max is not None:
        if verbose:
            print("Applying z cut: {}-{}.".format(z_min, z_max))
        z = hdu[hdu_idx].data[column_names["z"]]
        mask = np.logical_and(mask, np.logical_and(z > z_min, z <= z_max))

    ra = hdu[hdu_idx].data[column_names["x"]][mask]
    dec = hdu[hdu_idx].data[column_names["y"]][mask]
    w = hdu[hdu_idx].data[column_names["w"]][mask]
    e1 = hdu[hdu_idx].data[column_names["e1"]][mask]
    e2 = hdu[hdu_idx].data[column_names["e2"]][mask]

    # Apply c correction
    if c_correction == "catalog":
        # Use c correction supplied by the catalog
        if verbose:
            print("Applying c correction provided by the catalog.")
        c1 = hdu[hdu_idx].data[column_names["c1"]][mask]
        c2 = hdu[hdu_idx].data[column_names["c2"]][mask]
        c1_mask = c1 > -99
        c2_mask = c2 > -99
        e1[c1_mask] -= c1[c1_mask]
        e2[c2_mask] -= c2[c2_mask]
    elif c_correction == "data":
        # Calculate c correction from the weighted ellipticity average
        if verbose:
            print("Applying c correction calculated from the data.")
        c1 = np.sum(w*e1)/np.sum(w)
        c2 = np.sum(w*e2)/np.sum(w)
        e1 -= c1
        e2 -= c2

    # Apply m correction
    if m_correction == "catalog":
        if verbose:
            print("Applying m correction provided by the catalog.")
        m = hdu[hdu_idx].data[column_names["m"]][mask]
    else:
        m = np.zeros_like(w)

    hdu.close()

    return ra, dec, e1, e2, w, m


if __name__ == "__main__":
    Z_CUTS = [(0.1, 0.3),
              (0.3, 0.5),
              (0.5, 0.7),
              (0.7, 0.9),
              (0.9, 1.2),
              (0.1, 1.2)]

    catalog_stats = True
    map_stats = False

    create_compressed_catalogs = True
    convert_to_gal = True

    compressed_catalog_path = "../data/shear_catalogs_KiDS1000/"
    os.makedirs(compressed_catalog_path, exist_ok=True)

    nside = 2048

    if catalog_stats:
        patches = ["All", "N", "S"]

        print("Catalog-based stats")

        catalog = "/disk09/KIDS/KIDSCOLLAB_V1.0.0/WL_gold_cat_release_DR4.1/"\
                  "KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits"
        catalog_hdu = 1

        KiDS_column_names = {"x":  "ALPHA_J2000",
                             "y":  "DELTA_J2000",
                             "e1": "e1",
                             "e2": "e2",
                             "w":  "weight",
                             "z":  "Z_B"}

        for patch in patches:
            print("Patch: ", patch)

            KiDS_selection = [("weight", "gt", 0.0)]

            if patch == "N":
                KiDS_selection += [("DELTA_J2000", "gt", -20.0)]
            elif patch == "S":
                KiDS_selection += [("DELTA_J2000", "lt", -20.0)]

            sum_w = {}
            sum_w_sq = {}
            sum_w_sq_e1_sq = {}
            sum_w_sq_e2_sq = {}

            sigma_e_sq = {}

            for z_cut in Z_CUTS:
                ra, dec, e1, e2, w, m = prepare_catalog(
                                            catalog_filename=catalog,
                                            column_names=KiDS_column_names,
                                            c_correction="data",
                                            m_correction=None,
                                            z_min=z_cut[0], z_max=z_cut[1],
                                            selections=KiDS_selection,
                                            hdu_idx=catalog_hdu,
                                            verbose=True)

                sum_w[z_cut] = np.sum(w)
                sum_w_sq[z_cut] = np.sum(w**2)

                sum_w_sq_e1_sq[z_cut] = np.sum(w**2 * e1**2)
                sum_w_sq_e2_sq[z_cut] = np.sum(w**2 * e2**2)

                if create_compressed_catalogs:
                    catalog_name = f"KiDS-1000_{patch}_z{z_cut[0]}-{z_cut[1]}"
                    if convert_to_gal:
                        l, b, e1_gal, e2_gal = transform_cel_to_gal(
                                                        ra, dec, e1, e2)
                        pixel_idx_gal = healpy.ang2pix(nside,
                                                       l, b, lonlat=True)
                        filename = os.path.join(compressed_catalog_path,
                                                catalog_name+"_galactic")
                        np.savez(filename,
                                 l=l, b=b, w=w, e1=e1_gal, e2=e2_gal,
                                 pixel_idx=pixel_idx_gal)

                    pixel_idx = healpy.ang2pix(nside,
                                               ra, dec, lonlat=True)
                    filename = os.path.join(compressed_catalog_path,
                                            catalog_name)
                    np.savez(filename, ra=ra, dec=dec, w=w, e1=e1, e2=e2,
                             pixel_idx=pixel_idx)

            if patch == "N":
                area_file = "../data/shear_stats/K1000_N_eff_area.txt"
            elif patch == "S":
                area_file = "../data/shear_stats/K1000_S_eff_area.txt"
            else:
                area_file = "../data/shear_stats/K1000_eff_area.txt"

            A = np.loadtxt(area_file,
                           converters={0: lambda s: 0.0})[:, -1].sum()

            sigma_e = {}
            n_eff = {}

            for z_cut in Z_CUTS:
                sigma_e[z_cut] = (np.sqrt(0.5*(sum_w_sq_e1_sq[z_cut]
                                  + sum_w_sq_e2_sq[z_cut])/sum_w_sq[z_cut]))
                n_eff[z_cut] = sum_w[z_cut]**2/sum_w_sq[z_cut]/A

            header = file_header(
                        "sigma_e "
                        "(sqrt(0.5(\\sum w^2 e1^2 "
                        "+ \\sum w^2 e2^2)/\\sum w^2))\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")
            np.savetxt(f"../data/shear_stats/sigma_e_{patch}_catalog.txt",
                       np.array(list(sigma_e.values())), header=header)

            header = file_header(
                        "n_eff (1/A (\\sum w)^2/\\sum w^2)\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")

            np.savetxt(f"../data/shear_stats/n_eff_{patch}_catalog.txt",
                       np.array(list(n_eff.values())), header=header)

            header = file_header("Area [deg^2]")

            np.savetxt(f"../data/shear_stats/area_{patch}_catalog.txt",
                       np.array([A/60**2]), header=header)

            data = np.vstack((np.array(list(sum_w.values())),
                              np.array(list(sum_w_sq.values())),
                              np.array(list(sum_w_sq_e1_sq.values())),
                              np.array(list(sum_w_sq_e2_sq.values())))).T

            header = file_header(
                        "\\sum w, \\sum w^2, "
                        "\\sum w^2 e1^2, \\sum w^2 e2^2\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")

            np.savetxt(f"../data/shear_stats/raw_stats_{patch}_catalog.txt",
                       data, header=header)

    if map_stats:
        print("Map-based stats")

        maps = [("All_gal", ""),
                ("N_cel", "_cel_N")]

        for map_name, suffix in maps:
            print("Map suffix: ", map_name)
            A_KiDS = {}

            sum_w_map = {}
            sum_w_sq_map = {}
            sum_e1_sq_map = {}
            sum_e2_sq_map = {}
            sum_w_e1_sq_map = {}
            sum_w_e2_sq_map = {}
            sum_w_sq_e1_sq_map = {}
            sum_w_sq_e2_sq_map = {}
            var_e1 = {}
            var_e2 = {}
            var_w_e1 = {}
            var_w_e2 = {}
            n_pix = {}
            sigma_e_map = {}
            n_eff_map = {}

            for z_cut in Z_CUTS:
                shear_mask = healpy.read_map(
                                f"../data/shear_maps_KiDS1000{suffix}/"
                                f"z{z_cut[0]}-{z_cut[1]}/doublet_mask.fits",
                                verbose=False)
                shear_mask[shear_mask == healpy.UNSEEN] = 0

                shear_weight = healpy.read_map(
                                f"../data/shear_maps_KiDS1000{suffix}/"
                                f"z{z_cut[0]}-{z_cut[1]}/doublet_weight.fits",
                                verbose=False)
                shear_weight[shear_weight == healpy.UNSEEN] = 0

                e1, e2 = read_partial_map(
                                f"../data/shear_maps_KiDS1000{suffix}/"
                                f"z{z_cut[0]}-{z_cut[1]}/triplet.fits",
                                fields=[2, 3], fill_value=0,
                                scale=[1, 1])

                sum_w_map[z_cut] = np.sum(shear_weight)
                sum_w_sq_map[z_cut] = np.sum(shear_weight**2)
                sum_e1_sq_map[z_cut] = np.sum(e1**2)
                sum_e2_sq_map[z_cut] = np.sum(e2**2)
                sum_w_e1_sq_map[z_cut] = np.sum(shear_weight * e1**2)
                sum_w_e2_sq_map[z_cut] = np.sum(shear_weight * e2**2)
                sum_w_sq_e1_sq_map[z_cut] = np.sum(shear_weight**2 * e1**2)
                sum_w_sq_e2_sq_map[z_cut] = np.sum(shear_weight**2 * e2**2)

                var_e1[z_cut] = np.var(e1[shear_weight > 0], ddof=1)
                var_e2[z_cut] = np.var(e2[shear_weight > 0], ddof=1)
                var_w_e1[z_cut] = np.cov(e1, aweights=shear_weight)
                var_w_e2[z_cut] = np.cov(e2, aweights=shear_weight)

                n_pix[z_cut] = shear_mask.sum()

                A_KiDS[z_cut] = (shear_mask.sum()
                                 * healpy.nside2pixarea(2048, degrees=True))

                sigma_e_map[z_cut] = \
                    (np.sqrt(0.5*(sum_w_sq_e1_sq_map[z_cut]
                     + sum_w_sq_e2_sq_map[z_cut])/sum_w_sq_map[z_cut]))
                n_eff_map[z_cut] = (sum_w_map[z_cut]**2/sum_w_sq_map[z_cut]
                                    * 1/(A_KiDS[z_cut]*60**2))

            header = file_header(
                        "sigma_e "
                        "(sqrt(0.5(\\sum w^2 e1^2 "
                        "+ \\sum w^2 e2^2)/\\sum w^2))\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")
            np.savetxt(f"../data/shear_stats/sigma_e_{map_name}_map.txt",
                       np.array(list(sigma_e_map.values())), header=header)

            header = file_header(
                        "n_eff (1/A (\\sum w)^2/\\sum w^2)\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")

            np.savetxt(f"../data/shear_stats/n_eff_{map_name}_map.txt",
                       np.array(list(n_eff_map.values())), header=header)

            header = file_header(
                        "Area [deg^2]\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")

            np.savetxt(f"../data/shear_stats/area_{map_name}_map.txt",
                       np.array(list(A_KiDS.values())), header=header)

            data = np.vstack((np.array(list(sum_w_map.values())),
                              np.array(list(sum_w_sq_map.values())),
                              np.array(list(sum_e1_sq_map.values())),
                              np.array(list(sum_e2_sq_map.values())),
                              np.array(list(sum_w_e1_sq_map.values())),
                              np.array(list(sum_w_e2_sq_map.values())),
                              np.array(list(sum_w_sq_e1_sq_map.values())),
                              np.array(list(sum_w_sq_e2_sq_map.values())),
                              np.array(list(var_e1.values())),
                              np.array(list(var_e2.values())),
                              np.array(list(var_w_e1.values())),
                              np.array(list(var_w_e2.values())),
                              np.array(list(n_pix.values())))).T
            header = file_header(
                        "\\sum w, \\sum w^2, "
                        "\\sum e1^2, \\sum e2^2, \\sum w e1^2, \\sum w e2^2, "
                        "\\sum w^2 e1^2, \\sum w^2 e2^2, "
                        "Var[e1], Var[e2], "
                        "weighted Var[e1], weighted Var[e2], n_pix\n"
                        f"z-bins: {', '.join([str(z) for z in Z_CUTS])}")

            np.savetxt(f"../data/shear_stats/raw_stats_{map_name}_map.txt",
                       data, header=header)
