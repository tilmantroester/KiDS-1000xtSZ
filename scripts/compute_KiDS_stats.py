import os

import numpy as np

import healpy

import pylenspice.pylenspice as pylenspice

import sys
sys.path.append("../tools/")
from misc_utils import file_header, read_partial_map


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
        patches = ["N", "S", "All"]

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
                ra, dec, e1, e2, w, m = pylenspice.prepare_catalog(
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
                        l, b, e1_gal, e2_gal = \
                            pylenspice.convert_shear_to_galactic_coordinates(
                                ra, dec, e1, e2)
                        pixel_idx_gal = healpy.ang2pix(nside,
                                                       -b/180*np.pi+np.pi/2,
                                                       l/180*np.pi)
                        filename = os.path.join(compressed_catalog_path,
                                                catalog_name+"_galactic")
                        np.savez(filename,
                                 l=l, b=b, w=w, e1=e1_gal, e2=e2_gal,
                                 pixel_idx=pixel_idx_gal)
                    
                    pixel_idx = healpy.ang2pix(nside,
                                               -dec/180*np.pi+np.pi/2,
                                               ra/180*np.pi)
                    filename = os.path.join(compressed_catalog_path,
                                            catalog_name)
                    np.savez(filename, ra=ra, dec=dec, w=w, e1=e1, e2=e2,
                             pixel_idx=pixel_idx)

            if patch == "N":
                area_file = "/home/cech/KiDSLenS/THELI_catalogues/KIDS_conf/"\
                            "EFFECTIVE_AREA_NEFF/K1000_N_eff_area.txt"
            elif patch == "S":
                area_file = "/home/cech/KiDSLenS/THELI_catalogues/KIDS_conf/"\
                            "EFFECTIVE_AREA_NEFF/K1000_S_eff_area.txt"
            else:
                area_file = "/home/cech/KiDSLenS/THELI_catalogues/KIDS_conf/"\
                            "EFFECTIVE_AREA_NEFF/K1000_eff_area.txt"

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
