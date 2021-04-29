import numpy as np

import healpy

import pylenspice.pylenspice

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
    map_stats = True

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
                ra, dec, e1, e2, w, m = pylenspice.pylenspice.prepare_catalog(
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

    if map_stats:
        print("Map-based stats")

        maps = [("All_gal", ""),
                ("N_cel", "_cel_N")]

        for map_name, suffix in maps:
            print("Map suffix: ", map_name)
            A_KiDS = {}

            sum_w_map = {}
            sum_w_sq_map = {}
            sum_w_sq_e1_sq_map = {}
            sum_w_sq_e2_sq_map = {}
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
                sum_w_sq_e1_sq_map[z_cut] = np.sum(shear_weight**2 * e1**2)
                sum_w_sq_e2_sq_map[z_cut] = np.sum(shear_weight**2 * e2**2)

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
