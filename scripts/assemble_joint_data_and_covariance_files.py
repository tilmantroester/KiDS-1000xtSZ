import os

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import file_header  # noqa: E402


if __name__ == "__main__":
    n_z = 5

    covariance_paths = {"EEEE": "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov/",
                        "TETE": "../results/measurements/shear_KiDS1000_y_milca/cov/",
                        "EETE": "../results/measurements/shear_KiDS1000_y_milca/cov/"}
    data_Cl_path = {"EE": "../results/measurements/shear_KiDS1000_shear_KiDS1000/data/",
                    "TE": "../results/measurements/shear_KiDS1000_y_milca/data/"}

    Cl_file = {"EE": "../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/Cl_EE_gal.txt",
               "BB": "../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/Cl_BB_gal.txt",
               "TE": "../results/measurements/shear_KiDS1000_y_milca/likelihood/Cl_TE_gal.txt",
               "TB": "../results/measurements/shear_KiDS1000_y_milca/likelihood/Cl_TB_gal.txt"}

    covariance_files = {"EEEE": "../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_gaussian_EE.txt",
                        "BBBB": "../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/covariance_gaussian_BB.txt",
                        "TETE": "../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_gaussian_TE.txt",
                        "TBTB": "../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_gaussian_TB.txt",
                        "joint": "../results/measurements/shear_KiDS1000_y_milca/likelihood/covariance_gaussian_EE-TE.txt"}

    field_idx_EE = [(i, j) for i in range(n_z)
                    for j in range(i+1)]
    field_idx_TE = [(i, 0) for i in range(n_z)]

    Cl_headers = {"EE": (f"ell, Cl_EE (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_EE])})"),
                  "BB": (f"ell, Cl_BB (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_EE])})"),
                  "TE": (f"ell, Cl_TE (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_TE])})"),
                  "TB": (f"ell, Cl_BB (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_TE])})")}
    cov_headers = {"EEEE": (f"Covariance of Cl_EE (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_EE])})"),
                   "BBBB": (f"Covariance of Cl_BB (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_EE])})"),
                   "TETE": (f"Covariance of Cl_TE (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_TE])})"),
                   "TBTB": (f"Covariance of Cl_TB (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_TE])})"),
                   "joint": (f"Covariance of Cl_EE, Cl_TE "
                             f"(tomographic shear-shear bins: "
                             f"{', '.join([str(b) for b in field_idx_EE])}"
                             f" tomographic shear-foreground bins: "
                             f"{', '.join([str(b) for b in field_idx_TE])})")}

    Cl = {"EE": [], "BB": [], "TE": [], "TB": []}
    ell = {}
    for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
        d = np.load(os.path.join(data_Cl_path["EE"],
                                 f"Cl_gal_{idx_a1}-{idx_a2}.npz"))
        ell["EE"] = d["ell_eff"]
        ell["BB"] = d["ell_eff"]
        Cl["EE"].append(d["Cl_decoupled"][0])
        Cl["BB"].append(d["Cl_decoupled"][3])

    for i, (idx_a1, idx_a2) in enumerate(field_idx_TE):
        d = np.load(os.path.join(data_Cl_path["TE"],
                                 f"Cl_gal_{idx_a1}-{idx_a2}.npz"))
        ell["TE"] = d["ell_eff"]
        ell["TB"] = d["ell_eff"]
        Cl["TE"].append(d["Cl_decoupled"][0])
        Cl["TB"].append(d["Cl_decoupled"][1])

    for probe in ["EE", "BB", "TE", "TB"]:
        data = np.vstack([ell[probe]] + Cl[probe]).T
        header = file_header(Cl_headers[probe])
        np.savetxt(Cl_file[probe], data, header=header)

    n_ell_bin_EE = len(ell["EE"])
    n_ell_bin_TE = len(ell["TE"])
    n_EE = n_ell_bin_EE*len(field_idx_EE)
    n_TE = n_ell_bin_TE*len(field_idx_TE)

    cov_gaussian = {"EEEE": np.zeros((n_EE, n_EE)),
                    "BBBB": np.zeros((n_EE, n_EE)),
                    "TETE": np.zeros((n_TE, n_TE)),
                    "TBTB": np.zeros((n_TE, n_TE)),
                    "EETE": np.zeros((n_EE, n_TE)),
                    "BBTB": np.zeros((n_EE, n_TE))}
    # EEEE/BBBB
    for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
        for j, (idx_b1, idx_b2) in enumerate(field_idx_EE[:i+1]):
            c = np.load(os.path.join(
                        covariance_paths["EEEE"],
                        f"cov_shear_shear_{idx_a1}-{idx_a2}_"
                        f"{idx_b1}-{idx_b2}.npz"))["ssss"].reshape(
                                                    n_ell_bin_EE, 4,
                                                    n_ell_bin_EE, 4)

            cov_gaussian["EEEE"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                 j*n_ell_bin_EE:(j+1)*n_ell_bin_EE] = c[:, 0, :, 0]
            cov_gaussian["BBBB"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                 j*n_ell_bin_EE:(j+1)*n_ell_bin_EE] = c[:, 3, :, 3]

    # TETE/TBTB
    for i, (idx_a1, idx_a2) in enumerate(field_idx_TE):
        for j, (idx_b1, idx_b2) in enumerate(field_idx_TE[:i+1]):
            c = np.load(os.path.join(
                        covariance_paths["TETE"],
                        f"cov_shear_{idx_a1}_foreground_{idx_a2}_"
                        f"shear_{idx_b1}_foreground_{idx_b2}.npz"))["ssss"]
            c = c.reshape(n_ell_bin_TE, 2, n_ell_bin_TE, 2)

            cov_gaussian["TETE"][i*n_ell_bin_TE:(i+1)*n_ell_bin_TE,
                                 j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c[:, 0, :, 0]
            cov_gaussian["TBTB"][i*n_ell_bin_TE:(i+1)*n_ell_bin_TE,
                                 j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c[:, 1, :, 1]

    # EETE/BBTB (in the upper triangle)
    for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
        for j, (idx_b1, idx_b2) in enumerate(field_idx_TE):
            c = np.load(os.path.join(
                        covariance_paths["EETE"],
                        f"cov_shear_{idx_a1}_shear_{idx_a2}_"
                        f"shear_{idx_b1}_foreground_{idx_b2}.npz"))["ssss"]
            c = c.reshape(n_ell_bin_EE, 4, n_ell_bin_TE, 2)

            cov_gaussian["EETE"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                 j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c[:, 0, :, 0]
            cov_gaussian["BBTB"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                 j*n_ell_bin_TE:(j+1)*n_ell_bin_TE] = c[:, 3, :, 1]

    cov_gaussian["joint"] = np.zeros((n_EE+n_TE, n_EE+n_TE))

    cov_gaussian["joint"][:n_EE, :n_EE] = cov_gaussian["EEEE"]
    cov_gaussian["joint"][n_EE:n_EE+n_TE, :n_EE] = cov_gaussian["EETE"].T
    cov_gaussian["joint"][n_EE:n_EE+n_TE, n_EE:n_EE+n_TE] = cov_gaussian["TETE"]

    for cov_block in ["EEEE", "BBBB", "TETE", "TBTB", "joint"]:
        c = cov_gaussian[cov_block]
        cov_gaussian[cov_block][np.triu_indices_from(c, k=1)] = \
            c.T[np.triu_indices_from(c, k=1)]

        header = file_header(cov_headers[cov_block])
        np.savetxt(covariance_files[cov_block], cov_gaussian[cov_block],
                   header=header)
