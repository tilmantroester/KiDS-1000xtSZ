import os

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import file_header  # noqa: E402


if __name__ == "__main__":
    n_z = 5

    base_path_EE = "../results/measurements/shear_KiDS1000_shear_KiDS1000/"
    base_path_TE = "../results/measurements/shear_KiDS1000_y_milca/"
    # base_path_TE = "../results/measurements/shear_KiDS1000_545GHz_CIB/"

    probes = []
    cov_blocks = ["TETE", "EEEE", "joint"]

    Cl_suffix = "gal"

    covariance_paths = {"EEEE": os.path.join(base_path_EE, "cov_3x2pt_MAP/nka/"),
                        "TETE": os.path.join(base_path_TE, "cov_3x2pt_MAP/nka/"),
                        "EETE": os.path.join(base_path_TE, "cov_3x2pt_MAP/nka/")}
    data_Cl_path = {"EE": os.path.join(base_path_EE, "data/"),
                    "TE": os.path.join(base_path_TE, "data/")}

    Cl_file = {"EE": os.path.join(base_path_EE, f"likelihood/Cl_EE_gal.txt"),
               "BB": os.path.join(base_path_EE, f"likelihood/Cl_BB_gal.txt"),
               "TE": os.path.join(base_path_TE, f"likelihood/Cl_TE_{Cl_suffix}.txt"),
               "TB": os.path.join(base_path_TE, f"likelihood/Cl_TB_{Cl_suffix}.txt")}

    covariance_files = {"EEEE": os.path.join(base_path_TE, "likelihood/cov/covariance_gaussian_nka_EEEE.txt"),
                        "BBBB": os.path.join(base_path_TE, "likelihood/cov/covariance_gaussian_nka_BBBB.txt"),
                        "TETE": os.path.join(base_path_TE, f"likelihood/cov/covariance_gaussian_nka_TETE.txt"),
                        "TBTB": os.path.join(base_path_TE, f"likelihood/cov/covariance_gaussian_nka_TBTB.txt"),
                        "joint": os.path.join(base_path_TE, f"likelihood/cov/covariance_gaussian_nka_joint.txt")}

    NG_covariance_files = None
    NG_covariance_files = {"EEEE":  {"m": os.path.join(base_path_EE, "likelihood/covariance_m_EE.txt"),
                                     "SSC-disc": os.path.join(base_path_EE, "likelihood/covariance_SSC_disc_EE.txt"),
                                     "SSC-mask": os.path.join(base_path_EE, "likelihood/covariance_SSC_mask_EE.txt")},
                           "TETE":  {#"m": os.path.join(base_path_TE, "likelihood/covariance_m_TE.txt"),
                                     "SSC-disc": os.path.join(base_path_TE, "likelihood/covariance_hmx_SSC_disc_TE.txt"),
                                     #"SSC-mask": os.path.join(base_path_TE, "likelihood/covariance_hmx_SSC_mask_TE.txt"),
                                     "cNG-1h": os.path.join(base_path_TE, "likelihood/covariance_hmx_cNG_1h_TE.txt")},
                           "joint": {#"m": os.path.join(base_path_TE, "likelihood/covariance_m_TE.txt"),
                                     "SSC-disc": os.path.join(base_path_TE, "likelihood/covariance_hmx_SSC_disc_EE+TE.txt"),
                                     #"SSC-mask": os.path.join(base_path_TE, "likelihood/covariance_hmx_SSC_mask_EE+TE.txt"),
                                     "cNG-1h": os.path.join(base_path_TE, "likelihood/covariance_hmx_cNG_1h_EE+TE.txt")}}
    
    for probe in probes:
        os.makedirs(os.path.split(Cl_file[probe])[0], exist_ok=True)

    field_idx_EE = [(i, j) for i in range(n_z)
                    for j in range(i+1)]
    field_idx_TE = [(i, 0) for i in range(n_z)]

    Cl_headers = {"EE": (f"ell, Cl_EE (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_EE])})\n"
                         f"Data path: {data_Cl_path['EE']}"),
                  "BB": (f"ell, Cl_BB (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_EE])})\n"
                         f"Data path: {data_Cl_path['EE']}"),
                  "TE": (f"ell, Cl_TE (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_TE])})\n"
                         f"Data path: {data_Cl_path['TE']}"),
                  "TB": (f"ell, Cl_TB (tomographic bins "
                         f"{', '.join([str(b) for b in field_idx_TE])})\n"
                         f"Data path: {data_Cl_path['TE']}")}
    cov_headers = {"EEEE": (f"Covariance of Cl_EE (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_EE])})\n"
                            f"Data path: {covariance_paths['EEEE']}"),
                   "BBBB": (f"Covariance of Cl_BB (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_EE])})\n"
                            f"Data path: {covariance_paths['EEEE']}"),
                   "TETE": (f"Covariance of Cl_TE (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_TE])})\n"
                            f"Data path: {covariance_paths['TETE']}"),
                   "TBTB": (f"Covariance of Cl_TB (tomographic bins "
                            f"{', '.join([str(b) for b in field_idx_TE])})\n"
                            f"Data path: {covariance_paths['TETE']}"),
                   "joint": (f"Covariance of Cl_EE, Cl_TE "
                             f"(tomographic shear-shear bins: "
                             f"{', '.join([str(b) for b in field_idx_EE])}"
                             f" tomographic shear-foreground bins: "
                             f"{', '.join([str(b) for b in field_idx_TE])})\n"
                             f"Data paths:\n"
                             f"EEEE:{covariance_paths['EEEE']}\n"
                             f"TETE:{covariance_paths['TETE']}\n"
                             f"EETE:{covariance_paths['EETE']}")}

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
                                 f"Cl_{Cl_suffix}_{idx_a1}-{idx_a2}.npz"))
        ell["TE"] = d["ell_eff"]
        ell["TB"] = d["ell_eff"]
        Cl["TE"].append(d["Cl_decoupled"][0])
        Cl["TB"].append(d["Cl_decoupled"][1])

    # if "EE" in probes:
    #     for probe in ["EE", "BB"]:
    #         data = np.vstack([ell[probe]] + Cl[probe]).T
    #         header = file_header(Cl_headers[probe])
    #         np.savetxt(Cl_file[probe], data, header=header)
    # if "TE" in probes:
    #     for probe in ["TE", "TB"]:
    #         data = np.vstack([ell[probe]] + Cl[probe]).T
    #         header = file_header(Cl_headers[probe])
    #         np.savetxt(Cl_file[probe], data, header=header)

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
    if "EEEE" in cov_blocks:
        # EEEE/BBBB
        for i, (idx_a1, idx_a2) in enumerate(field_idx_EE):
            for j, (idx_b1, idx_b2) in enumerate(field_idx_EE[:i+1]):
                c = np.load(os.path.join(
                            covariance_paths["EEEE"],
                            f"cov_shear_{idx_a1}_shear_{idx_a2}_"
                            f"shear_{idx_b1}_shear_{idx_b2}.npz"))["aaaa"].reshape(
                                                        n_ell_bin_EE, 4,
                                                        n_ell_bin_EE, 4)

                cov_gaussian["EEEE"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                    j*n_ell_bin_EE:(j+1)*n_ell_bin_EE] = c[:, 0, :, 0]
                cov_gaussian["BBBB"][i*n_ell_bin_EE:(i+1)*n_ell_bin_EE,
                                    j*n_ell_bin_EE:(j+1)*n_ell_bin_EE] = c[:, 3, :, 3]

    if "TETE" in cov_blocks:
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

    if "EETE" in cov_blocks or "joint" in cov_blocks:
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
    if "joint" in cov_blocks:
        cov_gaussian["joint"] = np.zeros((n_EE+n_TE, n_EE+n_TE))

        cov_gaussian["joint"][:n_EE, :n_EE] = cov_gaussian["EEEE"]
        cov_gaussian["joint"][n_EE:n_EE+n_TE, :n_EE] = cov_gaussian["EETE"].T
        cov_gaussian["joint"][n_EE:n_EE+n_TE, n_EE:n_EE+n_TE] = cov_gaussian["TETE"]

    for cov_block in cov_blocks:
        c = cov_gaussian[cov_block]
        cov_gaussian[cov_block][np.triu_indices_from(c, k=1)] = \
            c.T[np.triu_indices_from(c, k=1)]

        header = file_header(cov_headers[cov_block])
        np.savetxt(covariance_files[cov_block], cov_gaussian[cov_block],
                   header=header)
