import os
import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import file_header  # noqa: E402


if __name__ == "__main__":
    n_z = 5
    field_idx_EE = [(i, j) for i in range(n_z)
                    for j in range(i+1)]
    field_idx_TE = [(i, 0) for i in range(n_z)]

    TE_tag = "y_yan2019_nocib_beta1.2"

    # TE_tag = "cel_y_ACT_BN_nocib"
    do_joint = True
    do_EEEE = True

    NG_base_path = "../data/xcorr/cov/NG_cov_Planck"
    # NG_base_path = "../data/xcorr/cov/NG_cov_ACT"

    TE_base_path = f"../results/measurements_incl_m/shear_KiDS1000_{TE_tag}/likelihood/cov/"                    # noqa: E501
    EE_base_path = "../results/measurements_incl_m/shear_KiDS1000_shear_KiDS1000/likelihood/cov/"               # noqa: E501

    cov_TETE = {tag: np.loadtxt(os.path.join(TE_base_path, f"covariance_{tag}_TETE.txt"))                # noqa: E501
                for tag in ("gaussian_nka",)}
    cov_TETE.update({tag: np.loadtxt(os.path.join(NG_base_path, f"covariance_{tag}_TETE.txt"))           # noqa: E501
                     for tag in ("hmx_cNG_1h", "hmx_SSC_mask_wl", "m")})
    cov_TBTB = {tag: np.loadtxt(os.path.join(TE_base_path, f"covariance_{tag}_TBTB.txt"))                # noqa: E501
                for tag in ("gaussian_nka",)}

    if do_EEEE:
        cov_EEEE = {tag: np.loadtxt(os.path.join(EE_base_path, f"covariance_gaussian_{tag}_EEEE.txt"))   # noqa: E501
                    for tag in ("exact_noise", "exact_noise_mixed_terms", "nka_sva", "nka")}             # noqa: E501
        cov_EEEE.update({tag: np.loadtxt(os.path.join(NG_base_path, f"covariance_{tag}_EEEE.txt"))       # noqa: E501
                        for tag in ("hmx_cNG_1h", "hmx_SSC_mask_wl", "m")})
        cov_BBBB = {tag: np.loadtxt(os.path.join(EE_base_path, f"covariance_gaussian_{tag}_BBBB.txt"))   # noqa: E501
                    for tag in ("exact_noise", "exact_noise_mixed_terms", "nka_sva", "nka")}             # noqa: E501

    if do_joint:
        cov_joint = {tag: np.loadtxt(os.path.join(TE_base_path, f"covariance_{tag}_joint.txt"))          # noqa: E501
                     for tag in ("gaussian_nka",)}
        cov_joint.update({tag: np.loadtxt(os.path.join(NG_base_path, f"covariance_{tag}_joint.txt"))     # noqa: E501
                          for tag in ("hmx_cNG_1h", "hmx_SSC_mask_wl", "m")})

    contributions_EEEE = {  "total_gaussian":   ("exact_noise",
                                                 "exact_noise_mixed_terms",
                                                 "nka_sva"),
                            "total_SSC_mask":   ("exact_noise",
                                                 "exact_noise_mixed_terms",
                                                 "nka_sva",
                                                 "hmx_cNG_1h",
                                                 "hmx_SSC_mask_wl",
                                                 "m"),
                            "total_NKA_SSC_mask":  ("nka",
                                                    "hmx_cNG_1h",
                                                    "hmx_SSC_mask_wl",
                                                    "m")}
    contributions_BBBB = {  "total_gaussian":   ("exact_noise",
                                                 "exact_noise_mixed_terms",
                                                 "nka_sva")}

    contributions_TETE = {  "total_gaussian":  ("gaussian_nka",),
                            "total_SSC_mask":  ("gaussian_nka",
                                                "hmx_cNG_1h",
                                                "hmx_SSC_mask_wl",
                                                "m"),
                            }
    contributions_TBTB = {  "total_gaussian":  ("gaussian_nka",)}

    contributions_joint = {"total_gaussian":       ("gaussian_nka",),
                           "total_NKA_SSC_mask":   ("gaussian_nka",
                                                    "hmx_cNG_1h",
                                                    "hmx_SSC_mask_wl",
                                                    "m")}

    for name, tags in contributions_TETE.items():
        cov_TETE[name] = sum([cov_TETE[tag] for tag in tags])
    for name, tags in contributions_TBTB.items():
        cov_TBTB[name] = sum([cov_TBTB[tag] for tag in tags])

    for name, tags in contributions_TETE.items():
        header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_TE "
                  "(tomographic bins "
                  f"{', '.join([str(b) for b in field_idx_TE])})")
        header = file_header(header)
        np.savetxt(os.path.join(TE_base_path, f"covariance_{name}_TETE.txt"),
                   cov_TETE[name], header=header)
    for name, tags in contributions_TBTB.items():
        header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_TB "
                  "(tomographic bins "
                  f"{', '.join([str(b) for b in field_idx_TE])})")
        header = file_header(header)
        np.savetxt(os.path.join(TE_base_path, f"covariance_{name}_TBTB.txt"),
                   cov_TBTB[name], header=header)

    if do_EEEE:
        for name, tags in contributions_EEEE.items():
            cov_EEEE[name] = sum([cov_EEEE[tag] for tag in tags])
        for name, tags in contributions_BBBB.items():
            cov_BBBB[name] = sum([cov_BBBB[tag] for tag in tags])

        for name, tags in contributions_EEEE.items():
            header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_EE "
                      "(tomographic bins "
                      f"{', '.join([str(b) for b in field_idx_EE])})")
            header = file_header(header)
            np.savetxt(os.path.join(EE_base_path, f"covariance_{name}_EEEE.txt"),
                       cov_EEEE[name], header=header)
        for name, tags in contributions_BBBB.items():
            header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_BB "
                      "(tomographic bins "
                      f"{', '.join([str(b) for b in field_idx_EE])})")
            header = file_header(header)
            np.savetxt(os.path.join(EE_base_path, f"covariance_{name}_BBBB.txt"),
                       cov_BBBB[name], header=header)

    if do_joint:
        for name, tags in contributions_joint.items():
            cov_joint[name] = sum([cov_joint[tag] for tag in tags])

        c = cov_joint["total_NKA_SSC_mask"].copy()
        EEEE_total = cov_EEEE["total_SSC_mask"]
        n_EE = EEEE_total.shape[0]
        c[:n_EE, :n_EE] = EEEE_total

        cov_joint["total_SSC_mask"] = c

        for name, tags in (list(contributions_joint.items())
                           + [("total_SSC_mask", ("exact_noise",
                                                  "exact_noise_mixed_terms",
                                                  "nka_sva",
                                                  "hmx_cNG_1h",
                                                  "hmx_SSC_mask_wl",
                                                  "m"))]):
            header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_EE, Cl_TE "
                      f"(tomographic shear-shear bins: "
                      f"{', '.join([str(b) for b in field_idx_EE])}"
                      f" tomographic shear-foreground bins: "
                      f"{', '.join([str(b) for b in field_idx_TE])})")
            header = file_header(header)
            np.savetxt(os.path.join(TE_base_path, f"covariance_{name}_joint.txt"),
                       cov_joint[name], header=header)
