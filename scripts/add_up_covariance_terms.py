import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import file_header  # noqa: E402


if __name__ == "__main__":
    n_z = 5
    field_idx_EE = [(i, j) for i in range(n_z)
                    for j in range(i+1)]
    field_idx_TE = [(i, 0) for i in range(n_z)]

    cov_EEEE = {tag: np.loadtxt(f"../results/measurements/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_gaussian_{tag}_EEEE.txt")
                for tag in ("exact_noise", "exact_noise_mixed_terms", "nka_sva", "nka")}

    cov_EEEE.update({tag: np.loadtxt(f"../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_{tag}_EEEE.txt")
                     for tag in ("hmx_cNG_1h", "hmx_SSC_disc", "hmx_SSC_mask_wl", "m")})

    cov_TETE = {tag: np.loadtxt(f"../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_{tag}_TETE.txt")
                for tag in ("gaussian_nka", "hmx_cNG_1h", "hmx_SSC_disc", "hmx_SSC_mask_wl", "hmx_SSC_mask_wl_overlap", "m")}

    cov_joint = {tag: np.loadtxt(f"../results/measurements/shear_KiDS1000_y_milca/likelihood/cov/covariance_{tag}_joint.txt")
                 for tag in ("gaussian_nka", "hmx_cNG_1h", "hmx_SSC_disc", "hmx_SSC_mask_wl", "hmx_SSC_mask_wl_overlap", "m")}

    contributions_EEEE =   {"total_SSC_disc":  ("exact_noise",
                                                "exact_noise_mixed_terms",
                                                "nka_sva",
                                                "hmx_cNG_1h",
                                                "hmx_SSC_disc",
                                                "m"),
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

    contributions_TETE =   {"total_SSC_disc":  ("gaussian_nka",
                                                "hmx_cNG_1h",
                                                "hmx_SSC_disc",
                                                "m"),
                            "total_SSC_mask":   ("gaussian_nka",
                                                "hmx_cNG_1h",
                                                "hmx_SSC_mask_wl",
                                                "m"),
                            "total_SSC_mask_overlap":  ("gaussian_nka",
                                                "hmx_cNG_1h",
                                                "hmx_SSC_mask_wl_overlap",
                                                "m"),}

    contributions_joint =   {"total_NKA_SSC_mask":   ("gaussian_nka",
                                                "hmx_cNG_1h",
                                                "hmx_SSC_mask_wl",
                                                "m"),}

    for name, tags in contributions_EEEE.items():
        cov_EEEE[name] = sum([cov_EEEE[tag] for tag in tags])
    
    for name, tags in contributions_TETE.items():
        cov_TETE[name] = sum([cov_TETE[tag] for tag in tags])
    
    for name, tags in contributions_joint.items():
        cov_joint[name] = sum([cov_joint[tag] for tag in tags])
    
    
    c = cov_joint["total_NKA_SSC_mask"].copy()
    EEEE_total = cov_EEEE["total_SSC_mask"]
    n_EE = EEEE_total.shape[0]
    c[:n_EE, :n_EE] = EEEE_total

    cov_joint["total_SSC_mask"] = c

    for name, tags in contributions_EEEE.items():
        header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_EE (tomographic bins "
                  f"{', '.join([str(b) for b in field_idx_EE])})")
        header = file_header(header)
        np.savetxt(f"../results/measurements/shear_KiDS1000_shear_KiDS1000/"
                   f"likelihood/cov/covariance_{name}_EEEE.txt",
                   cov_EEEE[name], header=header)

    for name, tags in contributions_TETE.items():
        header = (f"Covariance {name} ({'+'.join(tags)}) of Cl_TE (tomographic bins "
                  f"{', '.join([str(b) for b in field_idx_TE])})")
        header = file_header(header)
        np.savetxt(f"../results/measurements/shear_KiDS1000_y_milca/"
                   f"likelihood/cov/covariance_{name}_TETE.txt",
                   cov_TETE[name], header=header)

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
        np.savetxt(f"../results/measurements/shear_KiDS1000_y_milca/"
                   f"likelihood/cov/covariance_{name}_joint.txt",
                   cov_joint[name], header=header)
