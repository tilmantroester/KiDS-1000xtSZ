import os

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import file_header  # noqa: E402


if __name__ == "__main__":
    n_z = 5

    # key = "mmmm"
    # tag = "exact_noise_mixed_terms"
    # key = "nnnn"
    # tag = "exact_noise"
    key = "aaaa"
    tag = "nka"

    covariance_path = ("../results/measurements/"
                       f"shear_KiDS1000_shear_KiDS1000/cov_3x2pt_MAP/nka/")
    covariance_file_template = ("cov_shear_{}_shear_{}_"
                                "shear_{}_shear_{}.npz")
    # covariance_file_template = ("cov_shear_noise_{}_shear_noise_{}_"
    #                             "shear_noise_{}_shear_noise_{}.npz")

    data_Cl_path = ("../results/measurements/"
                    "shear_KiDS1000_shear_KiDS1000/data/")

    Cl_file = ("../results/measurements/"
               "shear_KiDS1000_shear_KiDS1000/likelihood/Cl_EE_gal.txt")

    covariance_file = (f"../results/measurements/"
                       f"shear_KiDS1000_shear_KiDS1000/likelihood/cov/"
                       f"covariance_gaussian_{tag}_EEEE.txt")

    field_idx = [(i, j) for i in range(n_z)
                 for j in range(i+1)]

    Cl_EE = []
    Cl_BB = []
    for i, (idx_a1, idx_a2) in enumerate(field_idx):
        d = np.load(os.path.join(data_Cl_path,
                                 f"Cl_gal_{idx_a1}-{idx_a2}.npz"))
        ell = d["ell_eff"]
        Cl_EE.append(d["Cl_decoupled"][0])

    # data = np.vstack([ell] + Cl_EE).T
    # header = (f"ell, Cl_EE (tomographic bins "
    #           f"{', '.join([str(b) for b in field_idx])})")
    # header = file_header(header)
    # np.savetxt(Cl_file, data, header=header)

    n_ell_bin = len(ell)
    cov_gaussian_EE = np.zeros((n_ell_bin*len(field_idx),
                                n_ell_bin*len(field_idx)))

    for i, (idx_a1, idx_a2) in enumerate(field_idx):
        for j, (idx_b1, idx_b2) in enumerate(field_idx[:i+1]):
            cov_file = covariance_file_template.format(
                            idx_a1, idx_a2, idx_b1, idx_b2)
            cov_file = os.path.join(covariance_path, cov_file)
            if not os.path.isfile(cov_file):
                print(f"No file {cov_file}. Skipping.")
                continue
            c = np.load(cov_file)[key].reshape(n_ell_bin, 4,
                                               n_ell_bin, 4)[:, 0, :, 0]

            cov_gaussian_EE[i*n_ell_bin:(i+1)*n_ell_bin,
                            j*n_ell_bin:(j+1)*n_ell_bin] = c

    cov_gaussian_EE[np.triu_indices_from(cov_gaussian_EE, k=1)] = \
        cov_gaussian_EE.T[np.triu_indices_from(cov_gaussian_EE, k=1)]

    header = (f"Covariance {tag} of Cl_EE (tomographic bins "
              f"{', '.join([str(b) for b in field_idx])})")
    header = file_header(header)
    np.savetxt(covariance_file, cov_gaussian_EE, header=header)
