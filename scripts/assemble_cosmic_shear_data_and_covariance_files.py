import os

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import file_header  # noqa: E402


if __name__ == "__main__":
    n_z = 5

    # key = "mmmm"
    # tag = "exact_noise_mixed_terms"
    key = "nnnn"
    tag = "exact_noise"
    # key = "aaaa"
    # tag = "nka"
    # key = "ssss"
    # tag = "nka_sva"
    # key = "nnnn"
    # tag = "nka_noise"

    if tag[:3] == "nka":
        input_dir = "nka"
    else:
        input_dir = tag

    covariance_path = ("../results/measurements/"
                       f"shear_KiDS1000_shear_KiDS1000/cov_3x2pt_MAP/{input_dir}/")
    if tag == "exact_noise":
        covariance_file_template = ("cov_shear_noise_{}_shear_noise_{}_"
                                    "shear_noise_{}_shear_noise_{}.npz")
    else:
        covariance_file_template = ("cov_shear_{}_shear_{}_"
                                    "shear_{}_shear_{}.npz")

    base_path = "../results/measurements/shear_KiDS1000_shear_KiDS1000/"

    data_Cl_path = os.path.join(base_path, "data/")

    Cl_file = {"EE": os.path.join(base_path, "likelihood/data/Cl_EE_shear_KiDS1000_gal.txt"),
               "BB": os.path.join(base_path, "likelihood/data/Cl_BB_shear_KiDS1000_gal.txt")}

    covariance_file = {"EEEE": os.path.join(base_path, f"likelihood/cov/covariance_gaussian_{tag}_EEEE.txt"),
                       "BBBB": os.path.join(base_path, f"likelihood/cov/covariance_gaussian_{tag}_BBBB.txt")}

    os.makedirs(os.path.split(Cl_file["EE"])[0], exist_ok=True)
    os.makedirs(os.path.split(covariance_file["EEEE"])[0], exist_ok=True)

    field_idx = [(i, j) for i in range(n_z)
                 for j in range(i+1)]

    Cl_EE = []
    Cl_BB = []
    for i, (idx_a1, idx_a2) in enumerate(field_idx):
        d = np.load(os.path.join(data_Cl_path,
                                 f"Cl_gal_{idx_a1}-{idx_a2}.npz"))
        ell = d["ell_eff"]
        Cl_EE.append(d["Cl_decoupled"][0])
        Cl_BB.append(d["Cl_decoupled"][3])

    data = np.vstack([ell] + Cl_EE).T
    header = (f"ell, Cl_EE (tomographic bins "
              f"{', '.join([str(b) for b in field_idx])})")
    header = file_header(header)
    np.savetxt(Cl_file["EE"], data, header=header)

    data = np.vstack([ell] + Cl_BB).T
    header = (f"ell, Cl_BB (tomographic bins "
              f"{', '.join([str(b) for b in field_idx])})")
    header = file_header(header)
    np.savetxt(Cl_file["BB"], data, header=header)

    n_ell_bin = len(ell)
    cov_gaussian_EEEE = np.zeros((n_ell_bin*len(field_idx),
                                  n_ell_bin*len(field_idx)))
    cov_gaussian_BBBB = np.zeros_like(cov_gaussian_EEEE)      

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
            cov_gaussian_EEEE[i*n_ell_bin:(i+1)*n_ell_bin,
                              j*n_ell_bin:(j+1)*n_ell_bin] = c

            c = np.load(cov_file)[key].reshape(n_ell_bin, 4,
                                               n_ell_bin, 4)[:, 3, :, 3]
            cov_gaussian_BBBB[i*n_ell_bin:(i+1)*n_ell_bin,
                              j*n_ell_bin:(j+1)*n_ell_bin] = c

    cov_gaussian_EEEE[np.triu_indices_from(cov_gaussian_EEEE, k=1)] = \
        cov_gaussian_EEEE.T[np.triu_indices_from(cov_gaussian_EEEE, k=1)]
    cov_gaussian_BBBB[np.triu_indices_from(cov_gaussian_BBBB, k=1)] = \
        cov_gaussian_BBBB.T[np.triu_indices_from(cov_gaussian_BBBB, k=1)]

    header = (f"Covariance {tag} of Cl_EE (tomographic bins "
              f"{', '.join([str(b) for b in field_idx])})")
    header = file_header(header)
    np.savetxt(covariance_file["EEEE"], cov_gaussian_EEEE, header=header)

    header = (f"Covariance {tag} of Cl_BB (tomographic bins "
              f"{', '.join([str(b) for b in field_idx])})")
    header = file_header(header)
    np.savetxt(covariance_file["BBBB"], cov_gaussian_BBBB, header=header)
