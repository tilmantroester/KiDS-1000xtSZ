import subprocess
import os


if __name__ == "__main__":
    workspace_path = ("/disk09/ttroester/project_triad/namaster_workspaces/"
                      "shear_KiDS1000_shear_KiDS1000/")

    Cl_cov_file = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"
                   "cov_Cls/Cl_cov_3x2pt_MAP_")

    # NKA signal-signal terms. Will also compute the NKA versions
    # of the noise terms
    mode = "nka"

    # Exact noise-noise terms.
    # mode = "exact_noise_noise"

    # Noise-signal terms, with exact noise-noise terms
    # mode = "exact_noise_signal"

    if mode == "exact_noise_noise":
        output_path = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"
                       "cov_3x2pt_MAP/exact_noise/")
    elif mode == "exact_noise_signal":
        output_path = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"
                       "cov_3x2pt_MAP/exact_noise_mixed_terms/")
    elif mode == "nka":
        output_path = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"
                       "cov_3x2pt_MAP/nka/")

    cov_idx_slice = slice(0, 120)
    # cov_idx_slice = slice(0, 30)
    # cov_idx_slice = slice(30, 60)
    # cov_idx_slice = slice(60, 90)
    # cov_idx_slice = slice(90, 120)

    n_z = 5

    os.environ["OMP_NUM_THREADS"] = "20"

    field_idx = [(i, j) for i in range(n_z)
                 for j in range(i+1)]

    cov_idx = [(idx_a, idx_b) for i, idx_a in enumerate(field_idx)
               for idx_b in field_idx[:i+1]]

    for idx_a, idx_b in cov_idx[cov_idx_slice]:
        print()
        print("Bin combination: ", idx_a, idx_b)
        print()

        if mode == "nka" or mode == "exact_noise_signal":
            cmd = ["python", "compute_cosmic_shear_covariance.py"]
        else:
            # Exact noise-noise
            cmd = ["python", "compute_exact_noise_cosmic_shear_covariance.py"]

        cmd += ["--pymaster-workspace-path", workspace_path]
        cmd += ["--output-path", output_path]

        cmd += ["--Cl-cov-file", Cl_cov_file]

        if mode == "nka":
            cmd += ["--compute-nka-terms"]
        elif mode == "exact_noise_signal":
            cmd += ["--compute-exact-noise-mixed-terms"]

        cmd += ["--idx-a", "{}-{}".format(*idx_a)]
        cmd += ["--idx-b", "{}-{}".format(*idx_b)]

        subprocess.check_call(cmd)
