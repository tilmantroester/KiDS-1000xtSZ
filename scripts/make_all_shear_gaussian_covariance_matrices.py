import subprocess
import os


if __name__ == "__main__":
    workspace_path = ("/disk09/ttroester/project_triad/namaster_workspaces/"
                      "shear_KiDS1000_shear_KiDS1000/")

    Cl_cov_file = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"
                   "cov_Cls/Cl_cov_CCL_gal_")

    output_path = "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov/"

    n_z = 5

    os.environ["OMP_NUM_THREADS"] = "44"

    field_idx = [(i, j) for i in range(n_z)
                 for j in range(i+1)]

    for i, idx_a in enumerate(field_idx):
        for idx_b in field_idx[:i+1]:
            print("Bin combination: ", idx_a, idx_b)
            print()

            cmd = ["python", "compute_cosmic_shear_covariance.py"]

            cmd += ["--pymaster-workspace-path", workspace_path]
            cmd += ["--output-path", output_path]

            cmd += ["--Cl-cov-file", Cl_cov_file]

            cmd += ["--idx-a", "{}-{}".format(*idx_a)]
            cmd += ["--idx-b", "{}-{}".format(*idx_b)]

            subprocess.check_call(cmd)
