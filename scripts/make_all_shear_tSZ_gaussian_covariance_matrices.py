import subprocess
import os


if __name__ == "__main__":
    workspace_path_EE = ("/disk09/ttroester/project_triad/namaster_workspaces/"
                        "shear_KiDS1000_shear_KiDS1000/")
    workspace_path_TE = ("/disk09/ttroester/project_triad/namaster_workspaces/"
                        "shear_KiDS1000_y_milca/")

    Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_CCL_gal_{}-{}.npz",  # noqa: E501
                    "shear-foreground", "../results/measurements/shear_KiDS1000_y_milca/cov_Cls/Cl_cov_CCL_gal_{}-{}.npz",    # noqa: E501
                    "foreground-foreground", "../results/measurements/y_milca_y_milca/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]     # noqa: E501

    output_path = "../results/measurements/shear_KiDS1000_y_milca/cov/"

    n_shear_field = 5

    os.environ["OMP_NUM_THREADS"] = "44"

    for cov_block in ["TETE", "EETE"]:
        print("Cov block: ", cov_block)
        if cov_block == "EETE":
            cov_fields_a = "shear-shear"
            cov_fields_b = "shear-foreground"
            field_idx_EE = [(i, j) for i in range(n_shear_field)
                            for j in range(i+1)]
            field_idx_TE = [(i, 0) for i in range(n_shear_field)]
            cov_idx = []
            for idx_a in field_idx_EE:
                for idx_b in field_idx_TE:
                    cov_idx.append((idx_a, idx_b))
        elif cov_block == "TETE":
            cov_fields_a = "shear-foreground"
            cov_fields_b = cov_fields_a
            field_idx_TE = [(i, 0) for i in range(n_shear_field)]
            cov_idx = []
            for i, idx_a in enumerate(field_idx_TE):
                for idx_b in field_idx_TE[:i+1]:
                    cov_idx.append((idx_a, idx_b))

        for idx_a, idx_b in cov_idx:
            print("Bin combination: ", idx_a, idx_b)
            print()

            cmd = ["python", "compute_shear_tSZ_covariance.py"]

            if cov_block == "TETE":
                cmd += ["--pymaster-workspace-path-a", workspace_path_TE]
            elif cov_block == "EETE":
                cmd += ["--pymaster-workspace-path-a", workspace_path_EE]
            cmd += ["--pymaster-workspace-path-b", workspace_path_TE]

            cmd += ["--output-path", output_path]

            cmd += ["--Cl-cov-file"] + Cl_cov_files

            cmd += ["--idx-a", "{}-{}".format(*idx_a)]
            cmd += ["--idx-b", "{}-{}".format(*idx_b)]

            cmd += ["--fields-a", cov_fields_a]
            cmd += ["--fields-b", cov_fields_b]

            subprocess.check_call(cmd)
