import subprocess
import os


if __name__ == "__main__":
    # workspace_path_EE = ("/disk09/ttroester/project_triad/namaster_workspaces/"
    #                      "shear_KiDS1000_shear_KiDS1000/")
    # workspace_path_TE = ("/disk09/ttroester/project_triad/namaster_workspaces/"
    #                      "shear_KiDS1000_y_milca/")

    workspace_path_EE = ("/disk09/ttroester/project_triad/namaster_workspaces/"
                         "shear_KiDS1000_shear_KiDS1000_cel/")
    workspace_path_TE = ("/disk09/ttroester/project_triad/namaster_workspaces/"
                         "shear_KiDS1000_cel_y_ACT_BN/")

    # Planck milca
    # Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",   # noqa: E501
    #                 "shear-foreground", "../results/measurements/shear_KiDS1000_y_milca/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",     # noqa: E501
    #                 "foreground-foreground", "../results/measurements/y_milca_y_milca/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]            # noqa: E501
    # output_path = "../results/measurements/shear_KiDS1000_y_milca/cov_3x2pt_MAP/nka/"

    # Planck nilc
    # Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",  # noqa: E501
    #                 "shear-foreground", "../results/measurements/shear_KiDS1000_y_milca/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",    # noqa: E501
    #                 "foreground-foreground", "../results/measurements/y_nilc_y_nilc/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]     # noqa: E501
    # output_path = "../results/measurements/shear_KiDS1000_y_nilc/cov_3x2pt_MAP/nka/"

    # Ziang's map
    # Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",  # noqa: E501
    #                 "shear-foreground", "../results/measurements/shear_KiDS1000_y_milca/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",        # noqa: E501
    #                 "foreground-foreground", "../results/measurements/y_ziang_nocib_y_ziang_nocib/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]  # noqa: E501
    # output_path = "../results/measurements/shear_KiDS1000_y_ziang_nocib/cov_3x2pt_MAP/nka/"

    # ACT BN
    Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",   # noqa: E501
                    "shear-foreground", "../results/measurements/shear_KiDS1000_cel_y_ACT_BN/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",       # noqa: E501
                    "foreground-foreground", "../results/measurements/y_ACT_BN_y_ACT_BN/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]  # noqa: E501
    output_path = "../results/measurements/shear_KiDS1000_cel_y_ACT_BN/cov_3x2pt_MAP/nka/"
    # ACT BN nocib
    # Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",   # noqa: E501
    #                 "shear-foreground", "../results/measurements/shear_KiDS1000_cel_y_ACT_BN/cov_Cls/Cl_cov_3x2pt_MAP_gal_{}-{}.npz",       # noqa: E501
    #                 "foreground-foreground", "../results/measurements/y_ACT_BN_nocib_y_ACT_BN_nocib/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]  # noqa: E501
    # output_path = "../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocib/cov_3x2pt_MAP/nka/"

    # Planck 100 GHz HFI, zero cross-correlation
    # Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_CCL_gal_{}-{}.npz",           # noqa: E501
    #                 "shear-foreground", "../results/measurements/shear_KiDS1000_100GHz_HFI/cov_Cls/Cl_cov_zeros_{}-{}.npz",       # noqa: E501
    #                 "foreground-foreground", "../results/measurements/100GHz_HFI_100GHz_HFI/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]  # noqa: E501
    # output_path = "../results/measurements/shear_KiDS1000_100GHz_HFI/cov/"

    # Planck 545 GHz CIB, zero cross-correlation
    # Cl_cov_files = ["shear-shear", "../results/measurements/shear_KiDS1000_shear_KiDS1000/cov_Cls/Cl_cov_CCL_gal_{}-{}.npz",           # noqa: E501
    #                 "shear-foreground", "../results/measurements/shear_KiDS1000_545GHz_CIB/cov_Cls/Cl_cov_GP_{}-{}.npz",       # noqa: E501
    #                 "foreground-foreground", "../results/measurements/545GHz_CIB_545GHz_CIB/cov_Cls/Cl_cov_smoothed_{}-{}.npz"]  # noqa: E501
    # output_path = "../results/measurements/shear_KiDS1000_545GHz_CIB/cov/"


    os.makedirs(output_path, exist_ok=True)

    n_shear_field = 5

    os.environ["OMP_NUM_THREADS"] = "20"

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
