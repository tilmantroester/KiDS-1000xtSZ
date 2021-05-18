import subprocess
import os


if __name__ == "__main__":
    workspace_file_template = ("/disk09/ttroester/project_triad/namaster_workspaces/"  # noqa: E501
                               "shear_KiDS1000_shear_KiDS1000/"
                               "pymaster_workspace_shear_{}_shear_{}.fits")

    bandpower_window_file_template = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"  # noqa: E501
                                      "data/pymaster_bandpower_windows_{}-{}.npy")              # noqa: E501

    Cl_data_file_template = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"           # noqa: E501
                             "data/Cl_gal_{}-{}.npz")

    Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_shear_KiDS1000/"            # noqa: E501
                            "cov_Cls/Cl_cov_CCL_gal_{}-{}.npz")

    bin_operator_file = "../data/xcorr/bin_operator_log_n_bin_13_ell_51-2952_namaster.txt"      # noqa: E501

    catalog_files = ["../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz"]     # noqa: E501

    nofz_files = ["../runs/base_setup/data/load_source_nz/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO1_Nz.asc",  # noqa: E501
                  "../runs/base_setup/data/load_source_nz/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO2_Nz.asc",  # noqa: E501
                  "../runs/base_setup/data/load_source_nz/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO3_Nz.asc",  # noqa: E501
                  "../runs/base_setup/data/load_source_nz/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO4_Nz.asc",  # noqa: E501
                  "../runs/base_setup/data/load_source_nz/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.asc"]  # noqa: E501

    os.environ["OMP_NUM_THREADS"] = "44"

    field_idx = [(i, j) for i in range(len(catalog_files))
                 for j in range(i+1)]

    for idx in field_idx:
        print("Bin combination: ", idx)
        print()

        cmd = ["python", "namaster_cosmic_shear_measurements.py"]
        cmd += ["--bin-operator", bin_operator_file]

        cmd += ["--nofz-files", nofz_files[idx[0]]]
        if idx[0] != idx[1]:
            cmd += [nofz_files[idx[1]]]

        cmd += ["--shear-catalogs", catalog_files[idx[0]]]
        if idx[0] != idx[1]:
            cmd += [catalog_files[idx[1]]]

        workspace_file = workspace_file_template.format(*idx)
        cmd += ["--pymaster-workspace", workspace_file]
        if not os.path.isfile(workspace_file):
            cmd += ["--compute-coupling-matrix"]

        cmd += ["--n-iter", "3"]
        cmd += ["--n-randoms", "0"]

        Cl_cov_file = Cl_cov_file_template.format(*idx)
        cmd += ["--Cl-cov-filename", Cl_cov_file]

        Cl_data_file = Cl_data_file_template.format(*idx)
        cmd += ["--Cl-data-filename", Cl_data_file]

        bandpower_window_file = bandpower_window_file_template.format(*idx)
        cmd += ["--bandpower-windows-filename", bandpower_window_file]

        subprocess.check_call(cmd)
