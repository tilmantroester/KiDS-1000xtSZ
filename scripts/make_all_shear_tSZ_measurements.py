import subprocess
import os


if __name__ == "__main__":
    galactic_coordinates = True

    bin_operator_file = "../data/xcorr/bin_operator_log_n_bin_12_ell_51-2952_namaster.txt"      # noqa: E501

    workspace_file_template = ("/disk09/ttroester/project_triad/namaster_workspaces/"  # noqa: E501
                               "shear_KiDS1000_y_milca/"
                               "pymaster_workspace_shear_{}_y_{}.fits")

    bandpower_window_file_template = ("../results/measurements/shear_KiDS1000_y_milca/"  # noqa: E501
                                      "data/pymaster_bandpower_windows_{}-{}.npy")              # noqa: E501

    Cl_data_file_template = ("../results/measurements/shear_KiDS1000_y_milca/"           # noqa: E501
                             "data/Cl_gal_{}-{}.npz")

    Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_y_milca/"            # noqa: E501
                            "cov_Cls/Cl_cov_CCL_gal_{}-{}.npz")

    catalog_files = ["../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz",     # noqa: E501
                     "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz"]     # noqa: E501

    if not galactic_coordinates:
        catalog_files = ["../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2.npz"]     # noqa: E501

    foreground_map = "../data/y_maps/polspice/milca/triplet.fits"
    foreground_mask = "../data/y_maps/polspice/milca/singlet_mask.fits"

    raw_ell_file = "../runs/cov_theory_predictions_run0/output/data_block/shear_y_cl_beam_pixwin/ell.txt"  # noqa: E501

    raw_Cl_files = ["../runs/cov_theory_predictions_run0/output/data_block/shear_y_cl_beam_pixwin/bin_1_1.txt",  # noqa: E501
                    "../runs/cov_theory_predictions_run0/output/data_block/shear_y_cl_beam_pixwin/bin_2_1.txt",  # noqa: E501
                    "../runs/cov_theory_predictions_run0/output/data_block/shear_y_cl_beam_pixwin/bin_3_1.txt",  # noqa: E501
                    "../runs/cov_theory_predictions_run0/output/data_block/shear_y_cl_beam_pixwin/bin_4_1.txt",  # noqa: E501
                    "../runs/cov_theory_predictions_run0/output/data_block/shear_y_cl_beam_pixwin/bin_5_1.txt",  # noqa: E501
                    ]

    os.environ["OMP_NUM_THREADS"] = "44"

    field_idx = [(i, 0) for i in range(len(catalog_files))]

    for idx in field_idx:
        print("Bin combination: ", idx)
        print()

        cmd = ["python", "namaster_cosmic_shear_measurements.py"]
        cmd += ["--bin-operator", bin_operator_file]

        cmd += ["--Cl-signal-files", raw_Cl_files[idx[0]]]
        cmd += ["--Cl-signal-ell-file", raw_ell_file]

        cmd += ["--shear-catalogs", catalog_files[idx[0]]]

        cmd += ["--foreground-map", foreground_map]
        cmd += ["--foreground-mask", foreground_mask]

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

        if galactic_coordinates:
            cmd += ["--no-flip-e1"]

        subprocess.check_call(cmd)