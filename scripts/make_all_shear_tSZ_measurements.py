import subprocess
import os


if __name__ == "__main__":
    galactic_coordinates = False

    bin_operator_file = "../data/xcorr/bin_operator_log_n_bin_12_ell_51-2952_namaster.txt"                              # noqa: E501

    if galactic_coordinates:
        workspace_file_template = ("/disk09/ttroester/project_triad/namaster_workspaces/"                               # noqa: E501
                                   "shear_KiDS1000_y_milca/pymaster_workspace_shear_{}_foreground_{}.fits")             # noqa: E501
        bandpower_window_file_template = ("../results/measurements/shear_KiDS1000_y_milca/"                             # noqa: E501
                                          "data/pymaster_bandpower_windows_{}-{}.npy")                                  # noqa: E501
    else:
        workspace_file_template = ("/disk09/ttroester/project_triad/namaster_workspaces/"                               # noqa: E501
                                   "shear_KiDS1000_cel_y_ACT_BN/pymaster_workspace_shear_{}_foreground_{}.fits")        # noqa: E501
        bandpower_window_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN/"                        # noqa: E501
                                          "data/pymaster_bandpower_windows_{}-{}.npy")                                  # noqa: E501

    Cl_data_file_template = None
    Cl_cov_file_template = None
    foreground_beam = None

    # Planck milca
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_y_milca/"           # noqa: E501
    #                          "data/Cl_gal_{}-{}.npz")
    # Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_y_milca/"            # noqa: E501
    #                         "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    # foreground_map = "../data/y_maps/polspice/milca/triplet.fits"
    # foreground_mask = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # Planck nilc
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_y_nilc/"           # noqa: E501
    #                          "data/Cl_gal_{}-{}.npz")
    # Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_y_nilc/"            # noqa: E501
    #                         "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    # foreground_map = "../data/y_maps/Planck_processed/nilc_full.fits"
    # foreground_mask = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # Ziang's CIB deprojected map
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_y_yan2019_nocib/"           # noqa: E501
    #                          "data/Cl_gal_{}-{}.npz")
    # Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_y_yan2019_nocib/"            # noqa: E501
    #                         "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    # foreground_map = "../data/y_maps/Planck_processed/ziang/ymap_rawcov_needlet_galmasked_nomockcib_v1.02_bp.fits"  # noqa: E501
    # foreground_mask = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # Ziang's CIB deprojected map, beta=1.2
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_y_yan2019_nocib_beta1.2/"           # noqa: E501
    #                          "data/Cl_gal_{}-{}.npz")
    # Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_y_yan2019_nocib_beta1.2/"            # noqa: E501
    #                         "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    # foreground_map = "../data/y_maps/Planck_processed/ziang/ymap_rawcov_needlet_galmasked_nomockcib_v1.02_bp_beta1.2.fits"  # noqa: E501
    # foreground_mask = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # ACT BN
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN/"           # noqa: E501
    #                          "data/Cl_cel_{}-{}.npz")
    # Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN/"            # noqa: E501
    #                         "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    # foreground_map = "../data/y_maps/ACT/BN.fits"
    # foreground_mask = "../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits"
    # ACT BN nocib
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocib/"           # noqa: E501
    #                          "data/Cl_cel_{}-{}.npz")
    # Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocib/"            # noqa: E501
    #                         "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    # foreground_map = "../data/y_maps/ACT/BN_deproject_cib.fits"
    # foreground_mask = "../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits"
    # ACT BN nocmb
    Cl_data_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocmb/"           # noqa: E501
                             "data/Cl_cel_{}-{}.npz")
    Cl_cov_file_template = ("../results/measurements/shear_KiDS1000_cel_y_ACT_BN_nocmb/"            # noqa: E501
                            "cov_Cls/Cl_cov_3x2pt_MAP_{}-{}.npz")
    foreground_map = "../data/y_maps/ACT/BN_deproject_cmb.fits"
    foreground_mask = "../data/y_maps/ACT/BN_planck_ps_gal40_mask.fits"

    # Planck HFI 100 GHz
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_100GHz_HFI/"           # noqa: E501
    #                          "data/Cl_gal_{}-{}.npz")
    # foreground_map = "/disk09/ttroester/Planck/frequency_maps/HFI_SkyMap_100_2048_R3.01_full.fits"
    # foreground_mask = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"
    # Planck CIB 545 GHz
    # Cl_data_file_template = ("../results/measurements/shear_KiDS1000_545GHz_CIB/"           # noqa: E501
    #                          "data/Cl_gal_{}-{}.npz")
    # foreground_map = "../data/CIB_maps/CIB-GNILC-F545_beam10.fits"
    # foreground_mask = "../data/y_maps/Planck_processed/mask_ps_gal40.fits"

    if galactic_coordinates:
        catalog_files = ["../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3_galactic.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5_galactic.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7_galactic.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9_galactic.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2_galactic.npz"]     # noqa: E501
    else:
        catalog_files = ["../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.1-0.3.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.3-0.5.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.5-0.7.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.7-0.9.npz",     # noqa: E501
                         "../data/shear_catalogs_KiDS1000/KiDS-1000_All_z0.9-1.2.npz"]     # noqa: E501

    m_bias_old = [-0.009, -0.011, -0.015, 0.002, 0.007]
    m_bias_new = [-0.010, -0.009, -0.011, 0.008, 0.012]
    m_bias = m_bias_old

    # theory_run_name = "cov_theory_predictions_run1_hmx_nz128_beam10"
    theory_run_name = "cov_theory_predictions_run3_hmx_nocib_beam1.6"

    raw_ell_file = f"../data/xcorr/theory_predictions/{theory_run_name}/output/data_block/shear_y_cl/ell.txt"  # noqa: E501

    raw_Cl_files = [f"../data/xcorr/theory_predictions/{theory_run_name}/output/data_block/shear_y_cl_beam_pixwin/bin_1_1.txt",  # noqa: E501
                    f"../data/xcorr/theory_predictions/{theory_run_name}/output/data_block/shear_y_cl_beam_pixwin/bin_2_1.txt",  # noqa: E501
                    f"../data/xcorr/theory_predictions/{theory_run_name}/output/data_block/shear_y_cl_beam_pixwin/bin_3_1.txt",  # noqa: E501
                    f"../data/xcorr/theory_predictions/{theory_run_name}/output/data_block/shear_y_cl_beam_pixwin/bin_4_1.txt",  # noqa: E501
                    f"../data/xcorr/theory_predictions/{theory_run_name}/output/data_block/shear_y_cl_beam_pixwin/bin_5_1.txt",  # noqa: E501
                    ]

    os.environ["OMP_NUM_THREADS"] = "20"

    field_idx = [(i, 0) for i in range(len(catalog_files))]

    for idx in field_idx:
        print("Bin combination: ", idx)
        print()

        cmd = ["python", "namaster_cosmic_shear_measurements.py"]
        cmd += ["--bin-operator", bin_operator_file]

        cmd += ["--Cl-signal-files", raw_Cl_files[idx[0]]]
        cmd += ["--Cl-signal-ell-file", raw_ell_file]

        cmd += ["--shear-catalogs", catalog_files[idx[0]]]
        cmd += ["--shear-m", str(m_bias[idx[0]])]

        cmd += ["--foreground-map", foreground_map]
        cmd += ["--foreground-mask", foreground_mask]
        if foreground_beam is not None:
            cmd += ["--foreground-beam", foreground_beam]

        workspace_file = workspace_file_template.format(*idx)
        cmd += ["--pymaster-workspace", workspace_file]
        if not os.path.isfile(workspace_file):
            cmd += ["--compute-coupling-matrix"]

        cmd += ["--n-iter", "3"]
        cmd += ["--n-randoms", "0"]

        if Cl_cov_file_template is not None:
            Cl_cov_file = Cl_cov_file_template.format(*idx)
            cmd += ["--Cl-cov-filename", Cl_cov_file]

        if Cl_data_file_template is not None:
            Cl_data_file = Cl_data_file_template.format(*idx)
            cmd += ["--Cl-data-filename", Cl_data_file]

        bandpower_window_file = bandpower_window_file_template.format(*idx)
        if not os.path.isfile(bandpower_window_file):
            cmd += ["--bandpower-windows-filename", bandpower_window_file]

        if galactic_coordinates:
            cmd += ["--no-flip-e1"]

        subprocess.check_call(cmd)
