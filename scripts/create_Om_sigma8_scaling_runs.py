import sys
sys.path.append("tools/")
import pipeline_factory

import os

import numpy as np

if __name__ == "__main__":
    KCAP_PATH = "../KiDS/kcap/"

    CSL_PATH = "%(KCAP_PATH)s/cosmosis-standard-library"
    HMX_PATH = "%(KCAP_PATH)s/../HMx/python_interface/"
    TRIAD_PATH = "./"

    SHEAR_Y_WINDOW_FUNCTION_DIR = "%(TRIAD_PATH)s/results/measurements/shear_KiDS1000_y_milca/data/"
    SHEAR_SHEAR_WINDOW_FUNCTION_DIR = "%(TRIAD_PATH)s/results/measurements/shear_KiDS1000_shear_KiDS1000/data/"

    NZ_DATA_PATH = "%(KCAP_PATH)s/../Cat_to_Obs_K1000_P1/data/kids/nofz/"
    nz_files = [f"%(NZ_DATA_PATH)s/SOM_N_of_Z/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO{i}_Nz.asc" for i in range(1,6)]
    nz_cov_file = "%(NZ_DATA_PATH)s/SOM_cov_multiplied.asc"

    XCORR_DATA_PATH = "%(TRIAD_PATH)s/results/measurements/"
    XCORR_OLD_DATA_PATH = "%(TRIAD_PATH)s/results/measurements/"

    cov_EE_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_total_SSC_mask_EEEE.txt"
    shear_shear_data_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_shear_KiDS1000/likelihood/data/Cl_EE_shear_KiDS1000_gal.txt"

    cov_TE_file_template = "%(XCORR_DATA_PATH)s/shear_KiDS1000_{}/likelihood/cov/covariance_total_SSC_mask_TETE.txt"
    cov_joint_file_template = "%(XCORR_DATA_PATH)s/shear_KiDS1000_{}/likelihood/cov/covariance_total_SSC_mask_joint.txt"
    shear_y_data_file_template = "%(XCORR_DATA_PATH)s/shear_KiDS1000_{}/likelihood/data/Cl_TE_shear_KiDS1000_{}_{}.txt"

    shear_y_bandpower_window_file_template = "%(SHEAR_Y_WINDOW_FUNCTION_DIR)s/pymaster_bandpower_windows_{}-{}.npy"
    shear_shear_bandpower_window_file_template = "%(SHEAR_SHEAR_WINDOW_FUNCTION_DIR)s/pymaster_bandpower_windows_{}-{}.npy"

    CIB_data_file = "%(XCORR_OLD_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/likelihood/data/Cl_TE_shear_KiDS1000_gal_545GHz_CIB.txt"
    CIB_cov_file = "%(XCORR_OLD_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/likelihood/cov/covariance_gaussian_nka_TETE.txt"
    CIB_GP_state_file = "%(XCORR_OLD_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/GP_model/GP_state.torchstate"

    PATHS = {"CSL_PATH" : CSL_PATH, 
             "KCAP_PATH" : KCAP_PATH, 
             "HMX_PATH" : HMX_PATH,
             "TRIAD_PATH" : TRIAD_PATH,
             "SHEAR_Y_WINDOW_FUNCTION_DIR": SHEAR_Y_WINDOW_FUNCTION_DIR,
             "SHEAR_SHEAR_WINDOW_FUNCTION_DIR": SHEAR_SHEAR_WINDOW_FUNCTION_DIR,}

    DATA_DIRS = {"NZ_DATA_PATH" : NZ_DATA_PATH,
                 "XCORR_DATA_PATH" : XCORR_DATA_PATH,
                 "XCORR_OLD_DATA_PATH" : XCORR_OLD_DATA_PATH}

    config = pipeline_factory.tSZPipelineFactory(options=dict(source_nz_files=nz_files, 
                                dz_covariance_file=nz_cov_file, 
                                source_nz_sample="KiDS1000",
                                y_filter_fwhm=10.0, nside=2048,
                                data_covariance_file=cov_joint_file_template.format("y_milca"),
                                shear_y_data_file=shear_y_data_file_template.format("y_milca", "gal", "y_milca"),
                                shear_shear_data_file=shear_shear_data_file,
                                shear_y_bandpower_window_file_template=shear_y_bandpower_window_file_template,
                                shear_shear_bandpower_window_file_template=shear_shear_bandpower_window_file_template,
                                CIB_data_covariance_file=CIB_cov_file,
                                CIB_data_file=CIB_data_file,
                                CIB_GP_state_file=CIB_GP_state_file))

    # output_root_dir = "runs/theory_prediction_runs/vary_sigma8_omegam/"
    output_root_dir = "runs/theory_prediction_runs/vary_logt_heat/"
    # output_root_dir = "runs/theory_prediction_runs/vary_a_ia/"

    config_update = {"sample_S8": None}

    sigma8 = [0.5, 0.6, 0.7, 0.8, 0.9]
    omegam = [0.15, 0.25, 0.35, 0.45, 0.55]
    sigma8_fid = sigma8[2]
    omegam_fid = omegam[2]

    sigma8 = [0.6, 0.65, 0.7, 0.75]
    omegam = [0.1, 0.65]

    log_theat = [7.4, 7.6, 7.8, 8.0, 8.2]
    log_theat = [7.2, 8.4]

    a_ia = [-1.0, -0.0, 1.0, 2.0, 3.0]

    sigma8_fid = 0.7
    omegam_fid = 0.35

    ombh2_fid = 0.022492
    h_fid = 0.695021
    omegab_fid = ombh2_fid/h_fid**2
    omch2_fid = (omegam_fid - omegab_fid) * h_fid**2

    params_updates = {}

    # for p in sigma8:
    #     params_update = {"cosmological_parameters": {"omch2": omch2_fid,
    #                                                  "sigma_8_input": p}}
    #     params_updates[(p, omegam_fid)] = params_update

    # for p in omegam:
    #     omch2 = (p - omegab_fid) * h_fid**2
    #     params_update = {"cosmological_parameters": {"omch2": omch2,
    #                                                  "sigma_8_input": sigma8_fid}}
    #     params_updates[(sigma8_fid, p)] = params_update

    for p in log_theat:
        params_update = {"cosmological_parameters": {"omch2": omch2_fid,
                                                     "sigma_8_input": sigma8_fid},
                         "halo_model_parameters": {"log10_Theat": p}}
        params_updates[p] = params_update

    # for p in a_ia:
    #     params_update = {"cosmological_parameters": {"omch2": omch2_fid,
    #                                                  "sigma_8_input": sigma8_fid},
    #                      "intrinsic_alignment_parameters": {"A": p}}
    #     params_updates[p] = params_update

    for name, params_update in params_updates.items():
        config.reset_config()
        config.reset_params()
        config.update_config(config_update)
        config.update_params(params_update)

        sampler = "test"
        # run_name = f"sigma8_{name[0]}_omegam_{name[1]}"
        run_name = f"logt_heat_{name}"
        # run_name = f"a_ia_{name}"

        config.add_sampling_config(likelihood_name="shear_y_like",
                                   sampling_options=dict(verbose=True,
                                   debug=False,
                                   sampler_name=sampler,
                                   run_name=run_name))

        root_dir = output_root_dir
        config.stage_files(root_dir=os.path.join(root_dir, run_name), 
                            defaults={"RUN_NAME": run_name, 
                                        "ROOT_DIR": os.path.join(root_dir, "%(RUN_NAME)s/"),
                                        **PATHS},
                            data_file_dirs=DATA_DIRS, copy_data_files=True)
