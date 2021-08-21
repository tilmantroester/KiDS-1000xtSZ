import sys
sys.path.append("tools/")
import pipeline_factory

import os


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
    XCORR_NO_M_DATA_PATH = "%(TRIAD_PATH)s/results/measurements_no_m/"


    cov_EE_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_total_SSC_mask_EEEE.txt"
    shear_shear_data_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_shear_KiDS1000/likelihood/data/Cl_EE_shear_KiDS1000_gal.txt"

    cov_TE_file_template = "%(XCORR_DATA_PATH)s/shear_KiDS1000_{}/likelihood/cov/covariance_total_SSC_mask_TETE.txt"
    cov_joint_file_template = "%(XCORR_DATA_PATH)s/shear_KiDS1000_{}/likelihood/cov/covariance_total_SSC_mask_joint.txt"
    shear_y_data_file_template = "%(XCORR_DATA_PATH)s/shear_KiDS1000_{}/likelihood/data/Cl_TE_shear_KiDS1000_{}_{}.txt"

    shear_y_bandpower_window_file_template = "%(SHEAR_Y_WINDOW_FUNCTION_DIR)s/pymaster_bandpower_windows_{}-{}.npy"
    shear_shear_bandpower_window_file_template = "%(SHEAR_SHEAR_WINDOW_FUNCTION_DIR)s/pymaster_bandpower_windows_{}-{}.npy"

    CIB_data_file = "%(XCORR_NO_M_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/likelihood/data/Cl_TE_shear_KiDS1000_gal_545GHz_CIB.txt"
    CIB_cov_file = "%(XCORR_NO_M_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/likelihood/cov/covariance_gaussian_nka_TETE.txt"
    CIB_GP_state_file = "%(XCORR_NO_M_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/GP_model/GP_state.torchstate"

    PATHS = {"CSL_PATH" : CSL_PATH, 
             "KCAP_PATH" : KCAP_PATH, 
             "HMX_PATH" : HMX_PATH,
             "TRIAD_PATH" : TRIAD_PATH,
             "SHEAR_Y_WINDOW_FUNCTION_DIR": SHEAR_Y_WINDOW_FUNCTION_DIR,
             "SHEAR_SHEAR_WINDOW_FUNCTION_DIR": SHEAR_SHEAR_WINDOW_FUNCTION_DIR,}

    DATA_DIRS = {"NZ_DATA_PATH" : NZ_DATA_PATH,
                 "XCORR_DATA_PATH" : XCORR_DATA_PATH,
                 "XCORR_NO_M_DATA_PATH": XCORR_NO_M_DATA_PATH}

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

    output_root_dir = "runs/final_runs_fast/"

    config_updates = {}
    # Fiducial
    # EE
    cov_file = cov_EE_file
    config_update = pipeline_factory.EE_only_config_update(cov_file)
    params_update = {"cib_parameters": {"alpha": 0.0}}
    config_updates["EE_fid"] = (config_update, params_update)

    # TE
    cov_file = cov_TE_file_template.format("y_milca")
    config_updates["TE_fid"] = pipeline_factory.TE_only_config_update(cov_file)

    # Joint
    config_updates["joint_fid"] = {}

    # Different y-maps
    # Planck
    # TE only, nocib marginalisation
    for map_name in ["y_milca", "y_nilc", "y_yan2019_nocib", "y_yan2019_nocib_beta1.2"]:
        cov_file = cov_TE_file_template.format(map_name)
        data_file = shear_y_data_file_template.format(map_name, "gal", map_name)

        config_update = pipeline_factory.TE_only_config_update(cov_file)
        config_update["like"].update({"shear_y_data_file": data_file})
        params_update = {"cib_parameters": {"alpha": 0.0}}

        config_updates[map_name + "_nocib_marg"] = (config_update, params_update)

    # ACT
    # TE only, nocib marginalisation
    for map_name in ["y_ACT_BN", "y_ACT_BN_nocmb", "y_ACT_BN_nocib"]:
        cov_file = cov_TE_file_template.format("cel_" + map_name)
        data_file = shear_y_data_file_template.format("cel_" + map_name, "cel", map_name[2:])

        config_update = pipeline_factory.TE_only_config_update(cov_file)
        config_update["like"].update({"shear_y_data_file": data_file})
        config_update["beam_filter_cls"] = {"fwhm": 1.6}
        params_update = {"cib_parameters": {"alpha": 0.0}}

        config_updates[map_name + "_nocib_marg"] = (config_update, params_update)
    
    # No y IA
    # TE
    cov_file = cov_TE_file_template.format("y_milca")
    config_update = pipeline_factory.TE_only_config_update(cov_file)
    config_update["like"].update({"shear_y_ia_section_name": None})
    config_updates["TE_no_y_IA"] = config_update

    # TE
    config_update = {"like": {"shear_y_ia_section_name": None}}
    config_updates["joint_no_y_IA"] = config_update

    # No TETE SSC
    cov_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_no_SSC_TETE.txt"
    config_updates["TE_no_SSC"] = pipeline_factory.TE_only_config_update(cov_file)


    for config_name, config_update in config_updates.items():
        if isinstance(config_update, tuple):
            config_update, params_update = config_update
        else:
            params_update = {}

        config.reset_config()
        config.reset_params()
        config.update_config(config_update)
        config.update_params(params_update)

        sampler = "multinest"
        run_name = config_name

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
