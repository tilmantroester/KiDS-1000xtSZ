import os
import glob
import pickle
import argparse
import types

import numpy as np

KCAP_PATH = "/home/ttroester/Research/KiDS/kcap/"
# KCAP_PATH = "/Users/yooken/Research/KiDS/kcap/"

import sys
sys.path.append("tools/")
import pipeline_factory

sys.path.append(os.path.join(KCAP_PATH, "utils"))
import process_chains


if __name__ == "__main__":
    CSL_PATH = "%(KCAP_PATH)s/cosmosis-standard-library"
    HMX_PATH = "%(KCAP_PATH)s/../HMx/python_interface/"
    TRIAD_PATH = "./"

    SHEAR_Y_WINDOW_FUNCTION_DIR = "%(TRIAD_PATH)s/results/measurements/shear_KiDS1000_y_milca/data/"
    SHEAR_SHEAR_WINDOW_FUNCTION_DIR = "%(TRIAD_PATH)s/results/measurements/shear_KiDS1000_shear_KiDS1000/data/"

    NZ_DATA_PATH = "%(KCAP_PATH)s/../Cat_to_Obs_K1000_P1/data/kids/nofz/"
    nz_files = [f"%(NZ_DATA_PATH)s/SOM_N_of_Z/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO{i}_Nz.asc" for i in range(1,6)]
    nz_cov_file = "%(NZ_DATA_PATH)s/SOM_cov_multiplied.asc"

    XCORR_DATA_PATH = "%(TRIAD_PATH)s/results/measurements/"
    cov_EE_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_shear_KiDS1000/likelihood/cov/covariance_total_SSC_mask_EEEE.txt"
    cov_TE_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_SSC_mask_TETE.txt"
    cov_joint_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_y_milca/likelihood/cov/covariance_total_SSC_mask_joint.txt"
    shear_y_data_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_y_milca/likelihood/data/Cl_TE_shear_KiDS1000_gal_y_milca.txt"
    shear_shear_data_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_shear_KiDS1000/likelihood/data/Cl_EE_shear_KiDS1000_gal.txt"

    shear_y_bandpower_window_file_template = "%(SHEAR_Y_WINDOW_FUNCTION_DIR)s/pymaster_bandpower_windows_{}-{}.npy"
    shear_shear_bandpower_window_file_template = "%(SHEAR_SHEAR_WINDOW_FUNCTION_DIR)s/pymaster_bandpower_windows_{}-{}.npy"

    CIB_data_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/likelihood/data/Cl_TE_shear_KiDS1000_gal_545GHz_CIB.txt"
    CIB_cov_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/likelihood/cov/covariance_gaussian_nka_TETE.txt"
    CIB_GP_state_file = "%(XCORR_DATA_PATH)s/shear_KiDS1000_545GHz_CIB/GP_model/GP_state.torchstate"

    PATHS = {"CSL_PATH" : CSL_PATH, 
             "KCAP_PATH" : KCAP_PATH, 
             "HMX_PATH" : HMX_PATH,
             "TRIAD_PATH" : TRIAD_PATH,
             "SHEAR_Y_WINDOW_FUNCTION_DIR": SHEAR_Y_WINDOW_FUNCTION_DIR,
             "SHEAR_SHEAR_WINDOW_FUNCTION_DIR": SHEAR_SHEAR_WINDOW_FUNCTION_DIR,}

    DATA_DIRS = {"NZ_DATA_PATH" : NZ_DATA_PATH,
                 "XCORR_DATA_PATH" : XCORR_DATA_PATH}

    config = pipeline_factory.tSZPipelineFactory(options=dict(source_nz_files=nz_files, 
                                dz_covariance_file=nz_cov_file, 
                                source_nz_sample="KiDS1000",
                                y_filter_fwhm=10.0, nside=2048,
                                data_covariance_file=cov_joint_file,
                                shear_y_data_file=shear_y_data_file,
                                shear_shear_data_file=shear_shear_data_file,
                                shear_y_bandpower_window_file_template=shear_y_bandpower_window_file_template,
                                shear_shear_bandpower_window_file_template=shear_shear_bandpower_window_file_template,
                                CIB_data_covariance_file=CIB_cov_file,
                                CIB_data_file=CIB_data_file,
                                CIB_GP_state_file=CIB_GP_state_file))


    # config.run_pipeline(defaults={**PATHS, **DATA_DIRS})
    # make_EE_only(config, cov_TE_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("--chain-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--n-ppd")
    parser.add_argument("--n-MAP-start-points")

    parser.add_argument("--EE-chain", action="store_true")
    parser.add_argument("--TE-chain", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    n_z = 5
    n_ell_bin = 12

    field_idx_EE = [(i, j) for i in range(n_z)
                    for j in range(i+1)]
    field_idx_TE = [(i, 0) for i in range(n_z)]

    if args.n_ppd is not None:
        print("Computing PPDs")
        print("Output directory:", args.output_dir)

        n_ppd = int(args.n_ppd)

        dummy_chain = types.SimpleNamespace()
        dummy_chain.ranges = None
        dummy_chain.name_tag = "none"
        dummy_chain.chain_def = {"root_dir": args.chain_dir}
        
        equal_weight_chain = process_chains.load_equal_weight_chain(dummy_chain)

        n_sample = equal_weight_chain.samples.shape[0]
        ppd_idx = np.random.choice(n_sample, size=n_ppd)

        param_names = [n.name for n in equal_weight_chain.getParamNames().names]
        ppd_params = equal_weight_chain.getParamSampleDict(ppd_idx)

        np.savez(os.path.join(args.output_dir, "ppd_params.npz"), **ppd_params)
        
        ppd_blocks = []
        tpd_Cls = {"EE": [], "TE": []}
        ppd_Cls = {"EE": [], "TE": []}
        for i in range(n_ppd):
            p = {n: ppd_params[n][i] for n in ppd_params.keys()}
            params_update = config.create_base_params(p)
            config.reset_config()
            config.update_params(params_update)
            block = config.run_pipeline(defaults={**PATHS, **DATA_DIRS})

            loglike_input = p["loglike"]
            loglike_output = block["likelihoods", "shear_y_like_like"]
            if not np.isclose(loglike_input, loglike_output):
                raise RuntimeError(f"Log like mismatch on iteration {i}: {loglike_input} vs {loglike_output}")

            ppd_blocks.append(block)

            Cls = []
            for idx in field_idx_EE:
                Cls.append(block["shear_cl_binned", f"bin_{idx[0]+1}_{idx[1]+1}"])
            tpd_Cls["EE"].append(np.array(Cls).T)
            Cls = []
            for idx in field_idx_TE:
                Cls.append(block["shear_y_cl_beam_pixwin_binned", f"bin_{idx[0]+1}_{idx[1]+1}"])
            tpd_Cls["TE"].append(np.array(Cls).T)

            data_vector_sample = block["data_vector",
                                    "shear_y_like_sim_data_vector_unmasked"]
            d_EE = data_vector_sample[:len(field_idx_EE)*n_ell_bin].reshape(-1, n_ell_bin).T
            d_TE = data_vector_sample[len(field_idx_EE)*n_ell_bin:].reshape(-1, n_ell_bin).T
            ppd_Cls["EE"].append(d_EE)
            ppd_Cls["TE"].append(d_TE)

            np.savez(os.path.join(args.output_dir, "tpd_Cls.npz"), **{k: np.array(d) for k, d in tpd_Cls.items()})
            np.savez(os.path.join(args.output_dir, "ppd_Cls.npz"), **{k: np.array(d) for k, d in ppd_Cls.items()})
        
        with open(os.path.join(args.output_dir, "blocks.pickle"), "wb") as f:
            pickle.dump(ppd_blocks, f)
    
    elif args.n_MAP_start_points is not None:
        print("Creating MAP starting point runs")

        chain_run_name = os.path.split(args.chain_dir)[1]
        output_root_dir = os.path.join(args.output_dir, "MAP_" + chain_run_name)
        print("Output directory:", output_root_dir)

        os.makedirs(output_root_dir, exist_ok=True)

        n_start_point = int(args.n_MAP_start_points)

        chain_file = glob.glob(os.path.join(args.chain_dir, "output/samples_*.txt"))[0]
        chain = process_chains.load_chain(chain_file, ignore_inf=True)

        param_names = [n.name for n in chain.getParamNames().names]
        param_dict = chain.getParamSampleDict(ix=np.arange(chain.samples.shape[0]))

        logpost_sort_idx = np.argsort(param_dict["logpost"])[::-1]
        max_logpost_idx = logpost_sort_idx[0]

        chain_specifig_params_update = None
        params = None
        config_update = {}
        if args.EE_chain:
            cov_file = cov_EE_file
            config_update = pipeline_factory.EE_only_config_update(cov_file)
            chain_specifig_params_update = {"cib_parameters": {"alpha": 0.0}}
            params = {"alpha_cib": 0.0}
        if args.TE_chain:
            cov_file = cov_TE_file
            config_update = pipeline_factory.TE_only_config_update(cov_file)
            
        start_points = []
        for i, idx in enumerate(logpost_sort_idx[:n_start_point]):
            p = {n: param_dict[n][idx] for n in param_dict.keys()}
            if params is not None:
                p.update(params)
            params_update = config.create_base_params(p)
            if chain_specifig_params_update is not None:
                params_update.update(chain_specifig_params_update)
            config.reset_config()
            config.update_config(config_update)
            config.update_params(params_update)

            print(f"Logpost for idx {i}:", p["logpost"])
            if False: #i == 0:
                print("Checking logpost from new config")
                block = config.run_pipeline(defaults={**PATHS, **DATA_DIRS})
                loglike_input = p["loglike"]
                loglike_output = block["likelihoods", "shear_y_like_like"]
                if not np.isclose(loglike_input, loglike_output):
                    raise RuntimeError(f"Log like mismatch on iteration {i}: {loglike_input} vs {loglike_output}")
            
            sampler = "maxlike"
            run_name = f"MAP_start_idx_{i}"

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
    else:
        raise ValueError("Either --n-ppd or --n-MAP-start-points nee to be specified")
    # sampler = "test"
    # run_name = f"test_ppd_setup"

    # config.add_sampling_config(likelihood_name="shear_y_like",
    #                         sampling_options=dict(verbose=True,
    #                         debug=False,
    #                         sampler_name=sampler,
    #                         run_name=run_name))

    # root_dir = os.path.join("../runs/", run_name)
    # config.stage_files(root_dir=root_dir, 
    #                 defaults={"RUN_NAME": run_name, 
    #                             "ROOT_DIR": f"runs/%(RUN_NAME)s/",
    #                             **PATHS},
    #                 data_file_dirs=DATA_DIRS, copy_data_files=True)