import os
import glob
import pickle
import argparse
import types

import numpy as np

KCAP_PATH = "/home/ttroester/Research/KiDS/kcap/"
# KCAP_PATH = "/Users/yooken/Research/KiDS/kcap/"

import sys
sys.path.append(os.path.join(KCAP_PATH, "kcap"))
import cosmosis_utils

sys.path.append(os.path.join(KCAP_PATH, "utils"))
import process_chains


class tSZPipelineFactory(cosmosis_utils.CosmoSISPipelineFactory):
    def create_base_config(self,
                           source_nz_files,
                           dz_covariance_file,
                           source_nz_sample,
                           y_filter_fwhm,
                           nside,
                           data_covariance_file,
                           shear_y_data_file,
                           shear_shear_data_file,
                           shear_y_bandpower_window_file_template,
                           shear_shear_bandpower_window_file_template,
                           CIB_data_covariance_file,
                           CIB_data_file,
                           CIB_GP_state_file):
        fields = ["matter", "pressure"]
        power_spectra_section = ["matter_pressure_power_spectrum", 
                                 "pressure_pressure_power_spectrum"]
        
        config = {  #"sample_ln_As"       : {"file" : "%(KCAP_PATH)s/utils/sample_ln_As.py",},
                                                
                    "sample_S8"       : {"file" : "%(KCAP_PATH)s/utils/sample_S8.py",
                                         "S8_name"    : "S_8_input"},
                    

                    "sigma8toAs"       : {"file" : "%(KCAP_PATH)s/utils/sigma8toAs.py",},
            
                    "correlate_dz"       : {"file" : "%(KCAP_PATH)s/utils/correlated_priors.py",
                                            "uncorrelated_parameters" : "nofz_shifts/p_1 nofz_shifts/p_2 nofz_shifts/p_3 nofz_shifts/p_4 nofz_shifts/p_5",
                                            "output_parameters"       : "nofz_shifts/bias_1 nofz_shifts/bias_2 nofz_shifts/bias_3 nofz_shifts/bias_4 nofz_shifts/bias_5",
                                            "covariance"              : dz_covariance_file,
                                           },

                    "camb"               : {"file" : "%(CSL_PATH)s/boltzmann/pycamb/camb_interface.py",
                                            "do_reionization"    : False,
                                            "mode"               : "transfer",
                                            "nonlinear"          : "none",
                                            "neutrino_hierarchy" : "normal",
                                            "kmax"               : 20.0,
                                            "zmid"               : 1.5,
                                            "nz_mid"             : 8,
                                            "zmax"               : 6.0,
                                            "nz"                 : 11,
                                            "background_zmax"    : 6.0,
                                            "background_zmin"    : 0.0,
                                            "background_nz"      : 6000,
                                            },
                  
                    "hmx"               :  {"file" : "%(HMX_PATH)s/cosmosis_interface.py",
                                            "mode"   : "HMx2020_matter_pressure",
                                            "fields" : " ".join(fields),
                                            "verbose" : 0,
                                           },

                    "extrapolate_power" :  {"file" : "%(CSL_PATH)s/boltzmann/extrapolate/extrapolate_power.py",
                                            "kmax" : 500.0,
                                            "sections" : " ".join(power_spectra_section)},

                                            
                    "load_source_nz" :     {"file" : "%(CSL_PATH)s/number_density/load_nz/load_nz.py",
                                            "filepath" : " ".join(source_nz_files),
                                            "histogram" : True,
                                            "output_section" : f"nz_{source_nz_sample}"},
            
                    "source_photoz_bias" : {"file" : "%(CSL_PATH)s/number_density/photoz_bias/photoz_bias.py",
                                            "mode" : "additive",
                                            "sample" : f"nz_{source_nz_sample}",
                                            "bias_section" : "nofz_shifts",
                                            "interpolation" : "cubic",
                                            "output_deltaz" : True,
                                            "output_section_name" :  "delta_z_out"},
        
                    "linear_alignment" :   {"file" : "%(CSL_PATH)s/intrinsic_alignments/la_model/linear_alignments_interface.py",
                                            "method" : "bk_corrected",
                                            "X_matter_power_section" : "matter_pressure_power_spectrum",
                                            "X_intrinsic_power_output_section" : "pressure_intrinsic_power_spectrum",},

                    "projection" :         {"file" : "%(CSL_PATH)s/structure/projection/project_2d.py",
                                            "ell_min" : 2.0,
                                            "ell_max" : 6144.0,
                                            "n_ell"  : 400,
                                            "fast-shear-shear-ia" : f"{source_nz_sample}-{source_nz_sample}",
                                            "shear-y" : f"{source_nz_sample}-y",
                                            "intrinsic-y" : f"{source_nz_sample}-y",
                                            "verbose" : False,
                                            "get_kernel_peaks" : False},
            
                    "cib_contamination" :   {"file" : "%(TRIAD_PATH)s/tools/CIB_contamination.py",
                                             "cov_file" : CIB_data_covariance_file,
                                             "data_file" : CIB_data_file,
                                             "gp_state_file": CIB_GP_state_file,
                                             "CIB_section" : "shear_CIB_cl",
                                             "tSZ_CIB_contamination_section" : "shear_y_CIB_contamination_cl",},
            
                    "beam_filter_cls" :    {"file" : "%(TRIAD_PATH)s/tools/filter_cls.py",
                                            "filter"            : "gaussian",
                                            "fwhm"              : y_filter_fwhm,
                                            "sections"          : "shear_y_cl y_y_cl intrinsic_y_cl",
                                            "powers"            : [1,2,1],
                                            "new_section_suffix" : "beam"},
            
                    "pixwin_filter_cls" :  {"file" : "%(TRIAD_PATH)s/tools/filter_cls.py",
                                            "filter"            : "healpix_window",
                                            "nside"              : nside,
                                            "sections"          : "shear_y_cl_beam y_y_cl_beam intrinsic_y_cl_beam",
                                            "powers"            : [1, 2, 1],
                                            "new_section_suffix" : "pixwin"},
            
                    "like" :                {"file" : "%(TRIAD_PATH)s/tools/shear_tsz_like.py",
                                             "like_name" : "shear_y_like",
                                             "probes" : "shear_y shear_shear",
                                             "n_z_bin": 5,

                                             "cov_file" : data_covariance_file,

                                             "shear_y_data_file" : shear_y_data_file,
                                             "shear_y_bandpower_windows" : shear_y_bandpower_window_file_template,
                                             "shear_y_section_name" : "shear_y_cl_beam_pixwin",
                                             "shear_y_ia_section_name" : "intrinsic_y_cl_beam_pixwin",
                                             "shear_y_cib_contamination_section_name" : "shear_y_CIB_contamination_cl_binned",
                                             "shear_y_ell_range" : [100, 1500],

                                             "shear_shear_data_file" : shear_shear_data_file,
                                             "shear_shear_bandpower_windows" : shear_shear_bandpower_window_file_template,
                                             "shear_shear_ell_range" : [100, 1500],
                                             }

        }
        return config
    
    @property
    def base_config_data_files(self):
        return [("correlate_dz", "covariance"),
                ("load_source_nz", "filepath"),
                ("cib_contamination", "data_file"),
                ("cib_contamination", "cov_file"),
                ("cib_contamination", "gp_state_file"),
                ("like", "shear_y_data_file"),
                ("like", "shear_shear_data_file"),
                ("like", "cov_file")]
    
    def create_base_params(self, p=None):
        if p is None:
            # Use 3x2pt MAP
            p = {"omegach2":  0.123355,
                 "omegabh2":  0.022492,
                 "h":         0.695021,
                 "ns":        0.901206,
                 "s8proxy":   0.763871,
                 "logt_heat": 7.8,
                 "a_ia":      1.066324,
                 "alpha_cib": 2.3e-7,
                 "p_z1":       0.191855,
                 "p_z2":       0.875148,
                 "p_z3":      -2.117530,
                 "p_z4":      -1.637583,
                 "p_z5":       1.360465}
        
        params = {"cosmological_parameters":
                        {"omch2":     [ 0.051,  p["omegach2"],      0.255],
                         "ombh2":     [ 0.019,  p["omegabh2"],    0.026],
                         "h0":        [ 0.64,   p["h"],       0.82],
                         "n_s":       [ 0.84,   p["ns"],      1.1],
                         "S_8_input": [ 0.1,    p["s8proxy"],    1.3],
                         #"ln_1e10_A_s" : [ 1.5,    2.72,      4.0],
                         "omega_k":           0.0,
                         "w":                -1.0,
                         "mnu":               0.06,  # normal hierarchy
                        }, 
                  "halo_model_parameters":
                    {"log10_Theat": [ 7.0,    p["logt_heat"],       8.5]},

                  "intrinsic_alignment_parameters":
                    {"A":           [-6.0,    p["a_ia"],       6.0]},

                  "cib_parameters":
                    {"alpha":       [-1.0e-5, p["alpha_cib"],    1.0e-5]},

                  "nofz_shifts":
                    {"p_1":         [-5.0,   p["p_z1"],       5.0],
                     "p_2":         [-5.0,   p["p_z2"],       5.0],
                     "p_3":         [-5.0,   p["p_z3"],       5.0],
                     "p_4":         [-5.0,   p["p_z4"],       5.0],
                     "p_5":         [-5.0,   p["p_z5"],       5.0],}
                 }
        return params
    
    def create_base_priors(self):
        priors = {"nofz_shifts": {"p_1": "gaussian 0.0 1.0",
                                  "p_2": "gaussian -0.181245922114715 1.0",
                                  "p_3": "gaussian -1.110137141577241 1.0",
                                  "p_4": "gaussian -1.395013267869414 1.0",
                                  "p_5": "gaussian 1.2654336759290095 1.0"},
                 "cib_parameters": {"alpha": "gaussian 2.3e-07 13.2e-7"}}
        return priors

def make_TE_only(config, TE_cov_file):
    config_update = {"projection": {"fast-shear-shear-ia": None},
                     "like":       {"probes": "shear_y",
                                    "cov_file": TE_cov_file}}
    config.reset_config()
    config.update_config(config_update)

def make_EE_only(config, EE_cov_file):
    config_update = {"hmx":               {"fields": "matter"},
                     "projection":        {"shear-y" : None,
                                           "intrinsic-y" : None},
                     "cib_contamination": None,
                     "beam_filter_cls":   {"sections" : "", "powers" : ""},
                     "pixwin_filter_cls": {"sections" : "", "powers" : ""},
                     "like":              {"probes": "shear_shear",
                                           "cov_file": EE_cov_file}}
    config.reset_config()
    config.update_config(config_update)


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

    config = tSZPipelineFactory(options=dict(source_nz_files=nz_files, 
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
            
        start_points = []
        for i, idx in enumerate(logpost_sort_idx[:n_start_point]):
            p = {n: param_dict[n][idx] for n in param_dict.keys()}
            params_update = config.create_base_params(p)
            config.reset_config()
            config.update_params(params_update)

            print(f"Logpost for idx {i}:", p["logpost"])
            if False: # i == 0:
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