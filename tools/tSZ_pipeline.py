KCAP_PATH = "../../KiDS/kcap/"

import os
import sys
sys.path.append(os.path.join(KCAP_PATH, "kcap"))
import cosmosis_utils


class tSZPipelineFactory(cosmosis_utils.CosmoSISPipelineFactory):
    def create_base_config(self,
                           source_nz_files,
                           dz_covariance_file,
                           source_nz_sample,
                           y_filter_fwhm,
                           nside,
                           data_covariance_file=None,
                           data_file=None,
                           CIB_data_covariance_file=None,
                           CIB_data_file=None,
                           bin_operator_file=None
                          ):
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
                                            "ell_max" : 6001.0,
                                            "n_ell"  : 400,
                                            "shear-shear" : f"{source_nz_sample}-{source_nz_sample}",
                                            "shear-y" : f"{source_nz_sample}-y",
                                            "intrinsic-y" : f"{source_nz_sample}-y",
                                            "verbose" : False,
                                            "get_kernel_peaks" : False},
            
                    "cib_contamination" :   {"file" : "%(TRIAD_PATH)s/tools/CIB_contamination.py",
                                             "cov_file" : CIB_data_covariance_file,
                                             "data_file" : CIB_data_file,
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
                                            "powers"            : [2, 2, 2],
                                            "new_section_suffix" : "pixwin"},
            
                    "tSZ_like" :            {"file" : "%(TRIAD_PATH)s/tools/tsz_like.py",
                                             "like_name" : "tsz_like",
                                             "cov_file" : data_covariance_file,
                                             "data_file" : data_file,
                                             "bin_operator_file" : bin_operator_file,
                                             "input_section_name" : "shear_y_cl_beam_pixwin",
                                             "ia_section_name" : "intrinsic_y_cl_beam_pixwin",
                                             "cib_contamination_section_name" : "shear_y_CIB_contamination_cl_binned",
                                             "new_section_suffix" : "binned"
                                             }
        }
        if data_covariance_file is None and data_file is None and bin_operator_file is None:
            del config["tSZ_like"]
        if CIB_data_file is None and CIB_data_covariance_file is None:
            del config["cib_contamination"]
        return config
    
    @property
    def base_config_data_files(self):
        return [("correlate_dz", "covariance"),
                ("load_source_nz", "filepath"),
                ("cib_contamination", "data_file"),
                ("cib_contamination", "cov_file"),
                ("tSZ_like", "data_file"),
                ("tSZ_like", "cov_file"),
                ("tSZ_like", "bin_operator_file")]
    
    def create_base_params(self):
        params = {"cosmological_parameters" :      {"omch2"       : [ 0.051,  0.123355,      0.255],
                                                    "ombh2"       : [ 0.019,  0.022492,    0.026],
                                                    "h0"          : [ 0.64,   0.695021,       0.82],
                                                    "n_s"         : [ 0.84,   0.901206,      1.1],
                                                    "S_8_input"   : [ 0.1,    0.763871,    1.3],
                                                    #"ln_1e10_A_s" : [ 1.5,    2.72,      4.0],
                                                    "omega_k"     :           0.0,
                                                    "w"           :          -1.0,
                                                    "mnu"         :           0.06,             #normal hierarchy
                                                    }, 
                "halo_model_parameters" :          {"log10_Theat" : [ 7.0,    7.8,       8.5]},

                "intrinsic_alignment_parameters" : {"A"           : [-6.0,    1.066324,       6.0]},

                "cib_parameters" :                 {"alpha" :       [-4.3e-7, 2.3e-07,    8.9e-7]},

                "nofz_shifts"                    : {"p_1"         : 0.191855,
                                                    "p_2"         : 0.875148,
                                                    "p_3"         : -2.117530,
                                                    "p_4"         : -1.637583,
                                                    "p_5"         : 1.360465,},
        #         "nofz_shifts"                    : {"p_1"         : [-5.0,    0.191855,       5.0],
        #                                             "p_2"         : [-5.0,    0.875148,       5.0],
        #                                             "p_3"         : [-5.0,   -2.117530,       5.0],
        #                                             "p_4"         : [-5.0,   -1.637583,       5.0],
        #                                             "p_5"         : [-5.0,    1.360465,       5.0],},
                }
        return params