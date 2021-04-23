KCAP_PATH = "../../KiDS/kcap/"

CSL_PATH = "%(KCAP_PATH)s/cosmosis-standard-library"
HMX_PATH = "%(KCAP_PATH)s/../HMx/python_interface/"
TRIAD_PATH = "../"

NZ_DATA_PATH = "%(KCAP_PATH)s/../Cat_to_Obs_K1000_P1/data/kids/nofz/"
nz_files = [f"%(NZ_DATA_PATH)s/SOM_N_of_Z/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO{i}_Nz.asc" for i in range(1,6)]
nz_cov_file = "%(NZ_DATA_PATH)s/SOM_cov_multiplied.asc"

XCORR_DATA_PATH = "../data/xcorr/"
bin_op_file = "%(XCORR_DATA_PATH)s/bin_operator_log_n_bin_13_ell_51-2952.txt"
bin_op_squared_file = "%(XCORR_DATA_PATH)s/bin_operator_log_n_bin_13_ell_51-2952_squared_weights.txt"

cov_file = "%(XCORR_DATA_PATH)s/shear_y_KiDS1000_milca_TE_gaussian_cNG_cov.txt"
data_file = "%(XCORR_DATA_PATH)s/shear_y_KiDS1000_milca_TE.txt"


CIB_cov_file = "%(XCORR_DATA_PATH)s/shear_CIB_KiDS1000_545_TE_jk_3.4deg2_beam10_cov.txt"
CIB_data_file = "%(XCORR_DATA_PATH)s/shear_CIB_KiDS1000_545_TE_beam10.txt"

PATHS = {"CSL_PATH" : CSL_PATH, 
         "KCAP_PATH" : KCAP_PATH, 
         "HMX_PATH" : HMX_PATH,
         "TRIAD_PATH" : TRIAD_PATH,}

DATA_DIRS = {"NZ_DATA_PATH" : NZ_DATA_PATH,
             "XCORR_DATA_PATH" : XCORR_DATA_PATH}

