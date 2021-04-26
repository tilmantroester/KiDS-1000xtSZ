import os

from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors

import numpy as np
import scipy.interpolate


def log_interpolate(x_arr, y_arr, x):
    if np.any(y_arr <= 0):
        return scipy.interpolate.InterpolatedUnivariateSpline(np.log(x_arr),
                                                              y_arr,
                                                              ext=2)(np.log(x))
    else:
        return np.exp(scipy.interpolate.InterpolatedUnivariateSpline(
                    np.log(x_arr), np.log(y_arr), ext=2)(np.log(x)))


def setup(options):
    like_name = options.get_string(option_section, "like_name")

    estimator = options.get_string(option_section, "estimator",
                                   default="polspice")
    print(f"Using estimator {estimator}")

    bin_operator_file = options.get_string(option_section, "bin_operator_file")

    cov_file = options.get_string(option_section, "cov_file")
    data_files = options.get_string(option_section, "data_file")

    cov = np.loadtxt(cov_file)
    data = np.loadtxt(data_files)
    if estimator == "polspice":
        bin_operator = np.loadtxt(bin_operator_file)
    elif estimator == "namaster":
        bin_operator = np.load(bin_operator_file)
        if np.count_nonzero(bin_operator[0, :, 1]) > 0:
            raise RuntimeError("Warning: bandpower operator includes "
                               "TE-TB mxing, which is getting ignored.")
        bin_operator = bin_operator[0, :, 0]
    else:
        raise ValueError(f"Unsupported estimator {estimator}")

    n_z_bin = data.shape[1]-1
    n_ell_bin = data.shape[0]

    if bin_operator.shape[0] != n_ell_bin:
        raise ValueError(f"Inconsistent data files and binning operator.")

    ell = data[:, 0]

    data_vector = np.concatenate([data[:, i] for i in range(1, n_z_bin+1)])
    data_vector_mask = np.ones(n_z_bin*n_ell_bin, dtype=bool)

    for i in range(n_z_bin):
        if options.has_value(option_section, f"ell_range_{i+1}"):
            keep_ell = options[option_section, f"ell_range_{i+1}"]
            if keep_ell == "none":
                mask = np.zeros(n_ell_bin)
            else:
                mask = (ell >= keep_ell[0]) & (ell <= keep_ell[1])
            data_vector_mask[i*n_ell_bin:(i+1)*n_ell_bin] = mask
            print(f"Masking bins {mask} of tomographic bin {i+1}")

    data_vector = data_vector[data_vector_mask]
    cov = cov[np.ix_(data_vector_mask, data_vector_mask)]
    inv_cov = np.linalg.inv(cov)

    input_section_name = options.get_string(option_section,
                                            "input_section_name", "shear_y_cl")
    ia_section_name = options.get_string(option_section,
                                         "ia_section_name", "")
    CIB_cont_section_name = options.get_string(
                                        option_section,
                                        "cib_contamination_section_name", "")
    new_section_suffix = options.get_string(option_section,
                                            "new_section_suffix", "")

    return (data_vector, inv_cov, bin_operator, n_z_bin, data_vector_mask, cov,
            input_section_name, ia_section_name, CIB_cont_section_name,
            new_section_suffix, like_name)


def execute(block, config):
    data_vector, inv_cov, bin_operator, n_z_bin, data_vector_mask, cov,\
        input_section_name, ia_section_name, CIB_cont_section_name,\
        new_section_suffix, like_name = config

    ell_raw = block[input_section_name, "ell"]
    ell_bin_op = np.arange(bin_operator.shape[1])

    do_ia = ia_section_name != ""
    do_cib = CIB_cont_section_name != ""

    if new_section_suffix != "":
        output_section_name = input_section_name + "_" + new_section_suffix
        block[output_section_name, "ell"] = bin_operator @ ell_bin_op

        if do_ia:
            ia_output_section_name = ia_section_name + "_" + new_section_suffix
            block[ia_output_section_name, "ell"] = bin_operator @ ell_bin_op

    mu = []

    for i in range(n_z_bin):
        key = f"bin_{i+1}_1"
        Cl_raw = block[input_section_name, key]

        Cl = np.zeros(len(ell_bin_op))
        Cl[2:] = log_interpolate(ell_raw, Cl_raw, ell_bin_op[2:])
        binned_Cl = bin_operator @ Cl

        if new_section_suffix != "":
            block[output_section_name, key] = binned_Cl

        if do_ia:
            Cl_raw_ia = block[ia_section_name, key]
            Cl_ia = np.zeros(len(ell_bin_op))
            Cl_ia[2:] = log_interpolate(ell_raw, Cl_raw_ia, ell_bin_op[2:])
            binned_Cl_ia = bin_operator @ Cl_ia

            binned_Cl += binned_Cl_ia

            if new_section_suffix != "":
                block[ia_output_section_name, key] = binned_Cl_ia

        if do_cib:
            binned_Cl_CIB = block[CIB_cont_section_name, key]
            binned_Cl += binned_Cl_CIB

        mu.append(binned_Cl)

    mu = np.concatenate(mu)
    mu = mu[data_vector_mask]

    r = data_vector - mu
    chi2 = r @ inv_cov @ r

    ln_like = -0.5*chi2
    block[names.data_vector, like_name+"_CHI2"] = chi2
    block[names.likelihoods, like_name+"_LIKE"] = ln_like

    return 0


def cleanup(block):
    pass
