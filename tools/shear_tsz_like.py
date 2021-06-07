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


def create_data_vector_mask(ell, indices, options, prefix, tag_template):
    tag = f"{prefix}_ell_range"
    if options.has_value(option_section, tag):
        keep_ell = options[option_section, tag]
        global_mask = (ell >= keep_ell[0]) & (ell <= keep_ell[1])
    else:
        global_mask = np.ones_like(ell, dtype=bool)

    masks = []
    for idx in indices:
        tag = tag_template.format(idx=idx)
        if options.has_value(option_section, tag):
            keep_ell = options[option_section, tag]
            if keep_ell == "none":
                mask = np.zeros_like(ell, dtype=bool)
            else:
                mask = (ell >= keep_ell[0]) & (ell <= keep_ell[1])
            mask = mask & global_mask
        else:
            mask = global_mask
        masks.append(mask)
        mask_str = " ".join(["1" if m else "0" for m in mask])
        print(f"Masking bins {mask_str} of tomographic bin {idx}")

    return np.concatenate(masks)


def setup(options):
    like_name = options.get_string(option_section, "like_name")

    probes = options.get_string(option_section, "probes").split(" ")

    do_cosmic_shear = "shear_shear" in probes
    do_shear_y = "shear_y" in probes or "y_shear" in probes

    if do_cosmic_shear:
        shear_shear_bandpower_windows_template = \
            options.get_string(option_section, "shear_shear_bandpower_windows")
    if do_shear_y:
        shear_y_bandpower_windows_template = \
            options.get_string(option_section, "shear_y_bandpower_windows")

    n_z_bin = options.get_int(option_section, "n_z_bin")

    cov_file = options.get_string(option_section, "cov_file")

    window_functions = {}
    indices = {}

    if do_cosmic_shear:
        indices["shear_shear"] = [(i, j) for i in range(n_z_bin)
                                  for j in range(i+1)]

        shear_shear_data_file = options.get_string(option_section,
                                                   "shear_shear_data_file")
        shear_shear_data = np.loadtxt(shear_shear_data_file)
        shear_shear_ell = shear_shear_data[:, 0]
        shear_shear_data_vector = np.hstack(shear_shear_data[:, 1:].T)

        shear_shear_data_vector_mask = \
            create_data_vector_mask(
                        ell=shear_shear_ell,
                        indices=indices["shear_shear"],
                        options=options,
                        prefix="shear_shear",
                        tag_template="shear_shear_ell_range_{idx[0]}_{idx[1]}")
        shear_shear_data_vector = \
            shear_shear_data_vector[shear_shear_data_vector_mask]

        window_functions["shear_shear"] = {}
        for i, (idx_1, idx_2) in enumerate(indices["shear_shear"]):
            filename = (shear_shear_bandpower_windows_template
                        .format(idx_1, idx_2))
            window_functions["shear_shear"][(idx_1, idx_2)] = np.load(filename)

    if do_shear_y:
        indices["shear_y"] = list(range(n_z_bin))
        shear_y_data_file = options.get_string(option_section,
                                               "shear_y_data_file")
        shear_y_data = np.loadtxt(shear_y_data_file)
        shear_y_ell = shear_y_data[:, 0]
        shear_y_data_vector = np.hstack(shear_y_data[:, 1:].T)

        shear_y_data_vector_mask = \
            create_data_vector_mask(
                        ell=shear_y_ell,
                        indices=indices["shear_y"],
                        options=options,
                        prefix="shear_y",
                        tag_template="shear_y_ell_range_{idx}")
        shear_y_data_vector = \
            shear_y_data_vector[shear_y_data_vector_mask]

        window_functions["shear_y"] = {}
        for i, idx in enumerate(indices["shear_y"]):
            filename = shear_y_bandpower_windows_template.format(idx, 0)
            window_functions["shear_y"][idx] = np.load(filename)

    cov = np.loadtxt(cov_file)

    data_vector_mask = []
    data_vector = []
    if do_cosmic_shear:
        data_vector.append(shear_shear_data_vector)
        data_vector_mask.append(shear_shear_data_vector_mask)
    if do_shear_y:
        data_vector.append(shear_y_data_vector)
        data_vector_mask.append(shear_y_data_vector_mask)

    data_vector = np.concatenate(data_vector)
    data_vector_mask = np.concatenate(data_vector_mask)

    if cov.shape[0] != len(data_vector_mask):
        raise RuntimeError("Missmatch in covariance and mask shape")

    cov = cov[np.ix_(data_vector_mask, data_vector_mask)]
    inv_cov = np.linalg.inv(cov)

    if cov.shape[0] != len(data_vector):
        raise RuntimeError("Missmatch in covariance and data vector shape")

    section_names = {}
    if do_cosmic_shear:
        section_names["shear_shear"] = "shear_cl"
    if do_shear_y:
        section_names["shear_y"] = options.get_string(
                                        option_section,
                                        "shear_y_section_name", "shear_y_cl")
        section_names["shear_y_ia"] = options.get_string(
                                        option_section,
                                        "shear_y_ia_section_name", "")
        section_names["shear_y_cib"] = options.get_string(
                                    option_section,
                                    "shear_y_cib_contamination_section_name",
                                    "")

    binned_suffix = options.get_string(option_section,
                                       "binned_section_suffix", "")

    return (data_vector, inv_cov, window_functions,
            indices, do_cosmic_shear, do_shear_y,
            n_z_bin, data_vector_mask, cov,
            section_names, binned_suffix, like_name)


def execute(block, config):
    (data_vector, inv_cov, window_functions,
        indices, do_cosmic_shear, do_shear_y,
        n_z_bin, data_vector_mask, cov,
        section_names, binned_suffix, like_name) = config

    mu = []
    if do_cosmic_shear:
        ell_data_block = block[section_names["shear_shear"], "ell"]
        for idx_1, idx_2 in indices["shear_shear"]:
            B = window_functions["shear_shear"][(idx_1, idx_2)]
            ell = np.arange(B.shape[-1])

            Cl_data_block = block[section_names["shear_shear"],
                                  f"bin_{idx_1+1}_{idx_2+1}"]
            Cl_EE = np.zeros(len(ell))
            Cl_0 = np.zeros_like(Cl_EE)
            Cl_EE[2:] = log_interpolate(ell_data_block, Cl_data_block, ell[2:])

            Cl_EE_binned = np.einsum("ibjl,jl->ib",
                                     B, [Cl_EE, Cl_0, Cl_0, Cl_0])[0]
            mu.append(Cl_EE_binned)

    if do_shear_y:
        do_shear_y_ia = section_names["shear_y_ia"] != ""
        do_cib = section_names["shear_y_cib"] != ""

        ell_data_block = block[section_names["shear_y"], "ell"]
        for idx in indices["shear_y"]:
            tag = f"bin_{idx+1}_1"

            B = window_functions["shear_y"][idx]
            ell = np.arange(B.shape[-1])

            Cl_data_block = block[section_names["shear_y"], tag]
            Cl_TE = np.zeros(len(ell))
            Cl_0 = np.zeros_like(Cl_TE)
            Cl_TE[2:] = log_interpolate(ell_data_block, Cl_data_block, ell[2:])

            if do_shear_y_ia:
                Cl_IA_data_block = block[section_names["shear_y_ia"], tag]
                Cl_IA = np.zeros_like(ell)
                Cl_IA[2:] = log_interpolate(ell_data_block, Cl_IA_data_block,
                                            ell[2:])
                Cl_TE += Cl_IA

            Cl_TE_binned = np.einsum("ibjl,jl->ib",
                                     B, [Cl_TE, Cl_0])[0]

            if do_cib:
                Cl_CIB = block[section_names["shear_y_cib"], tag]
                Cl_TE_binned += Cl_CIB

            mu.append(Cl_TE_binned)

    mu = np.concatenate(mu)
    mu = mu[data_vector_mask]

    r = data_vector - mu
    chi2 = r @ inv_cov @ r

    ln_like = -0.5*chi2
    block[names.data_vector, like_name+"_CHI2"] = chi2
    block[names.likelihoods, like_name+"_LIKE"] = ln_like

    block[names.data_vector, like_name+"_data_vector"] = data_vector
    block[names.data_vector, like_name+"_theory_vector"] = mu
    block[names.data_vector, like_name+"_covariance"] = cov

    return 0


def cleanup(block):
    pass
