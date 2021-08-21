import os


from cosmosis.datablock import option_section, names
from cosmosis.datablock.cosmosis_py import errors


import numpy as np

from CIB_model import CIBModel


def setup(options):
    cov_file = options.get_string(option_section, "cov_file")
    data_files = options.get_string(option_section, "data_file")

    cov = np.loadtxt(cov_file)
    data = np.loadtxt(data_files)

    ell_data = data[:, 0]

    # Normalize the CIB a bit
    scaling_factor = 1e5 * ell_data**2/(2*np.pi)
    Y = scaling_factor[:, None] * data[:, 1:]
    S = np.diag(np.tile(scaling_factor, Y.shape[1]))
    Y_cov = S @ cov @ S
    X = np.log10(ell_data)

    CIB_model = CIBModel(X, Y, Y_cov)

    state_file = options.get_string(option_section, "gp_state_file",
                                    default="")
    if state_file != "":
        print("Loading GP state: ", state_file)
        CIB_model.load_state(state_file=state_file)
    else:
        print("Training GP")
        CIB_model.train(progress_bar=False)

    CIB_model.print_model_parameters()
    print(f"CIB model chi2 {CIB_model.chi2():.1f}")

    CIB_section = options.get_string(option_section, "CIB_section",
                                     default="shear_CIB_cl")
    tSZ_CIB_contamination_section = options.get_string(
                            option_section, "tSZ_CIB_contamination_section",
                            default="shear_y_CIB_contamination_cl")

    return CIB_model, ell_data, CIB_section, tSZ_CIB_contamination_section


def execute(block, config):
    CIB_model, ell_data, CIB_section, tSZ_CIB_contamination_section = config

    alpha_CIB = block["cib_parameters", "alpha"]

    CIB_prediction = CIB_model.predict(np.log10(ell_data))
    # Undo normalisation to get Cl
    CIB_prediction *= 1/(1e5 * ell_data**2/(2*np.pi))[:, None]

    n_ell, n_z = CIB_prediction.shape

    suffix = "_binned"

    block[CIB_section+suffix, "nbin"] = n_z
    block[tSZ_CIB_contamination_section+suffix, "nbin"] = n_z

    block[CIB_section+suffix, "ell"] = ell_data
    block[tSZ_CIB_contamination_section+suffix, "ell"] = ell_data

    for i in range(n_z):
        block[CIB_section+suffix, f"bin_{i+1}_1"] = CIB_prediction[:, i]
        block[tSZ_CIB_contamination_section+suffix, f"bin_{i+1}_1"] = \
            alpha_CIB*CIB_prediction[:, i]

    # Smooth prediction for plotting
    if ("shear_y_cl", "ell") in block:
        ell_smooth = block["shear_y_cl", "ell"]
        CIB_prediction = CIB_model.predict(np.log10(ell_smooth))
        # Undo normalisation to get Cl
        CIB_prediction *= 1/(1e5 * ell_smooth**2/(2*np.pi))[:, None]

        n_ell, n_z = CIB_prediction.shape
        suffix = ""

        block[CIB_section+suffix, "nbin"] = n_z
        block[tSZ_CIB_contamination_section+suffix, "nbin"] = n_z

        block[CIB_section+suffix, "ell"] = ell_smooth
        block[tSZ_CIB_contamination_section+suffix, "ell"] = ell_smooth

        for i in range(n_z):
            block[CIB_section+suffix, f"bin_{i+1}_1"] = CIB_prediction[:, i]
            block[tSZ_CIB_contamination_section+suffix, f"bin_{i+1}_1"] = \
                alpha_CIB*CIB_prediction[:, i]

    return 0


def cleanup(block):
    pass
