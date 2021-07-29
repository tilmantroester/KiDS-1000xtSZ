# Reduce the PolSpice CIB measurement and make GP plots

import os

import numpy as np

import sys
sys.path.append("../tools/")
from misc_utils import create_beam_operator, file_header
from CIB_model import CIBModel

# KCAP_PATH = "../../KiDS/kcap/"
# sys.path.append(os.path.join(KCAP_PATH, "kcap"))
# import cosmosis_utils


PI = np.pi


def plot_Cls(Cls, xlabel=r"$\ell$", ylabel=r"$\ell^2/2\pi\ C_\ell$",
             z_cuts=None, scaling=lambda ell: ell**2/(2*PI),
             xscale="log", yscale="linear",
             ylim=None, title="", filename=""):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(8, 5))
    fig.subplots_adjust(hspace=0, wspace=0)

    if z_cuts is None:
        z_cuts = [(0.1, 0.3),
                  (0.3, 0.5),
                  (0.5, 0.7),
                  (0.7, 0.9),
                  (0.9, 1.2),
                  ]
    for i, z_cut in enumerate(z_cuts):
        ax.flatten()[i].axhline(0, c="k", lw=1)
        
        for Cl_spec in Cls:
            c = Cl_spec.copy()
            Y = c.pop("Y")
            X = c.pop("X")
            n_ell_bin = X.size
            u = scaling(X)
            if Y.ndim == 1 and len(z_cuts) > 1:
                Y = Y[n_ell_bin*i:n_ell_bin*(i+1)]
            elif Y.ndim == 2:
                Y = Y[i]

            Y_err = c.pop("Y_err", None)
            Y_lower = c.pop("Y_lower", None)
            Y_upper = c.pop("Y_upper", None)
            if Y_err is not None:
                if Y_err.ndim == 1 and len(z_cuts) > 1:
                    Y_err = Y_err[n_ell_bin*i:n_ell_bin*(i+1)]
                elif Y_err.ndim == 2:
                    Y_err = Y_err[i]
                ax.flatten()[i].errorbar(X, u*Y, u*Y_err, **c)
            elif Y_lower is not None and Y_upper is not None:
                if Y_lower.ndim == 1 and len(z_cuts) > 1:
                    Y_lower = Y_lower[n_ell_bin*i:n_ell_bin*(i+1)]
                    Y_upper = Y_upper[n_ell_bin*i:n_ell_bin*(i+1)]
                elif Y_lower.ndim == 2:
                    Y_lower = Y_lower[i]
                    Y_upper = Y_upper[i]
                label_CI = c.pop("label_CI", None)
                label_mean = c.pop("label", None)
                color = c.pop("c", None)
                ax.flatten()[i].fill_between(X, u*Y_lower, u*Y_upper, **c,
                                             label=label_CI, facecolor=color)
                ax.flatten()[i].plot(X, u*Y, **c, c=color, label=label_mean)
            else:
                ax.flatten()[i].plot(X, u*Y, **c)

        ax.flatten()[i].set_xlabel(xlabel)
        ax.flatten()[i].set_title(f"z: {z_cut[0]}-{z_cut[1]}", x=0.25, y=0.85)

    ax[0,0].set_xscale(xscale)
    ax[0,0].set_yscale(yscale)

    [p[0].set_ylabel(ylabel) for p in ax]
    ax.flatten()[-1].axis("off")
    ax.flatten()[-2].legend(frameon=False, loc="upper left",
                            bbox_to_anchor=(1, 0.8))

    if ylim is not None:
        ax[0,0].set_ylim(**ylim)

    fig.suptitle(title)
    fig.dpi = 300
    if filename != "":
        fig.savefig(filename) 

if __name__ == "__main__":

    mode = "namaster"

    if mode == "namaster":
        compute_prediction_for_cov = False
        make_plots = True

        probe = "TE"
        field = "545GHz_CIB"
        label = f"KiDS-1000 x 545 GHz CIB, {probe}"
        units = r"[mJy]"

        # field = "100GHz_HFI"
        # probe = "TE"
        # label = f"KiDS-1000 x 100 GHz HFI, {probe}"
        # units = r"[$\mathrm{K}_\mathrm{CMB}$]"

        data_file = f"../results/measurements/shear_KiDS1000_{field}/likelihood/data/Cl_{probe}_shear_KiDS1000_gal_{field}.txt"
        cov_file = f"../results/measurements/shear_KiDS1000_{field}/likelihood/cov/covariance_gaussian_nka_{probe}{probe}.txt"

        data = np.loadtxt(data_file)
        ell_data = data[:, 0]
        data = data[:, 1:]

        cov = np.loadtxt(cov_file)

    elif mode == "polspice":
        import base_config

        defaults = {**base_config.PATHS, **base_config.DATA_DIRS}

        # binning_operator = np.loadtxt(
        #                         cosmosis_utils.emulate_configparser_interpolation(
        #                             base_config.bin_op_file, defaults))
        binning_operator = np.loadtxt("../data/xcorr/bin_operator_log_n_bin_12_ell_51-2952.txt")

        OLD_MEASUREMENT_DIR = "../../project-triad-obsolete/results/measurements/"

        target_beam = 10.0

        # CIB_maps = ["353", "545", "857"]
        CIB_maps = ["545"]

        z_cuts = [(0.1, 0.3),
                (0.3, 0.5),
                (0.5, 0.7),
                (0.7, 0.9),
                (0.9, 1.2),
                ]

        Cl_CIB_shear = {z: {c: {} for c in CIB_maps} for z in z_cuts}

        for z_cut in z_cuts:
            for CIB_map in CIB_maps:
                ell, TE, TB = np.loadtxt(
                                os.path.join(OLD_MEASUREMENT_DIR,
                                            f"shear_KiDS1000_CIB/"
                                            f"z{z_cut[0]:.1f}-{z_cut[1]:.1f}-Planck-{CIB_map}/spice.cl"),
                                unpack=True, usecols=[0, 7, 8])

                beam_operator = create_beam_operator(ell, 
                                                    fwhm_map=5.0,
                                                    fwhm_target=target_beam)

                Cl_CIB_shear[z_cut][CIB_map] = {"ell_raw"             : ell,
                                                "Cl_TE_raw"           : TE,
                                                "Cl_TB_raw"           : TB,
                                                "Cl_TE_raw_beam10"    : beam_operator @ TE,
                                                "Cl_TB_raw_beam10"    : beam_operator @ TB,
                                                "ell_binned"          : binning_operator @ ell,
                                                "Cl_TE_binned"        : binning_operator @ TE,
                                                "Cl_TB_binned"        : binning_operator @ TB,
                                                "Cl_TE_beam_binned" : binning_operator @ beam_operator @ TE,
                                                "Cl_TB_beam_binned" : binning_operator @ beam_operator @ TB}

        CIB_jk_data = np.load(
                        os.path.join(OLD_MEASUREMENT_DIR,
                                    f"shear_KiDS1000_CIB/shear_KiDS1000_CIB_jk_data.npz"),
                        allow_pickle=True)

        jk_resolutions = [64,]# 128, 256]

        cov_CIB_shear = {c: {j: {"TE": None, "TB": None} for j in jk_resolutions}
                        for c in CIB_maps}

        ell = np.arange(3001)

        for CIB_map in CIB_maps:
            for jk_res in jk_resolutions:
                d_CIB = []
                for z_cut in z_cuts:
                    tag = f"z{z_cut[0]:.1f}-{z_cut[1]:.1f}-Planck-{CIB_map}"
                    Cl = CIB_jk_data[tag][str(jk_res)]
                    binned = np.einsum("ij,kjl->lik", binning_operator @ beam_operator, Cl)
                    d_CIB.append(binned)

                d = np.concatenate(d_CIB, axis=1)
                print(d.shape)
                n_jk = d.shape[-1]
                effective_n_jk = n_jk

                cov_CIB_shear[CIB_map][jk_res]["TE"] = (
                    np.cov(d[0], ddof=1) * (effective_n_jk-1)**2/effective_n_jk)
                cov_CIB_shear[CIB_map][jk_res]["TB"] = (
                    np.cov(d[1], ddof=1) * (effective_n_jk-1)**2/effective_n_jk)

        # Save files

        header = file_header(
                    f"ell TE ({CIB_map} x KiDS-1000, beam {target_beam}' "
                    f"FWHM, z-bins: {', '.join([str(z) for z in z_cuts])})")

        for CIB_map in CIB_maps:
            data = [Cl_CIB_shear[z_cuts[0]][CIB_map]["ell_binned"]]
            for z_cut in z_cuts:
                data.append(Cl_CIB_shear[z_cut][CIB_map]["Cl_TE_beam_binned"])

            np.savetxt(f"../data/xcorr/CIB/"
                    f"shear_CIB_KiDS1000_{CIB_map}_TE_beam{target_beam}.txt",
                    np.array(data).T,
                    header=header)

        header = file_header(
                    f"TE covariance ({CIB_map} x KiDS-1000, beam {target_beam}' "
                    f"FWHM, z-bins: {', '.join([str(z) for z in z_cuts])})")

        for CIB_map in CIB_maps:
            for name, cov in [("jk_3.4deg2", cov_CIB_shear[CIB_map][64]["TE"]),
                            #("jk_13.4deg2", cov_CIB_shear[CIB_map][128]["TE"]),
                            #("jk_53.7deg2", cov_CIB_shear[CIB_map][256]["TE"])
                            ]:
                np.savetxt(
                    f"../data/xcorr/CIB/"
                    f"shear_CIB_KiDS1000_{CIB_map}_TE_{name}"
                    f"_beam{target_beam}_cov.txt",
                    cov,
                    header=header)

        CIB_map = "545"
        ell_data = Cl_CIB_shear[z_cuts[0]][CIB_map]["ell_binned"]
        data = []
        for z_cut in z_cuts:
            data.append(Cl_CIB_shear[z_cut][CIB_map]["Cl_TE_beam_binned"])
        data = np.array(data).T

        cov = cov_CIB_shear[CIB_map][64]["TE"]

    check_GP = True
    if check_GP:
        # Normalize the CIB a bit
        scaling_factor = 1e5 * ell_data**2/(2*np.pi)
        Y = scaling_factor[:, None] * data
        S = np.diag(np.tile(scaling_factor, Y.shape[1]))

        Y_cov = S @ cov @ S
        X = np.log10(ell_data)

        CIB_model = CIBModel(X, Y, Y_cov)

        # CIB_model.load_state(f"../results/measurements/shear_KiDS1000_{field}/GP_model/GP_state")
        CIB_model.train(n_step=5000, lr=1e-2)
        CIB_model.print_model_parameters()
        print("Chi2:", CIB_model.chi2())

        CIB_model.save_state(f"../results/measurements/shear_KiDS1000_{field}/GP_model/GP_state.torchstate")

        # ell_pred = np.geomspace(51, 3000, 100)
        ell_pred = np.arange(51, 2953)
        CIB_prediction, CI = CIB_model.predict(np.log10(ell_pred), CI=True)
        # Undo normalisation to get Cl
        CIB_prediction *= 1/(1e5 * ell_pred**2/(2*np.pi))[:, None]
        CIB_prediction_CI_l = CI[0] * 1/(1e5 * ell_pred**2/(2*np.pi))[:, None]
        CIB_prediction_CI_u = CI[1] * 1/(1e5 * ell_pred**2/(2*np.pi))[:, None]

        n_ell, n_z = CIB_prediction.shape

        if make_plots:
            plot_Cls(Cls=[{"X": ell_data, "Y": data.T,
                        "Y_err": np.sqrt(np.diag(cov)),
                        "marker": "o", "ls": "none", "c": "C0",
                        "label": label},
                        {"X": ell_pred, "Y": CIB_prediction.T,
                        "Y_lower": CIB_prediction_CI_l.T,
                        "Y_upper": CIB_prediction_CI_u.T,
                        "alpha": 0.5,  "c": "C1",
                        "label": "GP"}],
                    ylabel=r"$\ell^2/2\pi\ C_\ell$ " + units,
                    title=f"{label}, GP model",
                    filename=f"../notebooks/plots/CIB_model_{field}_{probe}_beam10.png")


        if compute_prediction_for_cov:
            ell_pred = np.arange(51, 2953)
            CIB_prediction = CIB_model.predict(np.log10(ell_pred))
            # Undo normalisation to get Cl
            CIB_prediction *= 1/(1e5 * ell_pred**2/(2*np.pi))[:, None]

            ell = np.arange(3*2048)
            for i in range(n_z):
                Cl = np.zeros((2, ell.size))
                Cl[0, 51:2953] = CIB_prediction[:, i]

                np.savez(f"../results/measurements/shear_KiDS1000_{field}/cov_Cls/Cl_cov_GP_{i}-0.npz", ell=ell, Cl_cov=Cl, Cl_noise_cov=np.zeros_like(Cl))
