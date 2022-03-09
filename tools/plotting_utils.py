import matplotlib.pyplot as plt
import matplotlib

import numpy as np


def no_y_ticklabels(ax):
    # From getdist
    ax.tick_params(labelleft=False)
    ax.yaxis.offsetText.set_visible(False)


def field_idx(probe, n_z):
    if probe == "EE":
        field_idx_EE = [(i, j) for i in range(n_z)
                        for j in range(i+1)]
        return field_idx_EE
    elif probe == "TE":
        field_idx_TE = [(i, 0) for i in range(n_z)]
        return field_idx_TE
    else:
        raise ValueError(f"Unsupported probe {probe}")


def plot_Cls(Cls, Cl_idx, ax, x_range=None, scaling=lambda x: x,
             x_offset=lambda x, i: x):
    plot_info = {}

    Cl_plot_idx = 0
    for Cl_specs in Cls:
        c = Cl_specs.copy()

        hidden = c.pop("hidden", False)
        if hidden:
            continue

        name = c.pop("name")
        label = c.pop("label", name)
        x = c.pop("X")
        y = c.pop("Y")
        y_err = c.pop("Y_error", None)
        y_lower, y_upper = c.pop("Y_lower", None), c.pop("Y_upper", None)
        plot_kwargs = c.pop("plot_kwargs", {})

        if y.ndim == 3:
            do_ppd = True
            ppd_percentiles = c.pop("PPD_percentiles", None)
            ppd_color_param = c.pop("PPD_color_param", None)
        else:
            do_ppd = False

        ratio = c.pop("ratio_wrt", None)
        ratio_wrt_error = c.pop("ratio_wrt_error", False)

        if x_range is not None:
            m = (x_range[0] < x) & (x <= x_range[1])
        else:
            m = np.ones_like(x, dtype=bool)
        x = x[m]
        u = scaling(x)
        if x_offset is not None:
            x_plot = x_offset(x, Cl_plot_idx)
        else:
            x_plot = x

        if do_ppd:
            y = y[:, m, Cl_idx]
            if ppd_percentiles is not None:
                if Cl_idx == 0:
                    print("Plotting PPD percentiles")
                for q in ppd_percentiles:
                    lower, upper = np.percentile(y, q, axis=0)
                    ax.fill_between(x_plot, u*lower, u*upper,
                                    label=label, **plot_kwargs)
                    label = None
            else:
                if Cl_idx == 0:
                    print("Plotting PPD samples")
                if ppd_color_param is not None:
                    segments = [np.column_stack([x, u*y_]) for y_ in y]
                    lc = matplotlib.collections.LineCollection(segments,
                                                               label=label,
                                                               **plot_kwargs)
                    lc.set_array(np.asarray(ppd_color_param))

                    plot_info["colorbar"] = lc

                    # add lines to axes and rescale
                    #    Note: adding a collection doesn't autoscalee xlim/ylim
                    ax.add_collection(lc)
                    ax.autoscale()
                else:
                    ax.plot(x_plot, (u*y).T, label=None, **plot_kwargs)
                    ax.plot([], [], label=label, **plot_kwargs)
                label = None
        else:
            y = y[m, Cl_idx]
            if y_err is not None:
                y_err = y_err[m, Cl_idx]
            if y_lower is not None:
                y_lower = y_lower[m, Cl_idx]
            if y_upper is not None:
                y_upper = y_upper[m, Cl_idx]

            if ratio is not None:
                for Cl_specs_other in Cls:
                    if Cl_specs_other["name"] == ratio:
                        x_other = Cl_specs_other["X"][m]
                        y_other = Cl_specs_other["Y"][m, Cl_idx]
                        if ratio_wrt_error:
                            y_err_other = Cl_specs_other["Y_error"][m, Cl_idx]
                        break
                else:
                    raise ValueError(f"Name {ratio} not found, "
                                     "cannot take ratio.")

                delta = y# - y_other
                if not ratio_wrt_error:
                    norm = y_other
                else:
                    norm = y_err_other
                y = delta
                u = 1/norm

            if y_err is not None:
                ax.errorbar(x_plot, u*y, u*y_err, label=label, **plot_kwargs)
            elif y_upper is not None:
                if y_lower is None:
                    y_lower = np.zeros_like(y_upper)
                ax.fill_between(x_plot, u*y_lower, u*y_upper, label=label,
                                **plot_kwargs)
            else:
                ax.plot(x_plot, u*y, label=label, **plot_kwargs)

        Cl_plot_idx += 1

    return plot_info


def plot_cosmic_shear(Cls, n_z_bin, figsize=None, y_label=None, title=None,
                      probe="EE", filename=None,
                      x_offset=lambda x, i: x*1.03**i,
                      x_data_range=None, x_range=None, y_range=None,
                      scaling=lambda x: x, sharey=True,
                      legend_fontsize=8, label_fontsize=6,
                     ):
    figsize = figsize or (8, 8)
    fig, ax = plt.subplots(n_z_bin, n_z_bin, sharex=True, sharey=sharey,
                           figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)

    field_idx_EE = field_idx("EE", n_z_bin)
    for i, idx in enumerate(field_idx_EE):
        ax[idx].axhline(0, c="k", lw=1)

        plot_Cls(Cls, i, ax[idx], x_range=x_data_range, scaling=scaling,
                 x_offset=x_offset)

        ax[idx].set_title(f"S{idx[0]+1}-S{idx[1]+1}", y=0.75, fontsize=legend_fontsize)

        if idx[0] != idx[1]:
            ax[idx[::-1]].axis("off")

    # Apply tick label and offsettext fontsizes to all panels
    [a.tick_params(labelsize=label_fontsize) for a in ax.flat]
    [a.yaxis.offsetText.set_fontsize(label_fontsize) for a in ax.flat]

    [a.set_xlabel(r"$\ell$", fontsize=label_fontsize) for a in ax[-1]]
    if y_label is None:
        [a.set_ylabel(r"$\ell C^{\rm " + probe + r"}_\ell$", fontsize=label_fontsize) for a in ax[:, 0]]
    else:
        [a.set_ylabel(y_label, fontsize=label_fontsize) for a in ax[:, 0]]

    if y_range is not None:
        [a.set_ylim(*y_range) for a in ax.flat]

    if x_range is not None:
        ax[0, 0].set_xlim(*x_range)

    ax[0, 0].set_xscale("log")
    ax[0, 0].legend(loc=2, frameon=False, bbox_to_anchor=(1, 1), fontsize=legend_fontsize)

    if title is not None:
        fig.suptitle(title, y=0.95)

    fig.dpi = 150
    if filename is not None:
        fig.savefig(filename)


def plot_xcorr(Cls, n_z_bin, figsize=None, x_data_range=None, x_range=None, scaling=lambda x: x,
               x_offset=lambda x, i: x*1.03**i, axhline=0,
               y_range=None,
               y_label=None, title=None, units="", probe="TE", field="y",
               colorbar_label=None,
               legend_fontsize=8, label_fontsize=6,
               filename=None, sharey=True):
    figsize = figsize or (8, 4)
    fig, ax = plt.subplots(2, n_z_bin//2+1, sharex=True, sharey=False,
                           figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)

    field_idx_TE = field_idx("TE", n_z_bin)

    for i, idx in enumerate(field_idx_TE):
        col = i % 3
        row = i // 3
        a = ax[row, col]
        if axhline is not None:
            a.axhline(axhline, c="k", lw=1)

        plot_info = plot_Cls(Cls, i, a, x_range=x_data_range, scaling=scaling,
                             x_offset=x_offset)

        a.set_title(f"S{idx[0]+1}-{field}", y=0.8, fontsize=legend_fontsize)
        if y_range is not None:
            a.set_ylim(*y_range)
        if col != 0 and sharey:
            no_y_ticklabels(a)

    for a in np.concatenate((ax[0,3:], ax[1,2:])):
        a.axis("off")

    x_label_pad = -3.0

    # Apply tick label and offsettext fontsizes to all panels
    [a.tick_params(labelsize=label_fontsize) for a in ax.flat]
    [a.yaxis.offsetText.set_fontsize(label_fontsize) for a in ax.flat]

    ax_with_label = np.concatenate((np.atleast_1d(ax[0,2]), ax[1]))
    [a.set_xlabel(r"$\ell$", labelpad=x_label_pad, fontsize=label_fontsize) for a in ax_with_label]
    ax[0, 2].tick_params(labelbottom=True)

    if x_range is not None:
        ax[0, 0].set_xlim(*x_range)

    y_label_pad = -3.0
    if y_label is None:
        y_label = r"$\ell^2/2\pi\ C^{\rm " + probe + r"}_\ell$" + units
    [a.set_ylabel(y_label, fontsize=label_fontsize, labelpad=y_label_pad) for a in ax[:, 0]]

    ax[0, 0].set_xscale("log")
    ax[1, 1].legend(loc=2, frameon=False,
                                        fontsize=legend_fontsize,
                                        bbox_to_anchor=(1, 0.6))

    if "colorbar" in plot_info:
        print("Adding colorbar")
        axcb = fig.colorbar(plot_info["colorbar"], ax=ax[1, 3], location="right", fraction=0.6, anchor=(0.5, 0.0), shrink=2.0)
        colorbar_label = colorbar_label or r"$\log_{10} T_\mathrm{AGN} / \mathrm{K}$"
        axcb.set_label(colorbar_label, fontsize=legend_fontsize)
        axcb.ax.tick_params(labelsize=label_fontsize)

    if title is not None:
        fig.suptitle(title, y=0.95)

    fig.dpi = 300
    if filename is not None:
        fig.savefig(filename)

    return fig, ax



def plot_joint(Cls_EE, Cls_TE, n_z_bin, figsize=None, x_data_range=None,
               x_range=None,
               y_range_EE=None, y_range_TE=None,
               scaling_EE=lambda x: x, scaling_TE=lambda x: x,
               x_offset=lambda x, i: x*1.03**i,
               y_label_EE=None, y_label_TE=None, title=None, units="",
               legend_fontsize=8, label_fontsize=6,
               probe="TE", field="y",
               filename=None, sharey=True):
    if figsize is None:
        figsize = (8, 8*0.9)
    fig, ax = plt.subplots(n_z_bin, n_z_bin+1, sharex=True, sharey=False,
                           figsize=figsize)
    fig.subplots_adjust(hspace=0, wspace=0)

    bin_label_size = legend_fontsize

    field_idx_EE = field_idx("EE", n_z_bin)
    field_idx_TE = field_idx("TE", n_z_bin)

    [a.axis("off") for a in ax.flat]
    for i, idx in enumerate(field_idx_EE):
        a = ax[idx]
        a.axis("on")
        a.axhline(0, c="k", lw=1)

        plot_info = plot_Cls(Cls_EE, i, a, x_range=x_data_range,
                             scaling=scaling_EE, x_offset=x_offset)

        a.set_title(f"S{idx[0]+1}-S{idx[1]+1}", y=0.8, fontsize=bin_label_size)
        if y_range_EE is not None:
            a.set_ylim(*y_range_EE)
        if idx[1] > 0 and sharey:
            no_y_ticklabels(a)

    TE_col_starts = [3, 4]
    ax_TE = np.concatenate((ax[0, TE_col_starts[0]:],
                            ax[1, TE_col_starts[1]:]))
    for i, idx in enumerate(field_idx_TE):
        a = ax_TE[i]
        a.axis("on")
        a.axhline(0, c="k", lw=1)

        plot_info = plot_Cls(Cls_TE, i, a, x_range=x_data_range,
                             scaling=scaling_TE, x_offset=x_offset)

        a.set_title(f"S{idx[0]+1}-$y$", y=0.8, fontsize=bin_label_size)
        if y_range_TE is not None:
            a.set_ylim(*y_range_TE)
        if i != 0 and i != 3 and sharey:
            no_y_ticklabels(a)

    x_label_pad = -3.0

    # Apply tick label and offsettext fontsizes to all panels
    [a.tick_params(labelsize=label_fontsize) for a in ax.flat]
    [a.yaxis.offsetText.set_fontsize(label_fontsize) for a in ax.flat]

    [a.set_xlabel(r"$\ell$", labelpad=x_label_pad, fontsize=label_fontsize) for a in ax[-1]]

    if y_label_EE is not None:
        [a.set_ylabel(y_label_EE, fontsize=label_fontsize) for a in ax[:, 0]]

    if x_range is not None:
        ax[0, 0].set_xlim(*x_range)

    ax[0, TE_col_starts[0]].set_xlabel(r"$\ell$", labelpad=x_label_pad, fontsize=label_fontsize)
    ax[0, TE_col_starts[0]].tick_params(labelbottom=True)

    [a.set_xlabel(r"$\ell$", labelpad=x_label_pad, fontsize=label_fontsize) for a in ax[1, TE_col_starts[1]:]]
    [a.tick_params(labelbottom=True) for a in ax[1, TE_col_starts[1]:]]
    [a.yaxis.offsetText.set_visible(True) for a in ax[1, TE_col_starts[1]:]]

    # [a.set_xlabel(r"$\ell$", labelpad=0.0) for a in ax[1, TE_col_starts[1]:]]
    if y_label_TE is not None:
        [a.set_ylabel(y_label_TE, fontsize=label_fontsize) for a in [ax[0, TE_col_starts[0]],
                                                                     ax[1, TE_col_starts[1]]]]

    if "colorbar" in plot_info:
        print("Adding colorbar")
        axcb = fig.colorbar(plot_info["colorbar"], ax=ax[2, 4], location="bottom", fraction=0.6, anchor=(0.0, 0.5), shrink=2.0)
        axcb.set_label(r"$\log_{10} T_\mathrm{AGN} / \mathrm{K}$", fontsize=legend_fontsize)
        axcb.ax.tick_params(labelsize=label_fontsize)

    ax[0, 0].set_xscale("log")
    ax[3, 3].legend(loc=2, frameon=False, bbox_to_anchor=(1, 1), fontsize=legend_fontsize)

    if title is not None:
        fig.suptitle(title, y=0.95)

    fig.dpi = 150
    if filename is not None:
        fig.savefig(filename)
