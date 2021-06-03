import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

from namaster_cosmic_shear_measurements import make_maps as make_maps


def create_nmt_covariance_workspace(idx_a1, idx_a2, idx_b1, idx_b2,
                                    w, w_sigma, noise_terms,
                                    output_path):
    f = []
    name = "pymaster_cov_workspace"
    for is_noise, idx in zip(noise_terms, [idx_a1, idx_a2, idx_b1, idx_b2]):
        if is_noise:
            f.append(w_sigma[idx])
            name += f"_shear_noise_{idx}"
        else:
            f.append(w[idx])
            name += f"_shear_{idx}"
    name += ".fits"

    printflush("Computing ", name)

    nmt_cov_workspace = nmt.NmtCovarianceWorkspace()
    nmt_cov_workspace.compute_coupling_coefficients(
                            fla1=f[0],
                            fla2=f[1],
                            flb1=f[2],
                            flb2=f[3])
    nmt_cov_workspace.write_to(os.path.join(output_path, name))


def create_general_nmt_covariance_workspace(idx_a1, idx_a2, idx_b1, idx_b2,
                                            f_a1, f_a2, f_b1, f_b2,
                                            w,
                                            output_path):
    f = []
    name = "pymaster_cov_workspace"
    for idx, field in zip([idx_a1, idx_a2, idx_b1, idx_b2],
                          [f_a1, f_a2, f_b1, f_b2]):
        name += f"_{field}_{idx}"
        f.append(w[field][idx])
    name += ".fits"

    printflush("Computing ", name)

    nmt_cov_workspace = nmt.NmtCovarianceWorkspace()
    nmt_cov_workspace.compute_coupling_coefficients(
                            fla1=f[0],
                            fla2=f[1],
                            flb1=f[2],
                            flb2=f[3])
    nmt_cov_workspace.write_to(os.path.join(output_path, name))


def printflush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--foreground-mask")
    parser.add_argument("--shear-catalogs", nargs="+")

    parser.add_argument("--pymaster-workspace-output-path")

    parser.add_argument("--map-names", nargs="+")

    parser.add_argument("--probes", nargs="+")

    parser.add_argument("--n-iter")
    parser.add_argument("--nside")

    parser.add_argument("--exact-noise-noise", action="store_true")
    parser.add_argument("--exact-noise-signal", action="store_true")

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    nside = 2048
    if args.nside is not None:
        nside = int(args.nside)
    printflush(f"Using nside = {nside}")

    n_iter = 3
    if args.n_iter is not None:
        n_iter = int(args.n_iter)
    printflush(f"Using n_iter = {n_iter}")

    if args.probes is not None:
        if len(args.probes) != 4:
            raise ValueError("4 probes need to be specified")
        if args.probes == ["shear", "shear", "shear", "shear"]:
            cov_block = "EEEE"
            field_types = ("shear", "shear", "shear", "shear")
        elif args.probes == ["shear", "shear", "shear", "foreground"]:
            cov_block = "EETE"
            field_types = ("shear", "shear", "shear", "foreground")
        elif args.probes == ["shear", "foreground", "shear", "foreground"]:
            cov_block = "TETE"
            field_types = ("shear", "foreground", "shear", "foreground")
        else:
            raise ValueError(f"Probe combination {args.probes} not supported.")
    else:
        cov_block = "EEEE"
        field_types = ("shear", "shear", "shear", "shear")
    
    print("Computing covariance block ", cov_block)

    do_exact_noise_noise = args.exact_noise_noise
    do_exact_noise_signal = args.exact_noise_signal
    if do_exact_noise_noise:
        printflush("Will compute exact noise-noise terms")
    if do_exact_noise_signal:
        printflush("Will compute exact noise-signal terms")

    if (do_exact_noise_noise or do_exact_noise_signal) and cov_block != "EEEE":
        raise ValueError("Exact noise is only supported for EEEE.")

    pymaster_workspace_output_path = args.pymaster_workspace_output_path
    os.makedirs(pymaster_workspace_output_path, exist_ok=True)

    w_fields = {"shear": [], "shear_noise": []}

    if args.foreground_mask is not None:
        printflush("Loading foreground mask ", args.foreground_mask)

        w = healpy.read_map(args.foreground_mask, verbose=False)
        w[w == healpy.UNSEEN] = 0

        printflush("  Creating field objects")
        w_field = nmt.NmtField(w, None, n_iter=n_iter, spin=0)
        w_fields["foreground"] = [w_field]

    if args.shear_catalogs is not None:
        n_shear_field = len(args.shear_catalogs)
        for shear_catalog_file in args.shear_catalogs:
            printflush("Loading shear catalog: ", shear_catalog_file)
            shear_data = np.load(shear_catalog_file)
            e1_map, e2_map, w_map, w2s2_map = make_maps(
                                                nside,
                                                -shear_data["e1"],
                                                shear_data["e2"],
                                                shear_data["w"],
                                                shear_data["pixel_idx"],
                                                rotate=False,
                                                return_w2_sigma2=True)

            printflush("  Creating field objects")
            w_field = nmt.NmtField(w_map, None, n_iter=n_iter, spin=0)
            w_sigma_field = nmt.NmtField(np.sqrt(w2s2_map), None,
                                         n_iter=n_iter, spin=0)
            w_fields["shear"].append(w_field)
            w_fields["shear_noise"].append(w_sigma_field)

    if cov_block == "EEEE":
        field_idx_EE = [(i, j) for i in range(n_shear_field)
                        for j in range(i+1)]
        cov_idx = []
        for i, idx_a in enumerate(field_idx_EE):
            for idx_b in field_idx_EE[:i+1]:
                cov_idx.append((idx_a, idx_b))
    elif cov_block == "EETE":
        field_idx_EE = [(i, j) for i in range(n_shear_field)
                        for j in range(i+1)]
        field_idx_TE = [(i, 0) for i in range(n_shear_field)]
        cov_idx = []
        for idx_a in field_idx_EE:
            for idx_b in field_idx_TE:
                cov_idx.append((idx_a, idx_b))
    elif cov_block == "TETE":
        field_idx_TE = [(i, 0) for i in range(n_shear_field)]
        cov_idx = []
        for i, idx_a in enumerate(field_idx_TE):
            for idx_b in field_idx_TE[:i+1]:
                cov_idx.append((idx_a, idx_b))

    printflush(f"Computing covariance coupling matrices for "
               f"fields {field_types}: {cov_idx}")

    for (idx_a1, idx_a2), (idx_b1, idx_b2) in cov_idx:
        printflush(f"  A {idx_a1}-{idx_a2}, B {idx_b1}-{idx_b2} "
                   f"({field_types})")

        printflush("      Computing signal-signal covariance")
        printflush("      ", end="")
        if args.dry_run:
            continue
        create_general_nmt_covariance_workspace(
                        idx_a1, idx_a2, idx_b1, idx_b2,
                        *field_types,
                        w_fields,
                        pymaster_workspace_output_path)

        if do_exact_noise_signal:
            if idx_a1 == idx_b1:
                printflush("      Computing signal-noise covariance")
                printflush("      ", end="")
                create_nmt_covariance_workspace(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                w_fields["shear"], w_fields["shear_noise"],
                                noise_terms=[True, False, True, False],
                                output_path=pymaster_workspace_output_path)

            if idx_a1 == idx_b2:
                printflush("      Computing signal-noise covariance")
                printflush("      ", end="")
                create_nmt_covariance_workspace(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                w_fields["shear"], w_fields["shear_noise"],
                                noise_terms=[True, False, False, True],
                                output_path=pymaster_workspace_output_path)

            if idx_a2 == idx_b1:
                printflush("      Computing signal-noise covariance")
                printflush("      ", end="")
                create_nmt_covariance_workspace(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                w_fields["shear"], w_fields["shear_noise"],
                                noise_terms=[False, True, True, False],
                                output_path=pymaster_workspace_output_path)

            if idx_a2 == idx_b2:
                printflush("      Computing signal-noise covariance")
                printflush("      ", end="")
                create_nmt_covariance_workspace(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                w_fields["shear"], w_fields["shear_noise"],
                                noise_terms=[False, True, False, True],
                                output_path=pymaster_workspace_output_path)
        if do_exact_noise_noise:
            if ((idx_a1 == idx_b1 and idx_a2 == idx_b2)
                    or (idx_a1 == idx_b2 and idx_a2 == idx_b1)):
                printflush("      Computing noise-noise covariance")
                printflush("      ", end="")
                create_nmt_covariance_workspace(
                                idx_a1, idx_a2, idx_b1, idx_b2,
                                w_fields["shear"], w_fields["shear_noise"],
                                noise_terms=[True, True, True, True],
                                output_path=pymaster_workspace_output_path)
