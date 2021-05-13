import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

from namaster_shear_randoms_measurements import make_maps as make_maps


def create_nmt_covariance_workspace(idx_a1, idx_a2, idx_b1, idx_b2,
                                    w, w_sigma, noise_terms):
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
    nmt_cov_workspace.write_to(os.path.join(
                                    pymaster_workspace_output_path, name))


def printflush(*args, **kwargs):
    print(*args, **kwargs, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--shear-w-maps", nargs="+")
    parser.add_argument("--shear-w2e2-maps", nargs="+")
    parser.add_argument("--shear-catalogs", nargs="+")

    parser.add_argument("--pymaster-workspace-output-path")

    parser.add_argument("--map-names", nargs="+")

    parser.add_argument("--n-iter")
    parser.add_argument("--nside")

    parser.add_argument("--exact-noise-noise", action="store_true")
    parser.add_argument("--exact-noise-signal", action="store_true")

    args = parser.parse_args()

    nside = 2048
    if args.nside is not None:
        nside = int(args.nside)
    printflush(f"Using nside = {nside}")

    n_iter = 3
    if args.n_iter is not None:
        n_iter = int(args.n_iter)
    printflush(f"Using n_iter = {n_iter}")

    do_exact_noise_noise = args.exact_noise_noise
    do_exact_noise_signal = args.exact_noise_signal
    if do_exact_noise_noise:
        printflush("Will compute exact noise-noise terms")
    if do_exact_noise_signal:
        printflush("Will compute exact noise-signal terms")

    pymaster_workspace_output_path = args.pymaster_workspace_output_path
    os.makedirs(pymaster_workspace_output_path, exist_ok=True)

    w_fields = []
    w_sigma_fields = []

    if args.shear_w_maps is not None:
        printflush("Loading maps")
        for weight_map_file, w2e2_map_file in zip(args.shear_w_maps,
                                                  args.shear_w2e2_maps):
            printflush(weight_map_file)
            w = healpy.read_map(weight_map_file, verbose=False)
            w[w == healpy.UNSEEN] = 0

            printflush(w2e2_map_file)
            w2e2 = healpy.read_map(w2e2_map_file, verbose=False)
            w2e2[w2e2 == healpy.UNSEEN] = 0

            nside = healpy.get_nside(w2e2)

            printflush("  Creating field objects")
            w_field = nmt.NmtField(w, None, n_iter=n_iter, spin=0)
            w_sigma_field = nmt.NmtField(np.sqrt(w2e2), None,
                                         n_iter=n_iter, spin=0)

            w_fields.append(w_field)
            w_sigma_fields.append(w_sigma_field)

    if args.shear_catalogs is not None:
        for shear_catalog_file in args.shear_catalogs:
            printflush("  Loading shear catalog: ", shear_catalog_file)
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
            w_fields.append(w_field)
            w_sigma_fields.append(w_sigma_field)

    field_idx = [(i, j) for i in range(len(w_sigma_fields))
                 for j in range(i+1)]

    printflush("Computing covariance coupling matrices for fields ", field_idx)

    for i, (idx_a1, idx_a2) in enumerate(field_idx):
        printflush(f"  A {idx_a1}-{idx_a2}")
        for idx_b1, idx_b2 in field_idx[:i+1]:
            printflush(f"    B {idx_b1}-{idx_b2}")

            printflush("      Computing signal-signal covariance")
            printflush("      ", end="")
            create_nmt_covariance_workspace(
                            idx_a1, idx_a2, idx_b1, idx_b2,
                            w_fields, w_sigma_fields,
                            noise_terms=[False, False, False, False])

            if do_exact_noise_signal:
                if idx_a1 == idx_b1:
                    printflush("      Computing signal-noise covariance")
                    printflush("      ", end="")
                    create_nmt_covariance_workspace(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    w_fields, w_sigma_fields,
                                    noise_terms=[True, False, True, False])

                if idx_a1 == idx_b2:
                    printflush("      Computing signal-noise covariance")
                    printflush("      ", end="")
                    create_nmt_covariance_workspace(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    w_fields, w_sigma_fields,
                                    noise_terms=[True, False, False, True])

                if idx_a2 == idx_b1:
                    printflush("      Computing signal-noise covariance")
                    printflush("      ", end="")
                    create_nmt_covariance_workspace(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    w_fields, w_sigma_fields,
                                    noise_terms=[False, True, True, False])

                if idx_a2 == idx_b2:
                    printflush("      Computing signal-noise covariance")
                    printflush("      ", end="")
                    create_nmt_covariance_workspace(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    w_fields, w_sigma_fields,
                                    noise_terms=[False, True, False, True])
            if do_exact_noise_noise:
                if ((idx_a1 == idx_b1 and idx_a2 == idx_b2)
                        or (idx_a1 == idx_b2 and idx_a2 == idx_b1)):
                    printflush("      Computing noise-noise covariance")
                    printflush("      ", end="")
                    create_nmt_covariance_workspace(
                                    idx_a1, idx_a2, idx_b1, idx_b2,
                                    w_fields, w_sigma_fields,
                                    noise_terms=[True, True, True, True])
