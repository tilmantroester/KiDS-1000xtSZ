import argparse
import os

import pymaster as nmt

import healpy

import numpy as np

import sys
sys.path.append("../tools/")

from misc_utils import read_partial_map, file_header


def make_maps(nside, e1, e2, w, idx, rotate=False):
    n_pix = healpy.nside2npix(nside)

    if rotate:
        alpha = np.pi*np.random.rand(len(e1))
        e = np.sqrt(e1**2 + e2**2)
        e1 = np.cos(2.0*alpha)*e
        e2 = np.sin(2.0*alpha)*e

    e1_map = np.bincount(idx, weights=w*e1, minlength=n_pix)
    e2_map = np.bincount(idx, weights=w*e2, minlength=n_pix)
    w_map = np.bincount(idx, weights=w, minlength=n_pix)

    return e1_map, e2_map, w_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output-path", required=True)
    parser.add_argument("--bin-operator", required=True)

    parser.add_argument("--shear-maps", nargs="+")
    parser.add_argument("--shear-masks", nargs="+")
    parser.add_argument("--shear-auto", action="store_true")
    parser.add_argument("--no-cross-shear", action="store_true")

    parser.add_argument("--foreground-map")
    parser.add_argument("--foreground-mask")
    parser.add_argument("--foreground-auto", action="store_true")

    parser.add_argument("--foreground-mask-already-applied",
                        action="store_true")

    parser.add_argument("--pymaster-workspace-output-path")
    parser.add_argument("--pymaster-workspace-input-path")

    parser.add_argument("--compute-covariance", action="store_true")

    parser.add_argument("--randomize-shear", action="store_true")

    parser.add_argument("--n-iter")

    args = parser.parse_args()

    n_iter = 3
    if args.n_iter is not None:
        n_iter = int(args.n_iter)
        print(f"Using n_iter = {n_iter}")

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    if args.pymaster_workspace_output_path is None:
        pymaster_workspace_output_path = output_path
    else:
        pymaster_workspace_output_path = args.pymaster_workspace_output_path
        os.makedirs(pymaster_workspace_output_path, exist_ok=True)

    if args.pymaster_workspace_input_path is None:
        print("Creating workspaces and computing coupling matrices")
        compute_coupling_matrices = True
    else:
        print("Reading existing workspaces from ",
              args.pymaster_workspace_input_path)
        compute_coupling_matrices = False

    if (args.shear_maps is not None
            and len(args.shear_maps) != len(args.shear_masks)):
        raise ValueError("Number of shear masks does not match number of "
                         "shear masks.")

    is_foreground_auto = args.foreground_auto
    is_shear_auto = args.shear_auto
    no_cross_shear = args.no_cross_shear

    if is_foreground_auto and is_shear_auto:
        raise ValueError("Can only compute auto power spectra of either "
                         "foreground or shear.")
    elif is_foreground_auto:
        print("Computing foreground auto spectrum")
    elif is_shear_auto:
        print("Computing shear auto spectra")
        if no_cross_shear:
            print("Ignoring cross-bin shear correlations")

    if args.shear_maps is not None:
        shear_fields = []
        print("Loading shear maps")
        for shear_map_file, mask_file in zip(args.shear_maps,
                                             args.shear_masks):
            print(shear_map_file)
            shear_mask = healpy.read_map(mask_file, verbose=False)
            shear_mask[shear_mask == healpy.UNSEEN] = 0

            nside = healpy.get_nside(shear_mask)

            shear_data = read_partial_map(shear_map_file,
                                          fields=[2, 3], fill_value=0,
                                          scale=[1, 1])

            if args.randomize_shear:
                print("  Randomising shear field")
                alpha = np.pi*np.random.rand(shear_data[0].size)
                e = np.sqrt(shear_data[0]**2 + shear_data[1]**2)
                shear_data[0] = np.cos(2.0*alpha)*e
                shear_data[1] = np.sin(2.0*alpha)*e

            print("  Creating field object")
            field_background = nmt.NmtField(shear_mask,
                                            shear_data,
                                            n_iter=n_iter)

            shear_fields.append(field_background)

    if args.shear_catalogs is not None:
        shear_fields = []
        print("Loading shear catalogs")
        for shear_catalog_file in args.shear_catalogs:
            print(shear_catalog_file)
            data = np.load(shear_catalog_file)

            e1_map, e2_map, w_map = make_maps(nside, -data["e1"], data["e2"],
                                              data["w"], data["pixel_idx"],
                                              rotate=args.randomize_shear)

            print("  Creating field object")
            field_background = nmt.NmtField(w_map,
                                            [e1_map, e2_map],
                                            n_iter=n_iter)

            shear_fields.append(field_background)

    if args.foreground_map is not None:
        print("Loading foreground map")
        print(args.foreground_map)
        if args.foreground_mask_already_applied:
            print("  Mask already applied to map")
            foreground_mask_already_applied = True
        else:
            foreground_mask_already_applied = False

        foreground_map = healpy.read_map(args.foreground_map, verbose=False)
        foreground_map[foreground_map == healpy.UNSEEN] = 0

        foreground_mask = healpy.read_map(args.foreground_mask, verbose=False)
        foreground_mask[foreground_mask == healpy.UNSEEN] = 0

        nside = healpy.get_nside(foreground_map)

        print("  Creating field object")
        foreground_field = nmt.NmtField(
                            foreground_mask,
                            [foreground_map],
                            masked_on_input=foreground_mask_already_applied,
                            n_iter=n_iter)

    if args.bin_operator.find("delta_ell_") == 0:
        delta_ell = int(args.bin_operator[len("delta_ell_"):])
        print("Using linear binning with bin width ", delta_ell)
        nmt_bins = nmt.NmtBin.from_nside_linear(
                        nside=nside,
                        nlb=delta_ell)
    else:
        print("Using binning operator from file ", args.bin_operator)
        binning_operator = np.loadtxt(args.bin_operator)
        ell = np.arange(binning_operator.size)

        nmt_bins = nmt.NmtBin(nside=nside,
                              bpws=binning_operator, ells=ell, weights=2*ell+1)

    nmt_workspaces = {}

    if is_foreground_auto:
        fields_A = [foreground_field]
        fields_B = [foreground_field]
        field_A_tag = "foreground"
        field_B_tag = ""
    elif is_shear_auto:
        fields_A = shear_fields
        fields_B = shear_fields
        field_A_tag = "shear_{idx}"
        field_B_tag = "shear_{idx}"
    else:
        fields_A = [foreground_field]
        fields_B = shear_fields
        field_A_tag = "foreground"
        field_B_tag = "shear_{idx}"

    print("Getting coupling matrices")
    for i, field_A in enumerate(fields_A):
        tag_A = field_A_tag.format(idx=i)
        print("  Field " + tag_A)
        for j, field_B in enumerate(fields_B):
            if is_shear_auto:
                if j > i:
                    continue
                if no_cross_shear and i != j:
                    continue

            tag_B = field_B_tag.format(idx=j)
            print("    Field " + tag_B)

            file_tag = tag_A + "_" + tag_B if tag_B != "" else tag_A

            nmt_workspace = nmt.NmtWorkspace()

            if compute_coupling_matrices:
                nmt_workspace.compute_coupling_matrix(fl1=field_A,
                                                      fl2=field_B,
                                                      bins=nmt_bins,
                                                      is_teb=False,
                                                      n_iter=n_iter)
                nmt_workspaces[(i, j)] = nmt_workspace

                np.save(
                    os.path.join(
                        output_path,
                        f"pymaster_bandpower_windows_{file_tag}.npy"),
                    nmt_workspace.get_bandpower_windows())
                np.save(
                    os.path.join(
                        output_path,
                        f"pymaster_coupling_matrix_{file_tag}.npy"),
                    nmt_workspace.get_coupling_matrix())

                nmt_workspace.write_to(os.path.join(
                                pymaster_workspace_output_path,
                                f"pymaster_workspace_{file_tag}.fits"))

            else:
                nmt_workspace.read_from(os.path.join(
                        args.pymaster_workspace_input_path,
                        f"pymaster_workspace_{file_tag}.fits"))
                nmt_workspaces[(i, j)] = nmt_workspace

    print("Computing Cls")
    Cls_coupled = {}
    Cls_decoupled = {}
    header_columns = {}
    for i, field_A in enumerate(fields_A):
        tag_A = field_A_tag.format(idx=i)
        print("  Field " + tag_A)
        for j, field_B in enumerate(fields_B):
            if is_shear_auto:
                if j > i:
                    continue
                if no_cross_shear and i != j:
                    continue

            tag_B = field_B_tag.format(idx=j)
            print("    Field " + tag_B)
            file_tag = tag_A + "_" + tag_B if tag_B != "" else tag_A
            header_columns[(i, j)] = "Cl_" + file_tag

            Cl_coupled = nmt.compute_coupled_cell(field_A, field_B)
            noise_bias = None
            Cl_decoupled = nmt_workspaces[(i, j)].decouple_cell(
                                                        cl_in=Cl_coupled,
                                                        cl_noise=noise_bias)

            Cls_coupled[(i, j)] = Cl_coupled
            Cls_decoupled[(i, j)] = Cl_decoupled

    ell_nmt = nmt_bins.get_effective_ells()
    header = "ell, " + ", ".join(header_columns.values())
    header = file_header(header_info=header)

    if is_foreground_auto:
        spectra = [("TT", 0)]
    elif is_shear_auto:
        spectra = [("EE", 0), ("EB", 1), ("BE", 2), ("BB", 3)]
    else:
        spectra = [("TE", 0), ("TB", 1)]

    for spectrum, spectrum_idx in spectra:
        Cl_data = [Cl[spectrum_idx] for Cl in Cls_decoupled.values()]
        np.savetxt(os.path.join(output_path, f"Cl_{spectrum}_decoupled.txt"),
                   np.vstack((ell_nmt, *Cl_data)).T,
                   header=header
                   )

        Cl_data = [Cl[spectrum_idx] for Cl in Cls_coupled.values()]
        ell_coupled = np.arange(Cl_data[0].size)
        np.savetxt(os.path.join(output_path, f"Cl_{spectrum}_coupled.txt"),
                   np.vstack((ell_coupled, *Cl_data)).T,
                   header=header
                   )

    if args.compute_covariance:
        print("Computing coupling matrices for Gaussian covariance")

        if not is_foreground_auto and not is_shear_auto:
            for i, shear_field_a in enumerate(shear_fields):
                for j, shear_field_b in enumerate(shear_fields[:i+1]):
                    print(f"  Field {i}-{j}")
                    nmt_cov_workspace = nmt.NmtCovarianceWorkspace()

                    nmt_cov_workspace.compute_coupling_coefficients(
                                                fla1=foreground_field,
                                                fla2=shear_field_a,
                                                flb1=foreground_field,
                                                flb2=shear_field_b)

                    nmt_cov_workspace.write_to(
                        os.path.join(
                            pymaster_workspace_output_path,
                            f"pymaster_cov_workspace_foreground_shear_{i}"
                            f"_foreground_shear_{j}.fits"))

        elif is_shear_auto:
            field_idx = [(i, j) for i in range(len(shear_fields))
                         for j in range(i+1)]

            for i, (idx_a1, idx_a2) in enumerate(field_idx):
                print(f"  A {idx_a1}-{idx_a2}")
                for idx_b1, idx_b2 in field_idx[:i+1]:
                    print(f"    B {idx_b1}-{idx_b2}")
                    nmt_cov_workspace = nmt.NmtCovarianceWorkspace()

                    nmt_cov_workspace.compute_coupling_coefficients(
                                                fla1=shear_fields[idx_a1],
                                                fla2=shear_fields[idx_a2],
                                                flb1=shear_fields[idx_b1],
                                                flb2=shear_fields[idx_b2])

                    nmt_cov_workspace.write_to(
                        os.path.join(
                            pymaster_workspace_output_path,
                            f"pymaster_cov_workspace"
                            f"_shear_{idx_a1}_shear_{idx_a2}"
                            f"_shear_{idx_b1}_shear_{idx_b2}.fits"))
